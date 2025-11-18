import sounddevice as sd
import numpy as np
import torch
import queue
import threading
import time
from panns_inference.models import Cnn14
from tuning import Tuning
import usb.core
import usb.util
import json
import websocket  # 🔹 websocket-client

# -----------------------------
# 설정
# -----------------------------
SAMPLE_RATE = 32000
PRE_BUFFER_DURATION = 0.3
DETECT_DURATION = 0.3
TARGET_CHANNEL = 0
DEVICE_INDEX = 1
MIN_WAVEFORM_LENGTH = SAMPLE_RATE
CONF_THRESH = 0.3

LABEL_ORDER = ["/m/09x0r", "/m/05tny_", "/m/0bt9lr",
               "/m/0912c9", "/m/014zdl", "/m/07yv9", "/m/03kmc9"]

BASE_CKPT_PATH = r"C:\Users\juuip\panns_data\Cnn14_mAP=0.431.pth"
FINETUNED_CKPT = r"C:\OpenSourcePJ\server\best_panns6_acc0.828.pt"

NODE_WS_URL = "ws://localhost:8080"   # 🔹 같은 PC에서 Node 서버로 붙음


# -----------------------------
# ReSpeaker 장치 초기화
# -----------------------------
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
if not dev:
    raise RuntimeError("❌ ReSpeaker 장치를 찾을 수 없습니다.")
Mic_tuning = Tuning(dev)


# -----------------------------
# PANNs6Head
# -----------------------------
class PANNs6Head(torch.nn.Module):
    def __init__(self, checkpoint_path: str, num_classes: int):
        super().__init__()
        self.backbone = Cnn14(
            sample_rate=SAMPLE_RATE,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        self.backbone.load_state_dict(state, strict=False)

        in_dim = self.backbone.fc_audioset.in_features
        self.head = torch.nn.Linear(in_dim, num_classes)

    def forward(self, wav):
        out = self.backbone(wav)
        emb = out["embedding"]
        logits = self.head(emb)
        return {"logits": logits}


# -----------------------------
#  파인튜닝 가중치 로드
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device:", device)

audio_model = PANNs6Head(BASE_CKPT_PATH, num_classes=len(LABEL_ORDER)).to(device)

ckpt = torch.load(FINETUNED_CKPT, map_location=device)
LABEL_NAMES = ckpt.get(
    "label_names",
    ["Speech", "Bark", "Dog", "Vehicle horn", "Explosion", "Vehicle", "Siren"]
)
print("[INFO] LABEL_NAMES:", LABEL_NAMES)

audio_model.load_state_dict(ckpt["state_dict"], strict=False)
audio_model.eval()

# -----------------------------
#  WebSocket 클라이언트 (파이썬 → Node)
# -----------------------------
ws_lock = threading.Lock()
ws_conn = None
ws_should_run = True


def ws_connect_loop():
    """Node 서버와 WebSocket 연결 유지 (끊기면 자동 재연결)"""
    global ws_conn
    while ws_should_run:
        try:
            print(f"[WS] Connecting to {NODE_WS_URL} ...")
            ws = websocket.create_connection(NODE_WS_URL)
            with ws_lock:
                ws_conn = ws
            print("[WS] Connected")

            # 서버에서 오는 메세지를 굳이 안 봐도 되면 그냥 블록 없이 sleep 루프
            while ws_should_run:
                # 서버 ping/ack 무시하고 그냥 살아 있게만 둠
                time.sleep(1)

        except Exception as e:
            print("[WS] connection error:", e)
            with ws_lock:
                ws_conn = None
            time.sleep(3)  # 3초 후 재시도


def send_detection_to_ws(payload: dict):
    """detection 결과를 Node 서버로 JSON 전송"""
    global ws_conn
    msg = json.dumps({"type": "detection", **payload}, ensure_ascii=False)
    with ws_lock:
        if ws_conn is None:
            return
        try:
            ws_conn.send(msg)
        except Exception as e:
            print("[WS] send error:", e)
            ws_conn = None


# -----------------------------
#  큐 및 버퍼
# -----------------------------
audio_queue = queue.Queue()
pre_buffer_size = int(SAMPLE_RATE * PRE_BUFFER_DURATION)
pre_buffer = np.zeros(pre_buffer_size, dtype=np.float32)


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata[:, TARGET_CHANNEL].astype(np.float32))


def audio_collector():
    global pre_buffer
    while True:
        chunk = audio_queue.get()
        if len(chunk) >= pre_buffer_size:
            pre_buffer = chunk[-pre_buffer_size:]
        else:
            pre_buffer = np.roll(pre_buffer, -len(chunk))
            pre_buffer[-len(chunk):] = chunk


# -----------------------------
#  감지 + 분석 쓰레드
# -----------------------------
def audio_detector_analyzer():
    while True:
        try:
            if Mic_tuning.is_voice():
                doa = Mic_tuning.direction

                detect_size = int(SAMPLE_RATE * DETECT_DURATION)
                collected = []
                while len(collected) < detect_size:
                    chunk = audio_queue.get()
                    collected.extend(chunk)
                collected = np.array(collected[:detect_size], dtype=np.float32)

                waveform = np.concatenate([pre_buffer, collected])

                if len(waveform) < MIN_WAVEFORM_LENGTH:
                    waveform = np.pad(waveform, (0, MIN_WAVEFORM_LENGTH - len(waveform)))

                waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    out = audio_model(waveform_tensor)
                    logits = out["logits"][0]
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                scores = probs.tolist()
                pairs = [(LABEL_NAMES[i], scores[i]) for i in range(len(LABEL_NAMES))]

                # Dog / Bark 통합
                dog_score = next((s for l, s in pairs if l == "Dog"), 0.0)
                bark_score = next((s for l, s in pairs if l == "Bark"), 0.0)
                if (dog_score > 0.0) or (bark_score > 0.0):
                    combined_score = dog_score + bark_score
                    dog_label = "Dog" if dog_score >= bark_score else "Dog (Barking)"
                    pairs = [(l, s) for (l, s) in pairs if l not in ("Dog", "Bark")]
                    pairs.append((dog_label, combined_score))

                # Vehicle / Vehicle horn 통합
                vehicle_score = next((s for l, s in pairs if l == "Vehicle"), 0.0)
                horn_score = next((s for l, s in pairs if l == "Vehicle horn"), 0.0)
                if (vehicle_score > 0.0) or (horn_score > 0.0):
                    combined_score = vehicle_score + horn_score
                    vehicle_label = "Vehicle" if vehicle_score >= horn_score else "Vehicle (Horn)"
                    pairs = [(l, s) for (l, s) in pairs if l not in ("Vehicle", "Vehicle horn")]
                    pairs.append((vehicle_label, combined_score))

                pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
                top3 = pairs_sorted[:3]
                top1_score = top3[0][1] if top3 else 0.0

                if top1_score >= CONF_THRESH:
                    timestamp = time.strftime("%H:%M:%S")
                    output = {
                        "timestamp": timestamp,
                        "doa": doa,
                        "tags": [
                            {"label": label, "score": round(score, 3)}
                            for (label, score) in top3
                        ],
                    }

                    # 콘솔에도 찍고
                    print(json.dumps(output, ensure_ascii=False, indent=2))

                    # 🔹 Node 서버로 전송
                    send_detection_to_ws(output)

                time.sleep(DETECT_DURATION)
            else:
                time.sleep(0.01)
        except KeyboardInterrupt:
            break


# -----------------------------
#  시작 코드
# -----------------------------
if __name__ == "__main__":
    # WS 클라이언트 스레드 시작
    ws_thread = threading.Thread(target=ws_connect_loop, daemon=True)
    ws_thread.start()

    # 오디오 스트리밍 + 감지 스레드 시작
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=6,
        device=DEVICE_INDEX,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1),
    )
    collector_thread = threading.Thread(target=audio_collector, daemon=True)
    detector_thread = threading.Thread(target=audio_detector_analyzer, daemon=True)

    collector_thread.start()
    detector_thread.start()

    print("🎤 멀티쓰레드 기반 실시간 감지 + 분석 + WS 전송 시작...\n(CTRL+C로 종료)\n")

    with stream:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 실시간 분석 종료")
