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
import websocket  # websocket-client
import librosa    # 리샘플링용
import os

# -----------------------------
# 설정
# -----------------------------
# 입력(마이크) 샘플레이트와 모델 샘플레이트를 분리
MIC_SAMPLE_RATE = int(os.getenv("MIC_SAMPLE_RATE", "16000"))      # 예: GUI에서 설정
MODEL_SAMPLE_RATE = int(os.getenv("MODEL_SAMPLE_RATE", "32000"))  # 필요하면 GUI에서 바꿀 수도 있음

PRE_BUFFER_DURATION = float(os.getenv("PRE_BUFFER_DURATION", "0.2"))
DETECT_DURATION = float(os.getenv("DETECT_DURATION", "0.5"))

TARGET_CHANNEL = int(os.getenv("TARGET_CHANNEL", "0"))
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "1"))
MIN_WAVEFORM_LENGTH = MODEL_SAMPLE_RATE   # 최소 1초 @ MODEL_SAMPLE_RATE

CONF_THRESH = float(os.getenv("CONF_THRESH", "0.5"))

LABEL_ORDER = [
    "/m/09x0r", "/m/05tny_", "/m/0bt9lr",
    "/m/0912c9", "/m/014zdl", "/m/07yv9", "/m/03kmc9"
]

BASE_CKPT_PATH = os.getenv(
    "BASE_CKPT_PATH",
    r"C:\Users\juuip\panns_data\Cnn14_mAP=0.431.pth"
)
FINETUNED_CKPT = r"C:\OpenSourcePJ\server\best_panns6_acc0.828.pt"

# Node WS URL도 환경변수로
FINETUNED_CKPT = os.getenv(
    "FINETUNED_CKPT",
    r"C:\OpenSourcePJ\server\best_panns6_acc0.828.pt"
)

NODE_WS_URL = os.getenv("NODE_WS_URL", "ws://localhost:8080")

# -----------------------------
# ReSpeaker 장치 초기화
# -----------------------------
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
if not dev:
    raise RuntimeError("ReSpeaker 장치를 찾을 수 없습니다.")
Mic_tuning = Tuning(dev)


# -----------------------------
# PANNs6Head 정의
# -----------------------------
class PANNs6Head(torch.nn.Module):
    def __init__(self, checkpoint_path: str, num_classes: int):
        super().__init__()
        self.backbone = Cnn14(
            sample_rate=MODEL_SAMPLE_RATE,   # 모델은 32kHz 기준
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

            # 서버에서 오는 메세지는 안 보고 유지만
            while ws_should_run:
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
#  큐 및 버퍼 (입력 레이트 기준)
# -----------------------------
audio_queue = queue.Queue()
pre_buffer_size = int(MIC_SAMPLE_RATE * PRE_BUFFER_DURATION)
pre_buffer = np.zeros(pre_buffer_size, dtype=np.float32)


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    # indata: [frames, channels], MIC_SAMPLE_RATE 기준
    audio_queue.put(indata[:, TARGET_CHANNEL].astype(np.float32))


def audio_collector():
    """실시간으로 pre_buffer(과거 0.2초) 유지"""
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
    """Mic_tuning.is_voice()가 true면 감지 구간 + pre_buffer 묶어서 모델 추론"""
    while True:
        try:
            if Mic_tuning.is_voice():
                doa = Mic_tuning.direction

                # 🔹 detect_size: 입력 레이트 기준 0.5초
                detect_size = int(MIC_SAMPLE_RATE * DETECT_DURATION)
                collected = []
                while len(collected) < detect_size:
                    chunk = audio_queue.get()
                    collected.extend(chunk)
                collected = np.array(collected[:detect_size], dtype=np.float32)

                # 🔹 pre_buffer(0.2초) + 현재(0.5초) = 총 0.7초 @ MIC_SAMPLE_RATE
                waveform_in = np.concatenate([pre_buffer, collected])  # @ MIC_SAMPLE_RATE

                # 🔹 여기서 32kHz로 리샘플링
                waveform_32k = librosa.resample(
                    waveform_in,
                    orig_sr=MIC_SAMPLE_RATE,
                    target_sr=MODEL_SAMPLE_RATE
                )

                # 🔹 최소 1초(32000샘플) 보장 (부족하면 뒤에 0 패딩)
                if len(waveform_32k) < MIN_WAVEFORM_LENGTH:
                    pad_len = MIN_WAVEFORM_LENGTH - len(waveform_32k)
                    waveform_32k = np.pad(waveform_32k, (0, pad_len))

                waveform_tensor = (
                    torch.from_numpy(waveform_32k)
                    .float()
                    .unsqueeze(0)     # [1, T]
                    .to(device)
                )

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
                    dog_label = "Dog" # if dog_score >= bark_score else "Dog (Barking)"
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

                # 정렬 후 top1만 사용
                pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
                if not pairs_sorted:
                    time.sleep(0.01)
                    continue

                top1_label, top1_score = pairs_sorted[0]

                if top1_score >= CONF_THRESH:
                    output = {
                        "type": "detection",
                        "timestamp": time.time_ns() // 1_000_000,
                        "doa": doa,
                        "tags": [
                            {
                                "label": top1_label,
                                "score": round(float(top1_score), 2)
                            }
                        ]
                    }

                    # 콘솔 테스트용으로 보고 싶으면 이거 풀면 됨
                    # print(json.dumps(output, ensure_ascii=False, indent=2))

                    # Node 서버로 전송
                    send_detection_to_ws(output)

                # 너무 자주 연속 추론되는 거 막고 싶으면 감지 구간만큼 sleep
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
        samplerate=MIC_SAMPLE_RATE,              # 입력 레이트 기준
        channels=6,
        device=DEVICE_INDEX,
        callback=audio_callback,
        blocksize=int(MIC_SAMPLE_RATE * 0.1),   # 0.1초 블록
    )

    collector_thread = threading.Thread(target=audio_collector, daemon=True)
    detector_thread = threading.Thread(target=audio_detector_analyzer, daemon=True)

    collector_thread.start()
    detector_thread.start()

    print("실시간 감지 + 리샘플링(→32k) + PANNs + WS 전송 시작...\n(CTRL+C로 종료)\n")
    print("--------------------------------------")

    with stream:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n실시간 분석 종료")
