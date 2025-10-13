import sounddevice as sd
import numpy as np
import queue
import whisper
import time

# -----------------------------
# 설정
# -----------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
DEVICE_ID = 2        # sd.query_devices()로 확인한 마이크 ID
BLOCKSIZE = 8000     # 0.5초 단위 블록
MODEL_SIZE = "small" # tiny, base, small, medium, large
BUFFER_SECONDS = 2   # 2초씩 모아서 처리

# -----------------------------
# Whisper 모델 로드
# -----------------------------
device = "cuda" if False else "cpu"  # FP32 고정, GPU 사용 안 함
model = whisper.load_model(MODEL_SIZE).to(device)
print(f"✅ 모델 로드 완료 (device={device}, FP32, 한국어 지원)")

# -----------------------------
# 오디오 큐 생성
# -----------------------------
q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️ 상태:", status)
    q.put(indata.copy())

# -----------------------------
# 마이크 스트림 열기
# -----------------------------
try:
    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        device=DEVICE_ID,
                        blocksize=BLOCKSIZE,
                        dtype='float32',
                        callback=audio_callback):
        print("🎙️ 말하세요 (Ctrl+C로 종료)")

        audio_buffer = []

        while True:
            block = q.get()           # 블록 가져오기
            block = np.squeeze(block)

            # 2차원 남아있으면 첫 채널만
            if block.ndim > 1:
                block = block[:, 0]

            audio_buffer.append(block)

            # 일정 시간 모였으면 Whisper로 처리
            if sum(len(b) for b in audio_buffer) >= SAMPLE_RATE * BUFFER_SECONDS:
                audio_np = np.concatenate(audio_buffer)
                result = model.transcribe(audio_np, language="ko")
                text = result.get("text", "").strip()
                if text:
                    print("🗣️ 인식 결과:", text)
                audio_buffer = []

except KeyboardInterrupt:
    print("\n⏹️ 종료")
except Exception as e:
    print("❌ 오류 발생:", e)
