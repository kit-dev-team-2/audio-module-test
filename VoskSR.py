import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer

# 모델 경로 지정 (예: ./vosk-model-small-ko-0.22)
MODEL_PATH = "./model/vosk-model-small-ko-0.22"

print(sd.query_devices())

# 샘플레이트 (보통 16kHz)
SAMPLE_RATE = 16000
DEVICE_ID = 2  # ✅ 마이크(OMEN Cam & Voice)

# 모델 로드
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

# 오디오 스트림 큐
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    q.put(bytes(indata))

# 오디오 입력 스트림 열기
with sd.RawInputStream(samplerate=SAMPLE_RATE,
                       blocksize=8000,
                       dtype='int16',
                       channels=1,
                       device=DEVICE_ID,
                       callback=callback):
    print("🎙️ 말하세요 (종료하려면 Ctrl+C)")
    while True:
        data = q.get()
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            text = json.loads(result).get("text", "")
            if text:
                print("🗣️ 인식 결과:", text)
        else:
            partial = recognizer.PartialResult()
            partial_text = json.loads(partial).get("partial", "")
            if partial_text:
                print("⌛ 인식 중:", partial_text, end="\r")


# import sounddevice as sd
# import numpy as np

# def print_volume(indata, frames, time, status):
#     volume = np.linalg.norm(indata) * 10
#     print(f"볼륨: {volume:.2f}")

# DEVICE_ID = 1  # 마이크(OMEN Cam & Voice)
# SAMPLE_RATE = 16000

# with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
#                     device=DEVICE_ID, callback=print_volume):
#     print("말하면 볼륨이 나와야 함 (Ctrl+C 종료)")
#     import time
#     while True:
#         time.sleep(0.1)
