import sounddevice as sd
import numpy as np
import torch
import queue
import time
from panns_inference import AudioTagging, labels

# -----------------------------
# 설정
# -----------------------------
SAMPLE_RATE = 32000      # PANNs 모델 샘플링레이트
DURATION = .0           # 분석할 윈도우 길이(초)
print(len(labels)) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = r"C:\Users\juuip\panns_data\Cnn14_mAP=0.431.pth"

# PANNs 모델 초기화
audio_model = AudioTagging(checkpoint_path=checkpoint_path, device=device)

audio_queue = queue.Queue()

# -----------------------------
# 콜백 함수
# -----------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    mono_data = np.mean(indata, axis=1)  # 스테레오 -> 모노
    audio_queue.put(mono_data.astype(np.float32))

# -----------------------------
# 실시간 분석 루프
# -----------------------------
def process_audio():
    buffer = np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.float32)
    last_output_time = time.time()
    
    while True:
        chunk = audio_queue.get()
        chunk_len = len(chunk)
        buffer = np.roll(buffer, -chunk_len)
        buffer[-chunk_len:] = chunk

        # 3초 단위 출력
        if time.time() - last_output_time >= DURATION:
            waveform = torch.from_numpy(buffer).float().unsqueeze(0)
            
            with torch.no_grad():
                clipwise_output, _ = audio_model.inference(waveform)
                scores = clipwise_output[0].tolist()

            top_indices = np.argsort(scores)[-3:][::-1]
            timestamp = time.strftime("%H:%M:%S")
            print(f"\n[{timestamp}] Top3 분석 결과:")
            for idx in top_indices:
                print(f"  {labels[idx]}: {scores[idx]:.3f}")
            
            last_output_time = time.time()


# -----------------------------
# 마이크 스트리밍 시작
# -----------------------------
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    device=2,
    callback=audio_callback,
    blocksize=int(SAMPLE_RATE * 0.5)
)

try:
    print("실시간 음성 분석 시작...")
    with stream:
        process_audio()
except KeyboardInterrupt:
    print("실시간 분석 종료")
