# audio-module-test
모든 파일 다 sounddevice 라이프러리를 통해 실시간?으로 처리됨.


**PANNs.py** 
- PANNs를 통한 음성 분류
- 실행할 때 주의 사항
  - class_labels_indices.csv(라벨 파일)와
  - Cnn14_mAP=0.431.pth(체크 포인트 파일=가중치 파일) 경로 설정 필요
- 성능
  - 517개? 정도의 분류 라벨이 존재 
  - Speech 중심의 AudioSet을 통해 학습하여 Speech 분류를 나름 쓸만
  - 하지만 그외 차, 동물, 박수, 큰 소리 등의 분류는 어려움. 이러한 요소들은 주로 Slience로 분류됨..
  - 따라서 이를 보완할 새로운 모델이 필요할 듯...

**VoskSR.py**
- Vosk 모델을 통한 음성 인식(STT)
- 로컬(PC)에서 동작
- 성능
  - 인식이 이상하게 됨. 쓰지 마셈

**WhisperSR.py**
- Whisper 모델을 통한 음성 인식(STT)
- 로컬(PC)에서 동작
- 성능
  - 모델 용량이 다양함
  - small 모델로 실행해봤는데 Vosk보다 훨씬 괜찮음
  - 근데 간혹 이상한 말이 나타나지만.. 이정도면 괜춘
