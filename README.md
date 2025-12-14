# audio-module-test

**Required**
- ReSpeaker USB Mic Array
  - Need Mic Array Device + Module (https://wiki.seeedstudio.com/ReSpeaker-USB-Mic-Array/)
- PANNs Check Point
  - mAP=0.431 (Base CKP) (https://zenodo.org/records/3987831)
  - acc0.828 (fine-tuned CKP)

 **Module**
 - embedded_audio_module
   - 마이크 임베디드 시스템을 활용한 소리 감지 및 분류
   - (https://www.seeedstudio.com/ReSpeaker-USB-Mic-Array-p-4247.html)
 - pra_audio_module
   - 오픈소스(Pyroomacoustics)를 활용한 소리 감지 및 분류
   - (https://pyroomacoustics.readthedocs.io/en/pypi-release/index.html)
