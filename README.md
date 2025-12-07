# audio-module-test

**Version 4**
- server3.js
- sound3.js
- gui3.py : 마이크 감도 설정 추가

**Version 3** 
- server3.js : config_update 타입의 JSON 요청 허용 및 전달 로직 추가
- sound3.js : JSON 요청을 통한 실시간 설정 변경 로직 추가
- gui2.py : 실시간 설정 변경 로그 및 클라이언트 연결 상태창 추가

**Version 2**
- server2.js : GUI를 통한 내부 설정 값 변경 로직 추가
- sound2.js : GUI를 통한 내부 설정 값 변경 로직 추가
- gui.py : 간편하게 서버와 분류 코드를 실행할 GUI Runner

**Version 1**
- server.js : 웹소켓 서버
- sound.js : PANNs 기반 사운드 분류 및 감지

**필수 공통 모듈**
- tuning.py : 마이크 설정 모듈

**Required**
- ReSpeaker USB Mic Array
  - Need Mic Array Device + Module (https://wiki.seeedstudio.com/ReSpeaker-USB-Mic-Array/)
- PANNs Check Point
  - mAP=0.431 (Base CKP) (https://zenodo.org/records/3987831)
  - acc0.828 (fine-tuned CKP)
