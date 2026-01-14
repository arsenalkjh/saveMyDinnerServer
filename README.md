# SaveMyDinner OCR Server (Vast AI Guide)

이 문서는 Vast AI GPU 서버에서 OCR 모듈을 실행하고, Streamlit 앱에서 API를 호출하는 방법을 설명합니다.

## 1. Vast AI 서버 설정

Vast AI 인스턴스에 접속하여 아래 단계를 수행하세요.

### 필수 패키지 설치
`server` 디렉토리로 이동하여 필요한 패키지를 설치합니다.

```bash
cd server
pip install -r requirements.txt
```

### 서버 실행
Uvicorn을 사용하여 FastAPI 서버를 실행합니다.
(외부 접속을 허용하려면 `--host 0.0.0.0` 옵션이 필요합니다)

```bash
# 기본 포트 8000 사용 시
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

*Vast AI의 포트 포워딩 설정에 따라 포트가 다를 수 있으니 확인해주세요.*

## 2. Streamlit 앱 설정 (로컬)

Streamlit 앱이 원격 서버를 바라보도록 설정합니다.

### 환경 변수 설정
프로젝트 루트의 `.env` 파일에 `OCR_SERVER_URL`을 추가하거나 수정하세요.

```ini
# .env 파일 예시
OCR_SERVER_URL=http://<VAST_AI_PUBLIC_IP>:<MAPPED_PORT>
```

만약 `.env` 파일이 없다면 `ocr_client.py`의 `DEFAULT_SERVER_URL` 변수를 수정해도 됩니다.

## 3. 변경 사항 요약

- **`server/server.py`**: API가 결과를 반환하지 않던 버그를 수정했습니다 (`return` 구문 추가).
- **`modules/ui/ocr_client.py`** [NEW]: 원격 OCR 서버와 통신하는 클라이언트 모듈을 새로 생성했습니다.
- **`modules/ui/services.py`**: `detect_ingredients` 함수가 로컬 모델 대신 원격 API(`ocr_client`)를 사용하도록 수정했습니다.

이제 Streamlit 앱에서 이미지를 업로드하면, Vast AI 서버로 이미지가 전송되어 OCR 처리가 수행됩니다.
