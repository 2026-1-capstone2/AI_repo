# AI Server 작업 계획

본 문서는 골격 작업 이후 진행할 작업을 우선순위·의존관계에 따라 정리한다. 각 항목은 **목적**, **변경 대상**, **수행 절차**, **완료 기준**, **선행 의존성** 형식으로 기술한다.

---

## 0. 우선순위 체계

| 그룹 | 범위 | GPU 필요 | 병렬 가능 여부 |
|------|------|----------|-----------------|
| A | stub 모드 안정화 | 불필요 | 그룹 내 병렬 |
| B | Spring BE 실통합 | 불필요 | A 일부 완료 후 |
| C | VLM-3R 모델 통합 | 필요 | A·B와 병렬 가능 |
| D | 운영 안정화 | 부분 | C 완료 후 |
| E | 배포 | 필요 | D 완료 후 |

권장 진행 순서: **A1~A7 → (A6/A7 완료 시점부터 B1~B5와 C1~C13 병렬) → D → E**.

---

## A. stub 모드 안정화

`settings.stub_models=True` 상태에서 BE 통합 검증과 단위 테스트를 가능하게 만드는 작업.

### A1. `.env.example` 작성

- **목적**: 신규 개발자가 `.env` 구성을 즉시 복제 가능하도록 한다.
- **변경 대상**: `ai_repo/.env.example`
- **수행 절차**
  1. `app/core/config.py::Settings` 전 필드의 환경변수 키를 추출
  2. 민감 값은 `change-me` 또는 빈 문자열로 둔다
  3. 그룹별 주석 추가: `# === RabbitMQ ===`, `# === S3 ===`, `# === Model ===`
- **완료 기준**: `cp .env.example .env` 후 서버가 stub 모드로 정상 기동
- **의존성**: 없음

### A2. `internal_api_token` 검증 미들웨어 적용

- **목적**: BE → AI 호출에 인증 헤더 검사를 강제한다. 현재는 설정 키만 존재하고 라우터에서 검증되지 않는다.
- **변경 대상**: `app/routers/preprocess.py`, `app/routers/chat.py`, (선택) `app/core/security.py` 신설
- **수행 절차**
  1. `verify_internal_token(x_internal_token: str = Header(...))` 의존성 함수 정의
  2. `settings.internal_api_token` 과 비교하여 불일치 시 `HTTPException(401)`
  3. 각 라우터의 `dependencies=[Depends(verify_internal_token)]` 추가
  4. BE 팀과 헤더 키 이름 합의 (제안: `X-Internal-Token`)
- **완료 기준**: 토큰 누락/오류 요청이 401을 반환하고, 올바른 토큰 요청이 정상 처리
- **의존성**: 없음 (BE 합의는 B 그룹과 병행)

### A3. `video_downloader` 의 디렉토리 설정 일원화

- **목적**: 다운로드 경로 하드코딩을 제거해 환경별 분리(개발/스테이징/운영)를 가능하게 한다.
- **변경 대상**: `app/services/video_downloader.py`
- **수행 절차**
  1. 모듈 상단 `DOWNLOAD_DIR = Path("./temp/videos")` 를 제거
  2. `settings.download_dir` 을 사용하도록 변경
  3. 각 함수에서 디렉토리가 없으면 `mkdir(parents=True, exist_ok=True)` 호출
- **완료 기준**: `DOWNLOAD_DIR=./var/videos uvicorn ...` 환경변수 변경 시 실제 저장 위치가 바뀐다
- **의존성**: 없음

### A4. `/health` 실상태 반영

- **목적**: 현재는 모든 의존성 상태를 `False`로 고정 반환한다. 실제 상태를 반영하도록 갱신한다.
- **변경 대상**: `app/routers/health.py`
- **수행 절차**
  1. `ai_model_loaded`: `request.app.state.vlm is not None`
  2. `cut3r_loaded`: `request.app.state.cut3r is not None`
  3. `rabbitmq_connected`: `rabbitmq_publisher.connection is not None and not connection.is_closed`
  4. `s3_accessible`: 부팅 시 1회 head_bucket 시도 결과를 캐시
- **완료 기준**: 의존성 상태가 사실과 일치한다
- **의존성**: 없음

### A5. `/ready` 엔드포인트 추가

- **목적**: Kubernetes readiness probe 와 분리한다. `/health`는 liveness, `/ready`는 모델·큐 준비 완료를 의미한다.
- **변경 대상**: `app/routers/health.py`
- **수행 절차**
  1. `GET /ready` 엔드포인트 추가
  2. 다음 조건 모두 충족 시 200, 하나라도 미충족 시 503
     - stub 모드: 항상 200
     - 실모델 모드: `vlm`, `cut3r` 모두 로드 완료
     - RabbitMQ 연결 확인
  3. 응답 body에 현재 GPU VRAM 사용량 포함 (`torch.cuda.memory_allocated()`)
- **완료 기준**: stub 모드에서 즉시 ready, 실모델 모드에서 로드 완료 후에만 ready
- **의존성**: A4 완료 권장

### A6. stub 모드 end-to-end 테스트

- **목적**: 모델 통합과 무관하게 HTTP·MQ·캐시 흐름의 정합성을 검증한다.
- **변경 대상**: `ai_repo/tests/` 신설
- **수행 절차**
  1. `pytest`, `httpx`, `pytest-asyncio` 의존성 추가
  2. `conftest.py`: FastAPI TestClient + RabbitMQ mock (`unittest.mock`) 픽스처 정의
  3. 테스트 케이스
     - `test_preprocess_accepted`: `/preprocess` 호출 시 202 반환 및 job_store 등록
     - `test_preprocess_flow_publishes_mq`: BackgroundTask 완료 후 `publish_preprocess_completed` 호출 확인
     - `test_chat_before_ready_returns_425`: 전처리 미완료 상태에서 `/chat` 호출 시 425
     - `test_chat_stub_response`: 전처리 완료 상태에서 stub 응답 정상 반환
     - `test_duplicate_job_id_409`: 동일 `job_id` 재요청 시 409
- **완료 기준**: `pytest -v` 전체 통과
- **의존성**: A2 완료 권장 (인증 헤더 통합 테스트)

### A7. RabbitMQ 메시지 계약 문서화

- **목적**: BE 팀과 메시지 포맷 명세를 공유 가능한 형태로 고정한다.
- **변경 대상**: `ai_repo/docs/rabbitmq-contract.md` 신설
- **수행 절차**
  1. Exchange / Queue / Routing Key 표
  2. `analysis.completed`, `analysis.failed` payload 스키마 (JSON Schema 또는 Pydantic export)
  3. `event_type` 필드 확장 정책 (향후 chat 이벤트 추가 시)
  4. 에러 코드 카탈로그 (`INVALID_VIDEO_URL`, `VIDEO_TOO_LARGE` 등)
  5. 멱등성 처리 규약: BE는 `job_id` 기준 중복 메시지 처리 필요
- **완료 기준**: BE 팀 리뷰 완료 및 합의 사항 반영
- **의존성**: 없음

---

## B. Spring BE 실통합

stub 응답으로도 전 흐름 검증이 가능한 단계.

### B1. RabbitMQ 로컬 환경 구축

- **목적**: 로컬에서 BE 연동 테스트가 가능하도록 한다.
- **수행 절차**
  ```bash
  docker run -d --name rmq \
    -p 5672:5672 -p 15672:15672 \
    -e RABBITMQ_DEFAULT_USER=guest \
    -e RABBITMQ_DEFAULT_PASS=guest \
    rabbitmq:3-management
  ```
- **완료 기준**: `http://localhost:15672` (guest/guest) 로그인 후 `analysis_exchange`, `analysis_results` 생성 확인
- **의존성**: 없음

### B2. BE 큐 구독 정합성 검증

- **목적**: Spring BE가 발행된 메시지를 정상 수신·역직렬화하는지 확인한다.
- **수행 절차**
  1. AI 서버에서 `/preprocess` 호출 → stub 흐름이 `publish_preprocess_completed` 발행
  2. BE 측 큐 리스너 로그에서 메시지 수신 확인
  3. JSON 필드 매핑 검증 (snake_case ↔ camelCase 등)
- **완료 기준**: BE가 `job_id`, `spatial_features_s3_key`, `duration_ms` 모든 필드를 정확히 파싱
- **의존성**: A7 (계약 문서), B1

### B3. Pre-signed URL 다운로드 실테스트

- **목적**: BE가 발급한 Pre-signed URL이 AI 서버에서 정상 다운로드되는지 검증한다.
- **수행 절차**
  1. BE에서 테스트용 영상에 대해 GET pre-signed URL 발급
  2. AI 서버 `/preprocess` 에 해당 URL을 `video_url` 로 전달
  3. `services/video_downloader.py::download_video` 가 정상 다운로드 확인
  4. 만료 시간 검증: URL 만료 후 호출 시 `INVALID_VIDEO_URL` 반환
- **완료 기준**: 정상 URL은 다운로드 성공, 만료 URL은 에러 코드 정상 매핑
- **의존성**: B1

### B4. 에러 코드 합의

- **목적**: AI → BE 에러 메시지를 BE → 사용자 응답으로 매핑 가능하게 한다.
- **변경 대상**: `ai_repo/docs/error-codes.md` 신설
- **수행 절차**
  1. 현재 사용 중인 에러 코드 전수 조사 (`services/video_downloader.py`, `services/preprocess_service.py`)
  2. 각 코드에 대해 HTTP 응답 vs MQ payload 양쪽에서 동일한 코드 사용 확인
  3. BE 측 사용자 메시지 매핑표 작성
- **완료 기준**: BE 사용자 응답이 모든 에러 케이스에 대해 명확한 메시지를 반환
- **의존성**: A7

### B5. `job_id` 멱등성 정책 합의

- **목적**: BE 재시도 시 중복 처리/유실을 방지한다.
- **수행 절차**
  1. 현재 동작 검토: 동일 `job_id` 재요청 시 `409 Conflict` 반환
  2. BE 재시도 정책과의 정합성 검토
     - BE 클라이언트가 timeout 후 동일 ID로 재시도하는 경우의 처리
     - 이전 작업이 실패한 경우 재시도 허용 여부 (현재는 미허용)
  3. 정책 결정에 따라 `services/job_store.py::exists` 또는 `routers/preprocess.py` 수정
- **완료 기준**: 정책이 문서화되고 코드와 일치
- **의존성**: 없음

---

## C. VLM-3R 모델 통합

본 레포(`VLM-3Rdemo`)에 누적된 양자화·메모리 관리 패치를 ai_repo로 이식하고, 실모델 추론을 동작시키는 단계.

### C0. 컨텍스트: 본 레포에 누적된 패치 이력

이식 작업 전에 어떤 변경이 적용되어 있는지 인지가 필요하다. 이 정보는 `app/vlm/llava/model/builder.py` 가 왜 그렇게 작성되어 있는지 설명한다.

| 커밋 | 핵심 변경 | 파일 |
|------|-----------|------|
| `aabe5e7` | 4-bit 양자화 기본 지원. `load_in_4bit` 중복 kwarg 제거. 양자화 모델에 `device_map` 유지 | `llava/model/builder.py`, `playground/demo/video_demo.py` |
| `8e97ffe` | 8GB 환경 안정화. `low_cpu_mem_usage` 조건부 설정. `non_lora_trainables` 로딩을 `load_state_dict(strict=False)` 로 전환. 양자화 모델에 대해 `merge_and_unload()` 스킵 | `llava/model/builder.py` |
| `e1dca35` | 저VRAM 런타임 메모리 관리. `BitsAndBytesConfig.llm_int8_enable_fp32_cpu_offload=True`. `max_memory` 동적 계산 후 `PeftModel.from_pretrained` 에도 전달 | `llava/model/builder.py`, `Dockerfile`, `seperation.md` §14 |

이식 시 위 패치들은 `llava/` 디렉토리 복사에 따라 함께 따라온다.

### C1. `video_demo.py` 분리 동작 CLI 검증

- **목적**: ai_repo 통합 전에 본 레포에서 CUT3R 분리 추론이 정상 동작함을 확인한다.
- **변경 대상**: 본 레포 (`VLM-3Rdemo`) — ai_repo 변경 없음
- **수행 절차**
  1. `playground/demo/video_demo.py` 에 `--spatial-features-path` 옵션 추가
  2. `scripts/extract_spatial_features.py` 로 데모 영상 사전 처리하여 `.pt` 생성
  3. 추출된 `.pt` 를 `video_demo.py` 에 전달하여 CUT3R 단계 스킵 추론 실행
  4. 분리 전/후 응답 품질 비교 ([seperation.md](../seperation.md) §5.2)
- **완료 기준**: 분리 추론이 동등한 품질로 동작하고, VRAM 사용량이 ~1.5GB 감소
- **의존성**: 없음

### C2. `llava/` 디렉토리 이식

- **변경 대상**: 본 레포 `llava/` → `ai_repo/app/vlm/llava/`
- **수행 절차**
  ```bash
  cp -r /path/to/VLM-3Rdemo/llava ai_repo/app/vlm/llava
  ```
  이후 다음 하위 디렉토리만 유지 (학습 코드 제외):
  - `app/vlm/llava/model/` (필수)
  - `app/vlm/llava/constants.py`
  - `app/vlm/llava/conversation.py`
  - `app/vlm/llava/mm_utils.py`
  - `app/vlm/llava/utils.py`

  제외 대상: `train/`, `eval/`, `serve/` 등
- **완료 기준**: `ls ai_repo/app/vlm/llava/model/builder.py` 확인 가능
- **의존성**: 없음

### C3. `CUT3R/` 디렉토리 이식

- **변경 대상**: 본 레포 `CUT3R/` → `ai_repo/app/vlm/CUT3R/`
- **수행 절차**
  1. 본 레포 `CUT3R/src/` 전체 복사
  2. C++ 확장(`croco/models/curope`)도 함께 복사
  3. 가중치 파일 `cut3r_512_dpt_4_64.pth` 는 이미지에 포함하지 않고 볼륨 마운트로 처리 (Dockerfile에서 명시)
- **완료 기준**: `ai_repo/app/vlm/CUT3R/src/croco/models/curope/setup.py` 존재
- **의존성**: 없음

### C4. import 경로 일괄 치환

- **변경 대상**: `ai_repo/app/vlm/llava/**/*.py`, `ai_repo/app/vlm/CUT3R/**/*.py`
- **수행 절차**
  1. `llava.xxx` → `app.vlm.llava.xxx`
  2. `from llava import` → `from app.vlm.llava import`
  3. CUT3R 내부 상대 import는 변경 불필요 (패키지 내부 참조)
  4. sed 일괄 치환:
     ```bash
     cd ai_repo/app/vlm/llava
     find . -name "*.py" -exec sed -i '' 's/from llava\./from app.vlm.llava./g' {} +
     find . -name "*.py" -exec sed -i '' 's/import llava\./import app.vlm.llava./g' {} +
     ```
- **완료 기준**: `python -c "from app.vlm.llava.model.builder import load_pretrained_model"` 정상 import
- **의존성**: C2, C3

### C5. `load_vlm()` 실구현

- **변경 대상**: `ai_repo/app/vlm/loader.py`
- **수행 절차**
  1. `settings.stub_models=False` 분기에서 실제 로딩 코드 실행
  2. `overwrite_config["spatial_tower"] = None` 적용 시 builder.py가 spatial_tower 가중치를 메모리에 올리지 않는지 검증
     - 본 레포 builder.py 코드에서 `spatial_tower` config 처리 로직 확인 후 필요 시 builder.py 수정
  3. 로드 후 `torch.cuda.memory_allocated()` 출력하여 VRAM 사용량 측정
- **완료 기준**:
  - VLM 로드 시 CUT3R 가중치가 포함되지 않음 (메모리 사용량으로 검증)
  - `model.generate(spatial_features=...)` 호출 가능
- **의존성**: C4

### C6. `load_cut3r()` 실구현

- **변경 대상**: `ai_repo/app/vlm/loader.py`
- **수행 절차**
  1. `scripts/extract_spatial_features.py` 의 CUT3R 모델 로드 코드 분석
  2. 로드 함수 시그니처 매핑 (예: `Dust3R.from_pretrained` 또는 직접 weight load)
  3. fp16 + cuda device 배치
  4. 로드 후 forward 시그니처 확인하여 C7 작업의 입력 형태 결정
- **완료 기준**: `load_cut3r()` 반환 핸들이 `extract_spatial_features` 호출에 사용 가능
- **의존성**: C4

### C7. `extract_spatial_features()` 실구현

- **변경 대상**: `ai_repo/app/vlm/cut3r_runner.py`
- **수행 절차**
  1. `preprocessing.py::sample_video_frames` 로 32프레임 432×432 텐서 생성
  2. CUT3R forward 호출하여 `camera_tokens [F,1,768]`, `patch_tokens [F,729,768]` 반환
  3. fp16 + CPU 텐서로 변환 후 dict 반환 (S3 저장용 직렬화 가능 상태)
  4. 기존 `extract_spatial_features_stub` 은 보존 (테스트용)
- **완료 기준**: 본 레포 `extract_spatial_features.py` 출력과 동일한 텐서 shape·dtype·값 일치
- **의존성**: C6

### C8. `Dockerfile` 작성

- **변경 대상**: `ai_repo/Dockerfile`
- **수행 절차**
  1. 본 레포 `Dockerfile` 을 베이스로 복사
  2. 다음 항목 유지
     - `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`
     - `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"` (빌드 타임 GPU 미가용 환경 대응)
     - CUT3R curope C++ 확장 빌드 단계
  3. 경로 변경
     - `COPY ./llava ./llava` → `COPY ./app/vlm/llava ./app/vlm/llava`
     - `COPY ./CUT3R ./CUT3R` → `COPY ./app/vlm/CUT3R ./app/vlm/CUT3R`
  4. CMD: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1`
  5. 모델 가중치(`vlm-3r-llava-qwen2-lora`, CUT3R `.pth`)는 `VOLUME ["/data"]` 로 처리
- **완료 기준**: `docker build .` 성공, 컨테이너 기동 시 stub 모드 정상 동작
- **의존성**: C2, C3, C11

### C9. 환경 셋업 문서 이식

- **목적**: 본 레포 `setup.md` 의 Ubuntu 24.04 트러블슈팅 8개 항목과 `docker_run_windows.md` 의 WSL/CUT3R 빌드 이슈를 ai_repo로 보존한다.
- **변경 대상**: `ai_repo/docs/setup.md`, `ai_repo/docs/docker-windows.md`
- **수행 절차**: 본 레포 두 문서를 복사하고, 경로 참조를 ai_repo 기준으로 갱신
- **완료 기준**: ai_repo만 보고도 환경 셋업이 가능
- **의존성**: 없음

### C10. 메모리 관리 컨텍스트 문서

- **목적**: 본 레포 `seperation.md` §14 (builder.py 패치의 근거)를 ai_repo에 보존한다. 후속 개발자가 builder.py 수정 시 원인을 파악할 수 있어야 한다.
- **변경 대상**: `ai_repo/docs/memory-management.md`
- **수행 절차**: `seperation.md` §14.1~14.6 발췌, ai_repo 경로 기준으로 재작성
- **완료 기준**: builder.py의 `llm_int8_enable_fp32_cpu_offload`, `max_memory`, PEFT 호출 등 각 패치의 배경 설명 포함
- **의존성**: 없음

### C11. `requirements.txt` 실모델 의존성 활성화

- **변경 대상**: `ai_repo/requirements.txt`
- **수행 절차**: 현재 주석 처리된 다음 항목을 활성화
  ```
  torch==2.1.1
  torchvision==0.16.1
  transformers==4.40.0.dev0
  flash-attn==2.7.1
  bitsandbytes>=0.41.3
  accelerate==0.29.1
  peft==0.4.0
  tokenizers==0.15.2
  sentencepiece==0.1.99
  decord
  einops
  opencv-python-headless
  pillow
  numpy==1.26.4
  timm
  open-clip-torch
  ```
- **완료 기준**: `pip install -r requirements.txt` 성공
- **의존성**: 없음 (다만 C5/C6/C7 작업 전에 설치되어 있어야 함)

### C12. `stub_models=False` 전환 및 단일 영상 e2e

- **수행 절차**
  1. `.env` 에 `STUB_MODELS=false` 설정
  2. 모델 가중치 마운트 (`MODEL_PATH`, `CUT3R_WEIGHTS`)
  3. 서버 기동 후 `/ready` 가 200 반환할 때까지 대기 (모델 로드 ~30초)
  4. 단일 영상으로 `/preprocess` → `/chat` 흐름 검증
- **완료 기준**: stub 응답이 아닌 실모델 응답 반환, `metadata.stub == false`
- **의존성**: C5, C6, C7, C11

### C13. 캐시 적중 검증

- **수행 절차**
  1. 동일 `job_id` 에 대해 `/chat` 을 2회 연속 호출
  2. 1회차: `spatial_cache_hit=false`, 응답 시간 측정
  3. 2회차: `spatial_cache_hit=true`, 응답 시간 측정
- **완료 기준**: 2회차가 1회차보다 유의미하게 빠름 (S3 다운로드 시간 절감)
- **의존성**: C12

---

## D. 운영 안정화

### D1. `docker-compose.yml` 작성

- **변경 대상**: `ai_repo/docker-compose.yml`
- **수행 절차**
  1. ai-server 서비스: GPU 할당 (`deploy.resources.reservations.devices`), 포트 8000
  2. rabbitmq 서비스: 로컬 개발용
  3. 모델 볼륨 마운트 (`/data/vlm-3r-llava-qwen2-lora`, `/data/CUT3R/...`)
  4. HuggingFace 캐시 볼륨 분리
  5. 환경변수 `.env` 파일 참조
- **완료 기준**: `docker compose up` 으로 전체 스택 기동
- **의존성**: C8

### D2. 구조화 로깅

- **변경 대상**: `app/core/logging.py` 신설, 전 모듈 logger 호출 갱신
- **수행 절차**
  1. JSON 포맷 로거 설정 (`python-json-logger`)
  2. 모든 로그에 `job_id` context 자동 주입 (`contextvars`)
  3. 로그 레벨 환경변수 분리 (`LOG_LEVEL`)
- **완료 기준**: 단일 `job_id` 의 로그를 grep으로 시계열 추적 가능
- **의존성**: 없음

### D3. 메트릭 수집

- **변경 대상**: `app/core/metrics.py` 신설
- **수행 절차**
  1. `prometheus-fastapi-instrumentator` 통합
  2. 커스텀 메트릭:
     - `vlm_inference_duration_seconds` (histogram)
     - `spatial_cache_hit_total` (counter)
     - `gpu_memory_allocated_bytes` (gauge, 주기적 측정)
     - `preprocess_queue_depth` (gauge)
  3. `/metrics` 엔드포인트 노출
- **완료 기준**: Prometheus가 메트릭 스크랩 가능
- **의존성**: 없음

### D4. 영상 파일 정리 정책

- **목적**: 현재 `preprocess_service` 가 video 파일을 chat에서 재사용하기 위해 보관한다. 디스크 누적을 방지하는 정리 로직이 필요하다.
- **수행 절차**
  1. 정책 결정
     - 옵션 A: TTL 기반 (예: 마지막 chat 호출로부터 1시간 후 삭제)
     - 옵션 B: 명시적 `/jobs/{id}/cleanup` 엔드포인트로 BE가 트리거
  2. 선택한 정책에 따라 `services/cleanup_service.py` 또는 백그라운드 워커 구현
- **완료 기준**: 100개 영상 처리 후 디스크 사용량이 일정 임계치 이하 유지
- **의존성**: C12

### D5. OOM 가드

- **수행 절차**
  1. `chat_service::_real_response` 에 `torch.cuda.OutOfMemoryError` 핸들링 추가
  2. OOM 시 `torch.cuda.empty_cache()` 호출 후 503 반환
  3. 동시 요청 제한 (`asyncio.Semaphore`) 도입
- **완료 기준**: 동시 요청 부하 시 OOM이 서버 크래시로 이어지지 않음
- **의존성**: C12

### D6. RabbitMQ outbox 패턴

- **목적**: 전처리는 완료되었으나 MQ publish가 실패한 경우 메시지 유실을 방지한다.
- **수행 절차**
  1. `job_store` 에 `publish_pending: bool` 필드 추가
  2. publish 실패 시 pending 마킹
  3. 백그라운드 재발행 워커 (`services/outbox_worker.py`) 가 주기적으로 pending 작업을 스캔하여 재시도
- **완료 기준**: RabbitMQ 일시 중단 시에도 복구 후 메시지 자동 발행
- **의존성**: D2 (로깅)

---

## E. 배포

### E1. GPU 인스턴스 선정 및 프로비저닝

- **수행 절차**
  1. 후보 비교
     - AWS g5.xlarge (A10G 24GB, 약 $730/월)
     - 자체 서버 (RTX 3060 Ti 8GB, [seperation.md](../seperation.md) §1 환경)
  2. 운영 트래픽 예측 후 선택
  3. CUDA·드라이버·Docker GPU runtime 설치
- **완료 기준**: 인스턴스에서 `nvidia-smi` 및 `docker run --gpus all` 정상 동작
- **의존성**: 없음

### E2. S3 IAM 구성

- **수행 절차**
  1. AI 서버 인스턴스용 IAM Role 생성
  2. 필요 권한: `s3:GetObject`, `s3:PutObject` (특정 prefix 한정)
  3. EC2/EKS 인스턴스 프로파일에 Role 연결
  4. `services/s3_service.py` 에서 명시적 키 제거 (Role 기반 자동 인증)
- **완료 기준**: 명시적 키 없이 S3 PUT/GET 동작
- **의존성**: E1

### E3. 도메인 + TLS

- **수행 절차**
  1. 운영 도메인 설정 (Route 53 또는 CDN)
  2. ALB + ACM 인증서 또는 nginx + Let's Encrypt
  3. ngrok 의존성 제거 (`start_ngrok.py` 삭제 또는 dev-only로 격리)
- **완료 기준**: HTTPS 로 외부 접근 가능
- **의존성**: E1

### E4. 오케스트레이션 매니페스트

- **수행 절차**
  1. Kubernetes 또는 ECS 매니페스트 작성
  2. 핵심 항목
     - `replicas: 1` (모델 1회 로드 보장)
     - GPU 자원 요청 (`nvidia.com/gpu: 1`)
     - Liveness probe: `GET /health`
     - Readiness probe: `GET /ready`, 초기 지연 60초 (모델 로드 시간 고려)
     - 영상 임시 디렉토리용 emptyDir 또는 PVC
  3. RabbitMQ 연결 정보를 Secret으로 분리
- **완료 기준**: 운영 클러스터에서 자동 배포·롤백 가능
- **의존성**: D1, E1, E2, E3

---

## 부록: 의존성 그래프

```
A1, A3, A4         (독립)
A2 ──┐
     ├──→ A6
A5 ──┘
A7 ──→ B2, B4

B1 ──→ B2, B3
B5    (독립)

C1 ──→ C2, C3 ──→ C4 ──→ C5, C6 ──→ C7
                                    │
C9, C10              (독립)         ├──→ C12 ──→ C13
                                    │
C11 ─────────────────────────────────┘

C2, C3, C11 ──→ C8 ──→ D1
C12 ──→ D4, D5
D2 ──→ D6

E1 ──→ E2, E3
D1, E1, E2, E3 ──→ E4
```
