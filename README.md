# Spatial Analysis AI Server

> 단안 카메라 영상 → 3D 공간 분석(CUT3R) + 자연어 응답(VLM-3R)
> FastAPI 기반 AI 서버. Spring BE와 RabbitMQ로 비동기 통신.

---

## 1. 서비스 개요

사용자가 업로드한 실내 영상에 대해 다음을 수행한다:

1. **전처리** — CUT3R로 영상의 3D spatial feature를 추출해 S3에 저장
2. **챗봇 추론** — VLM-3R(LLaVA-NeXT-Video + Qwen2-7B + LoRA + spatial feature)로 자연어 질문 응답

### 1.1. 시스템 흐름

```
[1] 영상 업로드
    사용자 → Spring BE → S3 (Pre-signed PUT)

[2] 전처리 트리거
    Spring BE  ── POST /api/v1/preprocess ──▶  AI Server
                  { job_id, user_id, video_url(Pre-signed GET) }
    AI Server  ── 202 Accepted (즉시) ─────▶  Spring BE

[3] 백그라운드 처리
    AI Server:
      Pre-signed URL로 영상 다운로드
      → CUT3R로 spatial_features 추출
      → S3 업로드  (spatial_features/{user_id}/{job_id}.pt)

[4] 완료 통보 (RabbitMQ topic exchange)
    AI Server  ── analysis.completed | analysis.failed ──▶  Spring BE

[5] 사용자 질문
    Spring BE  ── POST /api/v1/chat ──▶  AI Server
                  { job_id, question, history[] }
    AI Server  ── 동기 응답 ─────────▶  Spring BE
                  { answer, metadata{tokens_used, inference_time_ms, ...} }
```

### 1.2. 핵심 설계 결정

| 항목 | 선택 | 이유 |
|------|------|------|
| 프로세스 구조 | 단일 프로세스 + 모델 별도 로드 | GPU 메모리 효율, 워커 모델 중복 적재 방지 |
| 모델 핸들 | `app.state.vlm` / `app.state.cut3r` 분리 | VLM 로드 시 `spatial_tower` 비활성화 → CUT3R 중복 적재 방지 |
| BE 통보 방식 | **RabbitMQ** topic exchange | 비동기·내구성 큐 |
| 캐시 | `@lru_cache` GPU LRU | spatial_features 35MB × N개 GPU 상주 |
| 영상 다운로드 | httpx 스트리밍 (Pre-signed URL) | boto3 IAM 의존 없이 BE 발급 URL 그대로 사용 |

---

## 2. 프로젝트 구조

```
ai_repo/
├── app/
│   ├── main.py                  ← FastAPI 진입점, lifespan, 라우터 등록
│   │
│   ├── core/
│   │   ├── config.py            ← Settings (pydantic-settings, .env 로드)
│   │   └── lifespan.py          ← VLM + CUT3R 라이프사이클
│   │
│   ├── routers/                 ← HTTP 인터페이스
│   │   ├── preprocess.py        ← POST /api/v1/preprocess
│   │   ├── chat.py              ← POST /api/v1/chat
│   │   ├── jobs.py              ← GET /api/v1/jobs/{job_id}
│   │   └── health.py            ← GET /health
│   │
│   ├── schemas/                 ← Pydantic DTO
│   │   ├── preprocess.py        ← 요청/응답 + MQ 메시지 payload
│   │   ├── chat.py              ← 요청/응답
│   │   └── common.py            ← 공통 응답 포맷
│   │
│   ├── services/                ← 비즈니스 로직 (HTTP ↔ vlm 사이)
│   │   ├── preprocess_service.py    ← 다운로드 → CUT3R → S3 → MQ 발행
│   │   ├── chat_service.py          ← stub/실모델 분기, 추론 실행
│   │   ├── cache_service.py         ← spatial_features / video_tensor GPU LRU
│   │   ├── s3_service.py            ← boto3 (spatial_features PUT/GET)
│   │   ├── rabbitmq_publisher.py    ← pika 기반 MQ publisher (lazy connect)
│   │   ├── video_downloader.py      ← httpx로 Pre-signed URL 다운로드
│   │   └── job_store.py             ← 인메모리 작업 상태 저장
│   │
│   └── vlm/                     ← VLM-3R 도메인 (HTTP 무관)
│       ├── loader.py            ← load_vlm() + load_cut3r() 분리
│       ├── inference.py         ← run_inference()
│       ├── cut3r_runner.py      ← extract_spatial_features() + stub
│       └── preprocessing.py     ← 프레임 샘플링, build_prompt
│
├── requirements.txt             ← FastAPI/pika/httpx/boto3 (PyTorch 등은 모델 통합 단계에 추가)
├── start_ngrok.py               ← 개발용 ngrok 터널
├── TODO.md                      ← 우선순위별 다음 작업 리스트
└── README.md
```

### 2.1. 폴더 책임 분리

| 폴더 | 의존 가능 | 의존 금지 | 핵심 |
|------|-----------|-----------|------|
| `app/routers/` | `services/`, `schemas/`, `core/` | `vlm/` 직접 호출 | HTTP만 알고 모델은 모름 |
| `app/services/` | `vlm/`, `core/`, `schemas/` | FastAPI 객체(Request 등) | HTTP와 모델 사이 비즈니스 로직 |
| `app/vlm/` | `app/vlm/llava`, `app/vlm/CUT3R`, `core/config` | FastAPI, 비즈니스 규칙 | HTTP 무관, 단독 CLI 호출도 가능해야 함 |
| `app/core/` | (모든 모듈에서 임포트) | 다른 `app/*` 폴더 | 설정·라이프사이클 같은 공통 |
| `app/schemas/` | (순수 Pydantic) | 비즈니스 로직 | 데이터 형태만 정의 |

---

## 3. API 엔드포인트

### 3.1. `POST /api/v1/preprocess` — 영상 전처리

**Request**
```json
{
  "job_id": "uuid-string",
  "user_id": "u_abc123",
  "video_url": "https://...s3.amazonaws.com/...?X-Amz-...",
  "metadata": { "video_duration_sec": 12, "video_resolution": "1080p" }
}
```

**Response** (`202 Accepted`, 즉시 반환)
```json
{
  "job_id": "uuid-string",
  "status": "accepted",
  "estimated_time_sec": 45,
  "received_at": "2026-05-14T10:00:00Z"
}
```

전처리 진행은 `BackgroundTasks`로 수행되며, 완료/실패는 RabbitMQ로 통보.

### 3.2. `POST /api/v1/chat` — 자연어 질문 응답

**Request**
```json
{
  "job_id": "uuid-string",
  "user_id": "u_abc123",
  "question": "소파는 어디에 있나요?",
  "history": [
    {"role": "user", "content": "방 크기는?"},
    {"role": "assistant", "content": "약 4m × 5m 입니다."}
  ],
  "max_new_tokens": 512,
  "temperature": 0.0
}
```

**Response** (`200 OK`, 동기 응답)
```json
{
  "job_id": "uuid-string",
  "answer": "소파는 영상 중앙의 창문 앞에 위치해 있습니다.",
  "metadata": {
    "tokens_used": 142,
    "inference_time_ms": 487,
    "spatial_cache_hit": true,
    "stub": false
  }
}
```

전처리 미완료(`status != "completed"`) 시 `425 Too Early`.

### 3.3. `GET /api/v1/jobs/{job_id}` — 작업 상태 조회

폴링용 — webhook 누락 시 BE가 복구 경로로 사용.

### 3.4. `GET /health` — 헬스 체크

현재는 단순 응답. 추후 `/ready` 분리 예정 (TODO A5).

---

## 4. RabbitMQ 메시지 계약

### 4.1. Exchange / Queue

| 항목 | 값 |
|------|-----|
| Exchange | `analysis_exchange` (topic, durable) |
| Queue | `analysis_results` (durable) |
| Binding | `analysis.*` |

### 4.2. Routing Key & Payload

**`analysis.completed`** (전처리 성공)
```json
{
  "job_id": "...",
  "user_id": "...",
  "status": "completed",
  "event_type": "preprocess_completed",
  "spatial_features_s3_key": "spatial_features/u_abc123/job-id.pt",
  "duration_ms": 28430,
  "completed_at": "2026-05-14T10:00:30Z"
}
```

**`analysis.failed`** (전처리 실패)
```json
{
  "job_id": "...",
  "user_id": "...",
  "status": "failed",
  "event_type": "preprocess_failed",
  "error": { "code": "VIDEO_TOO_LARGE", "message": "..." },
  "failed_at": "2026-05-14T10:00:10Z"
}
```

**`event_type` 필드**: 향후 `chat_completed` 등으로 확장 가능. BE는 routing key + event_type 조합으로 디스패치.

### 4.3. 에러 코드

| code | 의미 |
|------|------|
| `INVALID_VIDEO_URL` | Pre-signed URL 접근 실패 |
| `VIDEO_DOWNLOAD_FAILED` | 다운로드 중 네트워크/타임아웃 |
| `VIDEO_TOO_LARGE` | 500MB 초과 |
| `UNSUPPORTED_VIDEO_FORMAT` | mp4/mov/avi 외 |
| `INTERNAL_SERVER_ERROR` | 알 수 없는 오류 |

---

## 5. 실행 방법

### 5.1. 로컬 개발 (stub 모드)

GPU 없이도 **전체 흐름 시뮬레이션 가능** — `settings.stub_models=True` (기본값).

```bash
# 1) 가상환경
python3 -m venv venv
source venv/bin/activate

# 2) 의존성 설치
pip install -r requirements.txt

# 3) (선택) RabbitMQ 띄우기
docker run -d --name rmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management

# 4) 서버 실행
uvicorn app.main:app --reload --port 8000

# 5) Swagger UI
open http://localhost:8000/docs
```

### 5.2. 통신 테스트 (stub)

```bash
# 전처리 요청
curl -X POST http://localhost:8000/api/v1/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-job-1",
    "user_id": "u_demo",
    "video_url": "https://example.com/test.mp4"
  }'

# 작업 상태
curl http://localhost:8000/api/v1/jobs/test-job-1

# 채팅 (stub 응답)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-job-1",
    "user_id": "u_demo",
    "question": "방의 크기는?"
  }'
```

### 5.3. ngrok 외부 노출 (팀원 데모용)

```bash
python start_ngrok.py
# → 공개 URL이 출력됨
```

### 5.4. 실모델 모드 (GPU 필요, 미구현)

모델 통합은 [TODO.md](TODO.md) §C 참고.
완료 후:
```bash
export STUB_MODELS=false
export MODEL_PATH=/data/vlm-3r-llava-qwen2-lora
export CUT3R_WEIGHTS=/data/CUT3R/src/cut3r_512_dpt_4_64.pth
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **`--workers 1` 필수**: 모델이 GPU에 1번만 로드되어야 함. N개 워커 = VRAM N배.

---

## 6. 환경 설정

`app/core/config.py` 의 `Settings` 클래스가 환경변수/`.env`에서 자동 로드. 주요 키:

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `STUB_MODELS` | `true` | true면 실모델 미로드 (개발/CI용) |
| `MODEL_PATH` | `/data/vlm-3r-llava-qwen2-lora` | LoRA 가중치 경로 |
| `MODEL_BASE` | `lmms-lab/LLaVA-NeXT-Video-7B-Qwen2` | HF 베이스 모델 |
| `CUT3R_WEIGHTS` | `/data/CUT3R/src/cut3r_512_dpt_4_64.pth` | CUT3R 체크포인트 |
| `LOAD_4BIT` | `true` | 4-bit 양자화 로드 |
| `DISABLE_SPATIAL_TOWER_IN_VLM` | `true` | VLM 로드 시 spatial_tower 비활성화 |
| `UNLOAD_CUT3R_AFTER_PREPROCESS` | `false` | 저VRAM 시 true로 |
| `SPATIAL_CACHE_SIZE` | `20` | GPU에 들고 있을 spatial_features 개수 |
| `VIDEO_CACHE_SIZE` | `10` | GPU에 들고 있을 video tensor 개수 |
| `S3_BUCKET` | `vlm3r-bucket` | features 저장 버킷 |
| `AWS_REGION` | `ap-northeast-2` | S3 리전 |
| `RABBITMQ_HOST` | `localhost` | MQ 호스트 |
| `RABBITMQ_USER` / `RABBITMQ_PASSWORD` | `guest` / `guest` | MQ 자격증명 |
| `INTERNAL_API_TOKEN` | `dev-internal-token-change-me` | BE → AI 인증 토큰 (TODO A2에서 검증 적용) |

`.env.example` 작성은 [TODO.md](TODO.md) A1.

---

## 7. 작업 상태

### 7.1. 완료
- [x] FastAPI 골격 + 라우터 분리 (`/preprocess`, `/chat`, `/jobs`, `/health`)
- [x] Pydantic schemas (preprocess, chat, MQ payload)
- [x] RabbitMQ publisher (lazy connect, `analysis.*` routing)
- [x] httpx 기반 영상 다운로드 (Pre-signed URL)
- [x] 인메모리 `job_store`
- [x] `services/`: preprocess / chat / cache / s3 (boto3 lazy import)
- [x] `vlm/`: loader / inference / cut3r_runner / preprocessing 골격
- [x] stub 모드 (GPU·실모델 없이 전 흐름 시뮬레이션 가능)
- [x] FastAPI lifespan (모델 로드/해제)

### 7.2. 다음 작업

[TODO.md](TODO.md) 에서 우선순위별로 관리:

- **A: stub 모드로 가능** (인증·헬스·테스트·메시지 계약 문서)
- **B: BE 실통합** (RabbitMQ·Pre-signed URL·에러 코드 합의)
- **C: VLM-3R 모델 통합** (llava/CUT3R 이식 → 실모델 로드 → e2e 검증)
- **D: 안정화** (Dockerfile · 로깅 · 메트릭 · OOM 가드)
- **E: 배포** (GPU 인스턴스 · IAM · 도메인 · k8s/ECS)

---

## 8. 관련 문서

- [TODO.md](TODO.md) — 우선순위별 작업 리스트
