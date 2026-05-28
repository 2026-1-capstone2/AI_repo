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

## 2. 실행 방법

### 2.1. 요구사항

| 항목 | stub 모드 (개발/테스트) | 실모델 모드 (운영) |
|------|------------------------|-------------------|
| **Python** | 3.13 (현재 검증 기준) | **3.10** (PyTorch 2.1.1 호환) |
| **CUDA** | 불필요 | **12.1** |
| **GPU** | 불필요 | NVIDIA, Compute Capability 8.0+ (Ampere 이상) |
| **PyTorch** | 미사용 | 2.1.1 + cu121 |
| **컴파일러** | 불필요 | gcc/g++ 11 (CUT3R C++ 확장 빌드) |
| **OS** | macOS / Linux | Ubuntu 22.04 권장 |
| **메모리(VRAM)** | 불필요 | 8GB+ (RTX 3060 Ti 기준 동작 확인) |

> **참고**: PyTorch 2.1.1은 Python 3.13용 wheel을 제공하지 않는다. 실모델 통합 시(TODO C 그룹) 별도 가상환경을 Python 3.10 기반으로 새로 만들거나 PyTorch를 3.13 호환 버전(2.5+)으로 갱신해야 한다. 본 레포 표준 환경 셋업 절차는 [setup.md](../setup.md) 참고.

### 2.2. 로컬 개발 (stub 모드)

GPU 없이도 **전체 흐름 시뮬레이션 가능** — `settings.stub_models=True` (기본값).

```bash
# 1) 가상환경 (Python 3.13)
python -m venv .venv
source .venv/bin/activate

# 2) 의존성 설치
pip install -r requirements.txt

# 3) (선택) RabbitMQ 띄우기
docker run -d --name rmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management

# 4) 서버 실행
uvicorn app.main:app --reload --port 8000

# 5) Swagger UI
open http://localhost:8000/docs
```

### 2.3. 통신 테스트 (stub)

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

### 2.4. ngrok 외부 노출 (팀원 데모용)

```bash
python start_ngrok.py
# → 공개 URL이 출력됨
```

### 2.5. 모델 가중치 다운로드

실모델 모드 실행 전 다음 세 가지 가중치가 필요하다.

| 가중치 | 출처 | 크기 | 저장 경로 (기본값) |
|--------|------|------|-------------------|
| **CUT3R** | Google Drive | ~600 MB | `/data/CUT3R/src/cut3r_512_dpt_4_64.pth` |
| **VLM-3R LoRA** | HuggingFace `Journey9ni/vlm-3r-llava-qwen2-lora` | ~700 MB | `/data/vlm-3r-llava-qwen2-lora/` |
| **베이스 모델** (LLaVA-NeXT-Video-7B-Qwen2) | HuggingFace `lmms-lab/LLaVA-NeXT-Video-7B-Qwen2` | ~14 GB | HuggingFace 캐시 (자동) |

총 약 15~16 GB. 저장 경로는 `settings.cut3r_weights`, `settings.model_path` 와 일치시키거나 환경변수로 재정의한다.

#### 사전 설치

```bash
sudo apt install git-lfs -y    # LoRA의 LFS 파일 다운로드용
pip install gdown               # CUT3R의 Google Drive 다운로드용
```

#### 1) CUT3R 가중치

```bash
sudo mkdir -p /data/CUT3R/src && sudo chown $USER /data/CUT3R/src
cd /data/CUT3R/src
gdown 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD       # cut3r_512_dpt_4_64.pth (~600MB)
```

> gdown 6.x 부터 `--fuzzy` 옵션이 제거되었으므로 파일 ID 만 직접 전달해야 한다.

#### 2) VLM-3R LoRA 가중치

```bash
git lfs install                # 1회만, 미설치 시 134바이트 LFS 포인터만 받음
cd /data
git clone https://huggingface.co/Journey9ni/vlm-3r-llava-qwen2-lora
cd vlm-3r-llava-qwen2-lora
git lfs pull                   # adapter_model.bin (617MB), non_lora_trainables.bin (51MB)
```

#### 3) 베이스 모델 (자동 다운로드)

서버 첫 기동 시 `from_pretrained()` 호출로 HuggingFace에서 자동 다운로드된다(~14GB). 캐시 위치 지정 권장:

```bash
export HF_HOME=/data/huggingface_cache
```

미리 받아두려면:
```bash
huggingface-cli download lmms-lab/LLaVA-NeXT-Video-7B-Qwen2 \
  --local-dir /data/huggingface_cache/llava-next-video-7b-qwen2
```

#### 검증

```bash
ls -lh /data/CUT3R/src/cut3r_512_dpt_4_64.pth                # 약 600MB
ls -lh /data/vlm-3r-llava-qwen2-lora/adapter_model.bin       # 617MB
ls -lh /data/vlm-3r-llava-qwen2-lora/non_lora_trainables.bin # 51MB
```

세 파일 크기가 위와 같으면 정상. 자세한 트러블슈팅은 [setup.md](../setup.md) §A.6 ~ §A.7 참조.

### 2.6. 실모델 모드 실행 (GPU 필요)

모델 통합 작업은 [TODO.md](TODO.md) §C 참고. GPU 서버(Python 3.10 + CUDA 12.1)에서:

```bash
# 1) 가상환경 (Python 3.10)
python3.10 -m venv .venv
source .venv/bin/activate

# 2) 기본 + 실모델 의존성 설치
pip install -r requirements.txt
pip install -r requirements-ml.txt
# PyTorch는 cu121 인덱스로 직접 설치 권장:
pip install torch==2.1.1 torchvision==0.16.1 \
  --index-url https://download.pytorch.org/whl/cu121
# flash-attn 은 wheel URL 직접 설치 권장 (자세한 셋업은 ../setup.md 참고)

# 3) 환경변수 (모델 경로·캐시·stub 해제)
export STUB_MODELS=false
export MODEL_PATH=/data/vlm-3r-llava-qwen2-lora
export CUT3R_WEIGHTS=/data/CUT3R/src/cut3r_512_dpt_4_64.pth
export HF_HOME=/data/huggingface_cache

# 4) 서버 기동
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **`--workers 1` 필수**: 모델이 GPU에 1번만 로드되어야 함. N개 워커 = VRAM N배.

---

## 3. 프로젝트 구조

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

### 3.1. 폴더 책임 분리

| 폴더 | 의존 가능 | 의존 금지 | 핵심 |
|------|-----------|-----------|------|
| `app/routers/` | `services/`, `schemas/`, `core/` | `vlm/` 직접 호출 | HTTP만 알고 모델은 모름 |
| `app/services/` | `vlm/`, `core/`, `schemas/` | FastAPI 객체(Request 등) | HTTP와 모델 사이 비즈니스 로직 |
| `app/vlm/` | `app/vlm/llava`, `app/vlm/CUT3R`, `core/config` | FastAPI, 비즈니스 규칙 | HTTP 무관, 단독 CLI 호출도 가능해야 함 |
| `app/core/` | (모든 모듈에서 임포트) | 다른 `app/*` 폴더 | 설정·라이프사이클 같은 공통 |
| `app/schemas/` | (순수 Pydantic) | 비즈니스 로직 | 데이터 형태만 정의 |

---

## 통신 방식 개요

BE ↔ AI 통신은 **두 채널**로 나뉜다. 아래 표가 전체 그림이다.

| 구분 | 채널 | 방향 | 동기성 | 해당 섹션 |
|------|------|------|--------|-----------|
| 전처리 요청 | HTTP `POST /api/v1/preprocess` | BE → AI | 비동기 (202 즉시 반환) | **§4 (HTTP)** |
| 전처리 완료/실패 통보 | RabbitMQ `analysis.completed` / `analysis.failed` | AI → BE | 비동기 | **§5 (RabbitMQ)** |
| 질문/응답 | HTTP `POST /api/v1/chat` | BE → AI | 동기 (응답 즉시 반환) | **§4 (HTTP)** |
| 작업 상태 폴링 | HTTP `GET /api/v1/jobs/{job_id}` | BE → AI | 동기 | **§4 (HTTP)** |
| 헬스 체크 | HTTP `GET /health` | (외부) → AI | 동기 | **§4 (HTTP)** |

요약:
- **§4 = HTTP** — BE가 AI를 호출하는 모든 요청/응답 (요청 방향: BE → AI)
- **§5 = RabbitMQ** — AI가 전처리 결과를 BE에 되돌려주는 비동기 메시지 (방향: AI → BE)

전처리 한 건의 전체 흐름:
```
BE ──HTTP POST /preprocess──▶ AI        (§4)
BE ◀──── 202 Accepted ────── AI         (§4)
                              AI: 백그라운드 처리
BE ◀═══ RabbitMQ analysis.completed ═══ AI   (§5)
```

---

## 4. HTTP API (동기 요청 · BE → AI)

> 본 절(§4) 전체가 **HTTP** 통신이다. RabbitMQ 메시지 규약은 §5를 참조.
> 모든 에러 응답은 FastAPI 규약에 따라 최상위 `detail` 키로 래핑된다.
> (예: `{"detail": { ... }}`)

### 4.1. `POST /api/v1/preprocess` — 영상 전처리

**Request 필드**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `job_id` | string | ✅ | 작업 고유 ID (UUID) |
| `user_id` | string | ✅ | 사용자 ID |
| `video_url` | string | ✅ | S3 Pre-signed GET URL |
| `metadata` | object | ✕ | 영상 메타데이터 (아래) |
| `metadata.video_duration_sec` | int | ✕ | 영상 길이(초) |
| `metadata.video_resolution` | string | ✕ | 해상도 (예: `1080p`) |
| `metadata.uploaded_at` | datetime | ✕ | 업로드 시각 (ISO8601) |

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "u_abc123",
  "video_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/videos/xxx.mp4?X-Amz-...",
  "metadata": {
    "video_duration_sec": 12,
    "video_resolution": "1080p",
    "uploaded_at": "2026-05-19T10:00:00Z"
  }
}
```

**Response — `202 Accepted`**

| 필드 | 타입 | 설명 |
|------|------|------|
| `job_id` | string | 요청한 작업 ID |
| `status` | string | 항상 `"accepted"` |
| `estimated_time_sec` | int | 예상 처리 시간(초), 기본 45 |
| `received_at` | datetime | 수신 시각 |

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "accepted",
  "estimated_time_sec": 45,
  "received_at": "2026-05-19T10:00:01Z"
}
```

**Response — `409 Conflict`** (중복 `job_id`)

```json
{
  "detail": {
    "code": "DUPLICATE_JOB_ID",
    "message": "이미 처리 중인 job_id입니다",
    "details": "job_id: 550e8400-..."
  }
}
```

전처리 진행은 `BackgroundTasks`로 수행된다. **여기까지가 HTTP 응답이며, 처리 완료/실패는 HTTP가 아니라 RabbitMQ로 별도 통보된다 → §5 참조.**

> **stub 모드 동작**: 요청·응답(202)·RabbitMQ 발행은 실모드와 동일하게 수행된다. 차이는 CUT3R 추출이 더미 텐서로 대체되고 S3 업로드가 스킵된다는 점뿐이다. 따라서 `analysis.completed` 메시지의 `spatial_features_s3_key`가 가리키는 경로에는 실제 파일이 존재하지 않는다.

### 4.2. `POST /api/v1/chat` — 자연어 질문 응답

**Request 필드**

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `job_id` | string | ✅ | — | 전처리 완료된 영상의 job_id |
| `user_id` | string | ✅ | — | 사용자 ID |
| `question` | string | ✅ | — | 사용자 질문 |
| `history` | array | ✕ | `[]` | 이전 대화 (`{role, content}` 배열) |
| `history[].role` | string | — | — | `"user"` 또는 `"assistant"` |
| `history[].content` | string | — | — | 메시지 내용 |
| `max_new_tokens` | int | ✕ | 512 | 최대 생성 토큰 수 |
| `temperature` | float | ✕ | 0.0 | 샘플링 온도 |

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
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

**Response — `200 OK`**

| 필드 | 타입 | 설명 |
|------|------|------|
| `job_id` | string | 요청한 작업 ID |
| `answer` | string | 모델 응답 (stub 모드 시 `[stub]` 접두 고정 문구) |
| `metadata.tokens_used` | int | 생성 토큰 수 |
| `metadata.inference_time_ms` | int | 추론 소요 시간(ms), stub 시 0 |
| `metadata.spatial_cache_hit` | bool | 공간 특징 캐시 적중 여부 |
| `metadata.stub` | bool | stub 응답이면 `true` |

실모드 (`STUB_MODELS=false`):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "answer": "소파는 영상 중앙의 창문 앞에 위치해 있습니다.",
  "metadata": {
    "tokens_used": 142,
    "inference_time_ms": 487,
    "spatial_cache_hit": true,
    "stub": false
  }
}
```

stub 모드 (`STUB_MODELS=true`, 현재 기본값):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "answer": "[stub] '소파는 어디에 있나요?'에 대한 답변입니다. 현재는 실모델이 로드되지 않은 stub 모드입니다.",
  "metadata": {
    "tokens_used": 73,
    "inference_time_ms": 0,
    "spatial_cache_hit": false,
    "stub": true
  }
}
```

> **응답 구조·필드·타입은 두 모드가 100% 동일하다.** 값만 다르다: `answer`(stub은 `[stub]` 접두 고정 문구), `metadata.stub`(stub은 `true`), `inference_time_ms`(stub은 0). BE는 별도 stub용 명세 없이 동일 스키마로 통합 테스트 후, 모델 통합 시 환경변수만 전환하면 된다. `metadata.stub` 필드로 현재 모드를 구분·로깅할 수 있다.

**Response — `404 Not Found`** (존재하지 않는 `job_id`)

```json
{ "detail": { "code": "JOB_NOT_FOUND", "message": "해당 job_id를 찾을 수 없습니다" } }
```

**Response — `425 Too Early`** (전처리 미완료)

```json
{ "detail": { "code": "PREPROCESS_NOT_READY", "message": "전처리 미완료 (현재 status=processing)" } }
```

### 4.3. `GET /api/v1/jobs/{job_id}` — 작업 상태 조회

폴링용. RabbitMQ 메시지 누락 시 BE의 복구 경로로 사용한다.

**Response — `200 OK`**

`data` 필드는 작업 진행 단계에 따라 동적으로 확장된다.

| 필드 | 타입 | 항상 존재 | 설명 |
|------|------|-----------|------|
| `success` | bool | ✅ | 항상 `true` |
| `data.job_id` | string | ✅ | 작업 ID |
| `data.user_id` | string | ✅ | 사용자 ID |
| `data.status` | string | ✅ | `accepted` → `downloading` → `processing` → `completed` / `failed` |
| `data.progress` | int | ✅ | 진행률 0~100 |
| `data.current_step` | string | ✅ | `download_video` / `cut3r_extract` / `upload_features` / `publish_result` |
| `data.started_at` | datetime | ✅ | 작업 시작 시각 |
| `data.updated_at` | datetime | ✅ | 마지막 갱신 시각 |
| `data.video_path` | string | 처리 중 이후 | 로컬 다운로드 경로 |
| `data.spatial_features_s3_key` | string | 완료 시 | 추출 결과 S3 키 |
| `data.completed_at` | datetime | 완료 시 | 완료 시각 |
| `data.error_code` | string | 실패 시 | 에러 코드 (§5.3) |
| `data.error_message` | string | 실패 시 | 에러 메시지 |
| `data.failed_at` | datetime | 실패 시 | 실패 시각 |

```json
{
  "success": true,
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "u_abc123",
    "status": "completed",
    "progress": 100,
    "current_step": "publish_result",
    "started_at": "2026-05-19T10:00:01Z",
    "updated_at": "2026-05-19T10:00:30Z",
    "spatial_features_s3_key": "spatial_features/u_abc123/550e8400-....pt",
    "completed_at": "2026-05-19T10:00:30Z"
  }
}
```

**Response — `404 Not Found`**

> 주의: 이 엔드포인트의 404 body는 다른 엔드포인트와 구조가 다르다 (`detail` 안에 `success`/`error` 가 중첩됨). 향후 통일 대상 (TODO B4).

```json
{
  "detail": {
    "success": false,
    "error": { "code": "JOB_NOT_FOUND", "message": "해당 job_id를 찾을 수 없습니다" }
  }
}
```

### 4.4. `GET /health` — 헬스 체크

**Response — `200 OK`**

| 필드 | 타입 | 설명 |
|------|------|------|
| `success` | bool | 항상 `true` |
| `data.status` | string | `"healthy"` |
| `data.version` | string | 서버 버전 |
| `data.checked_at` | datetime | 확인 시각 |
| `data.dependencies.ai_model_loaded` | bool | 모델 로드 여부 (현재 고정 `false`, TODO A4) |
| `data.dependencies.rabbitmq_connected` | bool | MQ 연결 여부 (현재 고정 `false`, TODO A4) |
| `data.dependencies.s3_accessible` | bool | S3 접근 가능 여부 (현재 고정 `false`, TODO A4) |

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "checked_at": "2026-05-19T10:00:00Z",
    "dependencies": {
      "ai_model_loaded": false,
      "rabbitmq_connected": false,
      "s3_accessible": false
    }
  }
}
```

> `dependencies` 값은 현재 실제 상태를 반영하지 않고 고정값이다. 실상태 반영 및 `/ready` 분리는 TODO A4·A5에서 진행.

---

## 5. RabbitMQ 메시지 계약 (비동기 통보 · AI → BE)

> 여기서부터는 **HTTP가 아니다.** AI 서버가 전처리 작업을 마친 뒤 BE에게
> 결과를 알리기 위해 RabbitMQ 메시지 큐로 발행하는 메시지의 규약이다.
> HTTP 요청/응답 명세는 §4를 참조.

### 5.1. Exchange / Queue

| 항목 | 값 |
|------|-----|
| Exchange | `analysis_exchange` (topic, durable) |
| Queue | `analysis_results` (durable) |
| Binding | `analysis.*` |

### 5.2. Routing Key & Payload

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

### 5.3. 에러 코드

| code | 의미 |
|------|------|
| `INVALID_VIDEO_URL` | Pre-signed URL 접근 실패 |
| `VIDEO_DOWNLOAD_FAILED` | 다운로드 중 네트워크/타임아웃 |
| `VIDEO_TOO_LARGE` | 500MB 초과 |
| `UNSUPPORTED_VIDEO_FORMAT` | mp4/mov/avi 외 |
| `INTERNAL_SERVER_ERROR` | 알 수 없는 오류 |

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
