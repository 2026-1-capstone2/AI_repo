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

## 2. 설치 및 실행 (실 사용 기준)

GPU 서버에서 실모델 모드로 동작시키는 표준 절차. 개발 환경(GPU 없는 노트북)에서 흐름만 검증하려면 §3 stub 모드를 참고한다.

### 2.1. 요구사항

| 항목 | 버전·사양 |
|------|----------|
| **Python** | 3.10 (PyTorch 2.1.1 wheel 호환) |
| **CUDA** | 12.1 |
| **GPU** | NVIDIA, Compute Capability 8.0+ (Ampere 이상), VRAM 8 GB+ |
| **PyTorch** | 2.1.1 + cu121 |
| **컴파일러** | gcc/g++ 11 (CUT3R curope C++ 확장 빌드용) |
| **OS** | Ubuntu 22.04 권장. 24.04 사용 시 §2.3.1 (CUDA 12.1 toolkit 부분 설치 + gcc-11) 단계 추가 필요 |
| **기타** | git-lfs (LoRA 가중치 LFS), gdown (Google Drive 가중치), ninja (CUT3R 빌드용) |

자세한 환경별 셋업 트러블슈팅은 [docs/install-issues.md](docs/install-issues.md), [setup.md](../setup.md) 참고.

### 2.2. 클론

```bash
git clone https://github.com/2026-1-capstone2/AI_repo.git
cd AI_repo
```

### 2.3. 가상환경 + 의존성 설치

> ⚠️  **설치 순서를 반드시 지킬 것.** PyTorch (cu121) 를 먼저 설치한 뒤
> `requirements-ml.txt` 를 깔아야 한다. 순서를 바꾸면 `open-clip-torch`
> 가 무핀 의존성을 끌어와 사전 설치한 torch 2.1.1+cu121 을 silent 로
> 최신/cu13 wheel 로 덮어쓰며, 그 시점 이후 flash-attn 과 VLM-3R 코드
> 가정이 모두 깨진다. (사고 사례: [docs/install-issues.md §A3](docs/install-issues.md))

```bash
# 1) Python 3.10 가상환경
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 2) 기본 의존성 (FastAPI / RabbitMQ / httpx / boto3 등)
pip install -r requirements.txt

# 3) PyTorch (cu121 wheel — 별도 인덱스 필수)
pip install torch==2.1.1 torchvision==0.16.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 4) flash-attn (cu12 + torch2.1 + cpython 3.10 전용 wheel)
#    requirements-ml.txt 전에 깔아야 한다. (open-clip-torch 가 끌어올 가능성 차단)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 5) 실모델 의존성 (transformers / open-clip-torch / matplotlib / av / omegaconf 등)
pip install -r requirements-ml.txt

# 6) 모델 가중치 다운로드용 도구
sudo apt install git-lfs ninja-build -y      # ninja 는 CUT3R curope C++ 확장 빌드용
pip install gdown
```

설치 검증:
```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# → torch 2.1.1+cu121 cuda True
python -c "import flash_attn, transformers, open_clip; print('flash-attn', flash_attn.__version__, '/ transformers', transformers.__version__)"
# → flash-attn 2.7.1.post1 / transformers 4.40.0
```

이 시점에 torch 가 `2.1.1+cu121` 가 아니라면 4)~5) 순서가 깨졌거나, `requirements-ml.txt` 의 무핀 패키지가 다시 도입된 것이다. 핀 변경을 의심하고 [docs/install-issues.md §A3](docs/install-issues.md) 절차로 복구한다.

### 2.3.1. 환경별 추가 셋업 (권장 환경이 아닌 경우)

권장 환경(Ubuntu 22.04 + CUDA 12.1 + gcc 11)에서는 본 절을 건너뛴다. **Ubuntu 24.04** 또는 **시스템 기본 CUDA toolkit 이 12.1 이 아닌 경우** 다음 추가 단계가 필요하다 ([docs/install-issues.md §B](docs/install-issues.md) 참조).

#### CUDA 12.1 toolkit 부분 설치

`/usr/local/cuda` 가 12.1 이 아니면 PyTorch 2.1.1 cu121 wheel 과 nvcc 가 major mismatch 로 CUT3R 빌드 시 거부된다 (`The detected CUDA version (X) mismatches the version that was used to compile PyTorch (12.1)`). NVIDIA apt repo 추가 후 12.1 toolkit 만 별도 설치:

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-nvcc-12-1 cuda-cudart-dev-12-1 cuda-libraries-dev-12-1
```

#### gcc 11 설치

CUDA 12.1 nvcc 는 gcc ≤12 만 공식 지원하며, gcc 12 는 PyTorch 2.1.1 동봉 pybind11 헤더와 C++ 템플릿 파싱 충돌이 있다 → **gcc 11 이 유일한 검증 조합**.

```bash
sudo apt-get install -y gcc-11 g++-11
```

이 두 단계가 끝나면 CUT3R curope 빌드 시 다음과 같이 환경변수를 명시:
```bash
cd app/vlm/CUT3R/src/croco/models/curope
rm -rf build curope*.so
CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH \
CC=gcc-11 CXX=g++-11 python setup.py build_ext --inplace
```

결과: `curope.cpython-310-x86_64-linux-gnu.so` 생성 (deprecation 경고는 무해).

### 2.4. 모델 가중치 다운로드

세 가지 가중치(약 15~16 GB)를 받아야 한다. 저장 경로는 `settings.cut3r_weights`, `settings.model_path` 의 기본값과 일치시키거나 환경변수로 재정의한다.

| 가중치 | 출처 | 크기 | 저장 경로 (기본값) |
|--------|------|------|-------------------|
| **CUT3R** | Google Drive | ~600 MB | `/data/CUT3R/src/cut3r_512_dpt_4_64.pth` |
| **VLM-3R LoRA** | HuggingFace `Journey9ni/vlm-3r-llava-qwen2-lora` | ~700 MB | `/data/vlm-3r-llava-qwen2-lora/` |
| **베이스 모델** (LLaVA-NeXT-Video-7B-Qwen2) | HuggingFace `lmms-lab/LLaVA-NeXT-Video-7B-Qwen2` | ~14 GB | HuggingFace 캐시 (자동) |

#### CUT3R

```bash
sudo mkdir -p /data/CUT3R/src && sudo chown $USER /data/CUT3R/src
cd /data/CUT3R/src
gdown 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD       # cut3r_512_dpt_4_64.pth
```

> gdown 6.x 부터 `--fuzzy` 옵션이 제거되었으므로 파일 ID 만 직접 전달.

#### VLM-3R LoRA

```bash
git lfs install                # 1회만. 미설치 시 134바이트 LFS 포인터만 받음
cd /data
git clone https://huggingface.co/Journey9ni/vlm-3r-llava-qwen2-lora
cd vlm-3r-llava-qwen2-lora
git lfs pull                   # adapter_model.bin (617MB), non_lora_trainables.bin (51MB)
```

#### 베이스 모델 (자동 다운로드)

서버 첫 기동 시 HuggingFace 에서 자동 다운로드된다. 캐시 위치 지정 권장:

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
ls -lh /data/CUT3R/src/cut3r_512_dpt_4_64.pth                # ~600 MB
ls -lh /data/vlm-3r-llava-qwen2-lora/adapter_model.bin       # 617 MB
ls -lh /data/vlm-3r-llava-qwen2-lora/non_lora_trainables.bin # 51 MB
```

세 파일 크기가 위와 같으면 정상. 자세한 트러블슈팅은 [setup.md](../setup.md) §A.6 ~ §A.7 참고.

### 2.5. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 만들거나 셸에서 export 한다.

```bash
export STUB_MODELS=false
export MODEL_PATH=/data/vlm-3r-llava-qwen2-lora
export CUT3R_WEIGHTS=/data/CUT3R/src/cut3r_512_dpt_4_64.pth
export HF_HOME=/data/huggingface_cache

# Spring BE 연동
export INTERNAL_API_TOKEN=<공유 비밀값>
export RABBITMQ_HOST=<MQ 호스트>
export RABBITMQ_USER=<...>
export RABBITMQ_PASSWORD=<...>
```

> 전체 환경변수 목록은 §7 환경 설정 참고.

### 2.6. 서버 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **`--workers 1` 필수**: 모델이 GPU 에 1번만 로드되어야 한다. N개 워커 = VRAM N배.

부팅 로그 확인:
```
[loader] VLM loaded. VRAM=...GB
[loader] CUT3R loaded on cuda. VRAM=...GB
INFO:     Application startup complete.
```

### 2.7. 동작 검증

```bash
# 1) 헬스 체크
curl http://localhost:8000/health
#   → ai_model_loaded=true, cut3r_loaded=true

# 2) 전처리 요청 (BE 발급 Pre-signed GET URL)
curl -X POST http://localhost:8000/api/v1/preprocess \
  -H "X-Internal-Token: <설정한 토큰>" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-1",
    "user_id": "u_demo",
    "video_url": "<S3 Pre-signed GET URL>"
  }'

# 3) 작업 상태 폴링
curl http://localhost:8000/api/v1/jobs/test-1

# 4) 채팅 (실모델 응답, metadata.stub == false)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "X-Internal-Token: <설정한 토큰>" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-1",
    "user_id": "u_demo",
    "question": "방의 크기는 얼마인가요?"
  }'
```

전체 GPU 서버 단계별 검증 체크리스트(V1~V5) 는 [TODO.md §C-VERIFY](TODO.md) 참고.

---

## 3. stub 모드 (부가 — GPU 없는 개발 환경용)

`settings.stub_models=True` 인 상태로 서버를 띄우면 실모델·GPU 없이도 HTTP·RabbitMQ 흐름을 검증할 수 있다. 모델 호출 부분만 더미 응답으로 대체되고 라우팅·인증·메시지 발행은 실모드와 동일하게 동작한다.

### 3.1. 가상환경 (Python 3.10+ 또는 3.13)

PyTorch 와 모델 의존성(`requirements-ml.txt`) 은 설치하지 않는다.

```bash
git clone https://github.com/2026-1-capstone2/AI_repo.git
cd AI_repo

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # 기본만

# (선택) RabbitMQ 로컬 띄우기
docker run -d --name rmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management

# 서버 실행 — STUB_MODELS=true 가 기본값
uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

### 3.2. 통신 테스트

```bash
curl -X POST http://localhost:8000/api/v1/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-job-1",
    "user_id": "u_demo",
    "video_url": "https://example.com/test.mp4"
  }'

curl http://localhost:8000/api/v1/jobs/test-job-1

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-job-1",
    "user_id": "u_demo",
    "question": "방의 크기는?"
  }'
```

`/chat` 응답에 `metadata.stub == true` 가 표시된다.

### 3.3. ngrok 외부 노출

팀원 데모용으로 로컬 stub 서버를 공개 URL 로 노출:

```bash
python start_ngrok.py
```

---

## 4. 프로젝트 구조

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

### 4.1. 폴더 책임 분리

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
| 전처리 요청 | HTTP `POST /api/v1/preprocess` | BE → AI | 비동기 (202 즉시 반환) | **§5 (HTTP)** |
| 전처리 완료/실패 통보 | RabbitMQ `analysis.completed` / `analysis.failed` | AI → BE | 비동기 | **§6 (RabbitMQ)** |
| 질문/응답 | HTTP `POST /api/v1/chat` | BE → AI | 동기 (응답 즉시 반환) | **§5 (HTTP)** |
| 작업 상태 폴링 | HTTP `GET /api/v1/jobs/{job_id}` | BE → AI | 동기 | **§5 (HTTP)** |
| 헬스 체크 | HTTP `GET /health` | (외부) → AI | 동기 | **§5 (HTTP)** |

요약:
- **§5 = HTTP** — BE가 AI를 호출하는 모든 요청/응답 (요청 방향: BE → AI)
- **§6 = RabbitMQ** — AI가 전처리 결과를 BE에 되돌려주는 비동기 메시지 (방향: AI → BE)

전처리 한 건의 전체 흐름:
```
BE ──HTTP POST /preprocess──▶ AI        (§5)
BE ◀──── 202 Accepted ────── AI         (§5)
                              AI: 백그라운드 처리
BE ◀═══ RabbitMQ analysis.completed ═══ AI   (§6)
```

---

## 5. HTTP API (동기 요청 · BE → AI)

> 본 절(§5) 전체가 **HTTP** 통신이다. RabbitMQ 메시지 규약은 §6을 참조.
> 모든 에러 응답은 FastAPI 규약에 따라 최상위 `detail` 키로 래핑된다.
> (예: `{"detail": { ... }}`)

### 5.1. `POST /api/v1/preprocess` — 영상 전처리

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

전처리 진행은 `BackgroundTasks`로 수행된다. **여기까지가 HTTP 응답이며, 처리 완료/실패는 HTTP가 아니라 RabbitMQ로 별도 통보된다 → §6 참조.**

> **stub 모드 동작**: 요청·응답(202)·RabbitMQ 발행은 실모드와 동일하게 수행된다. 차이는 CUT3R 추출이 더미 텐서로 대체되고 S3 업로드가 스킵된다는 점뿐이다. 따라서 `analysis.completed` 메시지의 `spatial_features_s3_key`가 가리키는 경로에는 실제 파일이 존재하지 않는다.

### 5.2. `POST /api/v1/chat` — 자연어 질문 응답

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

### 5.3. `GET /api/v1/jobs/{job_id}` — 작업 상태 조회

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
| `data.error_code` | string | 실패 시 | 에러 코드 (§6.3) |
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

### 5.4. `GET /health` — 헬스 체크

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

## 6. RabbitMQ 메시지 계약 (비동기 통보 · AI → BE)

> 여기서부터는 **HTTP가 아니다.** AI 서버가 전처리 작업을 마친 뒤 BE에게
> 결과를 알리기 위해 RabbitMQ 메시지 큐로 발행하는 메시지의 규약이다.
> HTTP 요청/응답 명세는 §5를 참조.

### 6.1. Exchange / Queue

| 항목 | 값 |
|------|-----|
| Exchange | `analysis_exchange` (topic, durable) |
| Queue | `analysis_results` (durable) |
| Binding | `analysis.*` |

### 6.2. Routing Key & Payload

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

### 6.3. 에러 코드

| code | 의미 |
|------|------|
| `INVALID_VIDEO_URL` | Pre-signed URL 접근 실패 |
| `VIDEO_DOWNLOAD_FAILED` | 다운로드 중 네트워크/타임아웃 |
| `VIDEO_TOO_LARGE` | 500MB 초과 |
| `UNSUPPORTED_VIDEO_FORMAT` | mp4/mov/avi 외 |
| `INTERNAL_SERVER_ERROR` | 알 수 없는 오류 |

---

## 7. 환경 설정

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

## 8. Debug — 4-modality 흐름 진단 (VLM_DEBUG)

VLM-3R 추론에는 4종 입력이 LLM 으로 들어가야 한다:

| 종류 | 출처 | 변수 |
|------|------|------|
| ① 텍스트 | 질문 + history → Qwen2 tokenizer | `input_ids` |
| ② 시각 | 영상 프레임 → SigLip vision_tower | `image_features` |
| ③ 카메라 | CUT3R `camera_tokens` (F,1,768) | `spatial_features[0]["camera_tokens"]` |
| ④ 공간 | CUT3R `patch_tokens` (F,729,768) | `spatial_features[0]["patch_tokens"]` |

③④ 는 `encode_images()` 안에서 `fusion_block(cross_attention)` 으로 ② 와 융합된 뒤
`mm_projector` 를 거쳐 ① 의 `IMAGE_TOKEN_INDEX` 위치에 주입된다.

### 8.1. 흔한 silent 실패 3가지

| 함정 | 증상 | 원인 |
|------|------|------|
| `spatial_tower` 가 `None` | 환각, 3D 질문에 일반론 | `disable_spatial_tower_in_vlm` 잘못 적용 / 설정에서 spatial_tower 가 빈 값 |
| `fusion_block` 가중치 누락 | 답변이 무관한 방향, 반복 | builder.py 의 non-LoRA 가중치 로드 실패 → 랜덤 가중치 |
| `spatial_features` kwarg drop | CUT3R 결과 무시 | `model.generate` → `encode_images` 체인 중간에서 kwarg 가 silent 하게 제거 |

### 8.2. 진단 로그 켜기

[app/vlm/llava/model/llava_arch.py](app/vlm/llava/model/llava_arch.py) `encode_images()` 에 한 줄 진단 훅이 들어 있다.
환경변수로 활성화:

```bash
VLM_DEBUG=1 uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

`/chat` 요청을 한 번 보내면 매 forward 마다 한 줄씩 출력:

```
[VLM_DEBUG encode_images] spatial_tower=Cut3rEncoder fusion_block=CrossAttention \
spatial_features=YES(1) spatial_encoder_type=cut3r fusion_type=cross_attention \
select_feature=all
```

### 8.3. 기대값과 어긋남 해석

| 필드 | 정상 | 어긋남 → 원인 |
|------|------|----------------|
| `spatial_tower` | `Cut3rEncoder` (또는 유사 클래스) | `None` → 함정 #1 |
| `fusion_block` | `CrossAttentionBlock` 등 | `None` → 융합 자체 비활성 |
| `spatial_features` | `YES(1)` | `NO` → 함정 #3 (kwarg drop) |
| `spatial_encoder_type` | `cut3r` | 빈 값/다른 값 → fusion 분기 못 탐 |
| `fusion_type` | `cross_attention` | 다른 값 → 다른 분기, 코드 점검 필요 |
| `select_feature` | `all` | `patch_tokens` 만이면 카메라 정보 미사용 |

### 8.4. fusion_block 가중치 로드 확인

부팅 로그에서 다음 중 하나가 보여야 정상:

```bash
grep -iE "non.lora|fusion_block|Incompatible" <서버 로그>
```

기대 패턴:
- `Loaded non-LoRA trainables` — LoRA 어댑터 + 추가 가중치 동시 로드
- `Loaded fusion block weights from ...` — fusion_block 별도 로드
- 또는 builder.py 의 `setattr-loop` 통과 메시지

전부 없으면 fusion_block 이 랜덤 초기화 상태로 추론 중일 가능성 → 환각의 직접 원인이 된다.

### 8.5. 4-bit 양자화 영향 확인

`LOAD_4BIT=true` 디폴트는 8GB VRAM 환경 대응용이다. 16GB 이상 GPU 라면 `.env` 에서:

```env
LOAD_4BIT=false
```

로 끄고 fp16 로 띄우면 품질 손실이 즉시 줄어든다 (필요 VRAM ~14GB).

---

## 9. 작업 상태

### 9.1. 완료
- [x] FastAPI 골격 + 라우터 분리 (`/preprocess`, `/chat`, `/jobs`, `/health`)
- [x] Pydantic schemas (preprocess, chat, MQ payload)
- [x] RabbitMQ publisher (lazy connect, `analysis.*` routing)
- [x] httpx 기반 영상 다운로드 (Pre-signed URL)
- [x] 인메모리 `job_store`
- [x] `services/`: preprocess / chat / cache / s3 (boto3 lazy import)
- [x] `vlm/`: loader / inference / cut3r_runner / preprocessing 골격
- [x] stub 모드 (GPU·실모델 없이 전 흐름 시뮬레이션 가능)
- [x] FastAPI lifespan (모델 로드/해제)

### 9.2. 다음 작업

[TODO.md](TODO.md) 에서 우선순위별로 관리:

- **A: stub 모드로 가능** (인증·헬스·테스트·메시지 계약 문서)
- **B: BE 실통합** (RabbitMQ·Pre-signed URL·에러 코드 합의)
- **C: VLM-3R 모델 통합** (llava/CUT3R 이식 → 실모델 로드 → e2e 검증)
- **D: 안정화** (Dockerfile · 로깅 · 메트릭 · OOM 가드)
- **E: 배포** (GPU 인스턴스 · IAM · 도메인 · k8s/ECS)

---

## 10. 관련 문서

- [TODO.md](TODO.md) — 우선순위별 작업 리스트 (C-VERIFY: GPU 서버 검증 체크리스트 포함)
- [docs/install-issues.md](docs/install-issues.md) — 셋업 중 마주친 이슈 및 환경별 추가 단계 (A: 프로젝트 / B: 환경)
- [../setup.md](../setup.md) — 본 레포(VLM-3Rdemo) 표준 환경 셋업 트러블슈팅
