# AI Server 설치 / 초기화 이슈 정리

본 문서는 GPU 서버(`/root/AI_repo`)에서 README §2 절차에 따라 셋업을 진행하면서 마주친 모든 이슈를 두 카테고리로 분리해 기록한다:

- **A. 프로젝트 자체 이슈** — 다른 환경에서도 동일하게 발생. 코드/requirements 수정으로 정식 패치 필요.
- **B. 검증 환경 이슈** — 본 PC 의 OS/드라이버/컴파일러가 README 권장과 다른 데서 비롯. 권장 환경 (Ubuntu 22.04 + CUDA 12.1 + gcc 11) 에서는 발생하지 않음.

마지막에 **진행 현황** 과 **다음 단계 (V2~V5)** 를 정리한다.

---

## 검증 환경

| 항목 | 값 |
|------|-----|
| OS | Ubuntu 24.04.3 LTS (README 권장: 22.04) |
| GPU | NVIDIA A100-SXM4-80GB (VRAM 80GB) |
| 드라이버 | 580.105.08 |
| `nvidia-smi` CUDA | 13.0 |
| 기본 gcc/g++ | 13.3.0 (README 권장: gcc 11) |
| Python | 3.10.19 (`/usr/bin/python3.10`) |

---

# A. 프로젝트 자체 이슈 (모든 환경)

`requirements-ml.txt`, 소스 코드의 import 경로 등 — 권장 환경에서도 그대로 깨진다. 정식 패치 대상.

## A1. `transformers==4.40.0.dev0` 가 PyPI에 없음

**증상**
```
ERROR: Could not find a version that satisfies the requirement transformers==4.40.0.dev0
```

**원인**
`requirements-ml.txt` line 25 가 dev 릴리즈를 고정. dev 릴리즈는 PyPI 영구 호스팅 안 됨. 본 레포가 dev0 을 명시한 이유는 LLaVA-NeXT-Video 가 4.40 dev 시점에 머지된 후 정식 릴리즈에서 일부 종속성 요구사항이 변경됐기 때문으로 추정.

**해결**
정식 `transformers==4.40.0` 으로 변경 (A2 와 연쇄).

**정식 패치 후보**
- `requirements-ml.txt::transformers==4.40.0.dev0` → `transformers==4.40.0`

---

## A2. `tokenizers==0.15.2` ↔ `transformers==4.40.0` 충돌

**증상**
```
ERROR: Cannot install tokenizers==0.15.2 and transformers==4.40.0
The conflict is caused by:
    transformers 4.40.0 depends on tokenizers<0.20 and >=0.19
```

**원인**
transformers dev0 시점에는 tokenizers 0.15.2 와 호환됐으나, 정식 4.40.0 릴리즈가 ≥0.19 로 요구사항을 강화함.

**해결**
`tokenizers` 핀을 제거하고 pip 자동 해결로 위임 → tokenizers 0.19.1 설치됨.

**정식 패치 후보**
- `tokenizers==0.15.2` → `tokenizers>=0.19,<0.20`

---

## A3. `open-clip-torch` 무핀으로 torch 2.1.1 → 2.12.0 강제 업그레이드

**증상 (가장 위험)**
`pip install -r requirements-ml.txt` 가 진행되는 동안:
```
Attempting uninstall: torch
  Found existing installation: torch 2.1.1+cu121
Successfully installed ... torch-2.12.0 torchvision-0.27.0 triton-3.7.0
  nvidia-cublas-13.1.1.3 ... cuda-toolkit-13.0.2 ...
```
사전 설치한 `torch 2.1.1+cu121` 이 silent 하게 덮어쓰여, flash-attn (torch 2.1 wheel) 과 VLM-3R 코드 가정이 모두 깨진다.

**원인**
`requirements-ml.txt::open-clip-torch` 버전 미지정. 최신 `open-clip-torch 3.3.0` 이 최신 torch / CUDA 13 스택을 끌어옴.

**해결**
1. torch/torchvision/triton/cuda-13 관련 패키지를 일괄 제거 후 cu121 wheel 로 재설치
2. `open-clip-torch==2.24.0`, `timm==0.9.16`, `bitsandbytes 0.43.3` 으로 torch 2.1 호환 버전 핀

**정식 패치 후보**
```diff
- open-clip-torch
+ open-clip-torch==2.24.0
- timm
+ timm==0.9.16
- bitsandbytes>=0.41.3
+ bitsandbytes>=0.41.3,<0.44
```
설치 순서도 다음과 같이 분리하여 torch 가 덮어쓰이지 않도록:
```bash
pip install -r requirements.txt
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
pip install <flash-attn wheel URL>
pip install -r requirements-ml.txt
```

---

## A4. NumPy 1.x ↔ 2.x 일시적 불일치 (자체 해결)

**증상**
torch 2.1.1 import 시:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

**원인**
base requirements 가 numpy 2.x 를 끌어왔는데 torch 2.1.1 wheel 은 numpy 1.x 빌드.

**해결**
`requirements-ml.txt::numpy==1.26.4` 설치 시 자동 다운그레이드. 별도 조치 불필요.

**비고**
설치 순서를 잘 지키면 자체 해결되지만, 설치 도중에 잠깐 깨진 환경이 노출되는 게 거슬릴 수 있음. base requirements 에 `numpy<2` 를 명시하는 게 더 깔끔.

---

## A5. `requirements-ml.txt` 에 누락된 런타임 의존성

V1 검증 (`load_cut3r()`) 진행 시 다음 모듈들이 순차적으로 누락 에러를 일으킴.

| 패키지 | 누락 시 에러 | 용도 |
|--------|--------------|------|
| `matplotlib` | `Failed to import llava_qwen ... No module named 'matplotlib'` | LLaVA 언어 모델 임포트 경로에서 참조 |
| `av` (pyav) | `Please install pyav to use video processing functions` | LLaVA 비디오 처리 |
| `omegaconf` | `ModuleNotFoundError: No module named 'omegaconf'` (CUT3R 가중치 `torch.load` 시) | 가중치 안에 `omegaconf.DictConfig` pickled |

**해결**
```bash
pip install matplotlib av omegaconf
```

**정식 패치 후보**
`requirements-ml.txt` 의 비전·영상 그룹과 수치 그룹에 다음 추가:
```
matplotlib
av
omegaconf
```

`open3d` 는 `cut3r_spatial_encoder.py` 가 try-import 하지만 미설치 시 point cloud export 만 비활성화되므로 추론에 영향 없음. 옵션.

---

## A6. CUT3R 내부 import 경로 (`from src.dust3r.*`) 미치환

**증상**
```
ModuleNotFoundError: No module named 'src'
```

**원인**
세 파일에 `from src.dust3r.model import ARCroco3DStereo` 가 남아 있음. C4 (`llava.*` → `app.vlm.llava.*` 치환) 가 처리하지 않은 패턴.

| 파일 | line | 코드 |
|------|------|------|
| `app/vlm/llava/model/multimodal_spatial_encoder/cut3r_spatial_encoder.py` | 10~11 | `sys.path.append('CUT3R'); from src.dust3r.model import ARCroco3DStereo` |
| `app/vlm/llava/model/multimodal_spatial_encoder/cut3r_points.py` | 9 | `from src.dust3r.model import ARCroco3DStereo` |
| `app/vlm/scripts/extract_spatial_features.py` | 32 | `from src.dust3r.model import ARCroco3DStereo` |

원본 본 레포에서는 cwd 가 레포 루트이고 `CUT3R/src` 가 sys.path 에 있어서 `src.dust3r.model` 임포트가 동작했지만, ai_repo 에서는 동일 가정 안 됨. 또 `sys.path.append('CUT3R')` 는 ai_repo 구조에서 잘못된 경로.

**해결 (적용 완료)**
세 파일에서 `_CUT3R_SRC` 를 자기 위치 기준 상대 경로로 계산하여 sys.path 에 추가하고, 임포트 경로를 `from dust3r.model import ARCroco3DStereo` 로 변경.

```python
import sys, os
_CUT3R_SRC = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'CUT3R', 'src')  # encoder/points 기준
# scripts/extract_spatial_features.py 는 '..', 'CUT3R', 'src'
if _CUT3R_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_CUT3R_SRC))
from dust3r.model import ARCroco3DStereo
```

`dust3r/model.py` 자체가 `sys.path.append(os.path.dirname(os.path.dirname(__file__)))` 으로 자기 sys.path 를 set up 하므로, 한 번 임포트되면 `from croco.*`, `from models.*` 같은 내부 절대 임포트도 동작.

**정식 패치 후보**
- 위 변경을 그대로 커밋 (TODO §C4 의 후속 패치로)

---

# B. 검증 환경 이슈 (이 PC 의 OS/CUDA/gcc)

README 권장 환경 (Ubuntu 22.04 + CUDA 12.1 + gcc 11) 에서는 발생하지 않음. 본 PC 가 Ubuntu 24.04 + CUDA 13.0 + gcc 13 이라 추가 셋업이 필요했다.

## B1. 시스템 CUDA 13.0 ↔ torch cu121 불일치

**증상 (curope 빌드 시)**
```
RuntimeError:
The detected CUDA version (13.0) mismatches the version that was used to compile
PyTorch (12.1). Please make sure to use the same CUDA versions.
```

**원인**
`/usr/local/cuda` 는 CUDA 13.0 toolkit. PyTorch 2.1.1 은 cu121 wheel 이라서 `torch.utils.cpp_extension._check_cuda_version` 이 major mismatch 거부.

**시도와 실패**
- `pip install nvidia-cuda-nvcc-cu12` 는 wheel 안에 실제 nvcc 바이너리 없음 (`ptxas` 만). 무용.

**해결**
NVIDIA apt repo 추가 후 CUDA 12.1 toolkit 부분 설치:
```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-nvcc-12-1 cuda-cudart-dev-12-1 cuda-libraries-dev-12-1
```
빌드 시:
```bash
CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH ...
```

---

## B2. CUDA 12.1 nvcc ↔ gcc 13/12 호환성

**증상 (gcc 13.3 기본)**
```
host_config.h:132:2: error: #error -- unsupported GNU version!
gcc versions later than 12 are not supported!
```

**증상 (gcc 12.4)**
```
torch/include/pybind11/detail/../cast.h:45:120:
error: expected template-name before '<' token
```

**원인**
- CUDA 12.1 nvcc 는 gcc ≤12 만 공식 지원 → gcc 13 즉시 거부
- gcc 12 는 nvcc 통과하지만 PyTorch 2.1.1 동봉 pybind11 헤더와 C++ 템플릿 파싱 충돌
- README 가 gcc 11 을 권장한 이유는 이 조합 (CUDA 12.1 + torch 2.1.1 + pybind11) 이 gcc 11 에서 검증됐기 때문

**해결**
```bash
apt-get install -y gcc-11 g++-11
```
빌드 시:
```bash
cd app/vlm/CUT3R/src/croco/models/curope
rm -rf build curope*.so
CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH \
CC=gcc-11 CXX=g++-11 python setup.py build_ext --inplace
```
결과: `curope.cpython-310-x86_64-linux-gnu.so` 생성. deprecation 경고는 무해.

---

# 작동 확인된 핀 (V1 통과 환경)

| 항목 | 값 |
|------|-----|
| OS | Ubuntu 24.04.3 LTS |
| Python | 3.10.19 |
| gcc (curope 빌드 전용) | 11.5.0 |
| CUDA toolkit | 12.1.105 (`/usr/local/cuda-12.1`) |
| 드라이버 | 580.105.08 |
| torch | 2.1.1+cu121 |
| torchvision | 0.16.1+cu121 |
| triton | 2.1.0 |
| flash-attn | 2.7.1.post1+cu12torch2.1cxx11abiFALSE |
| transformers | 4.40.0 |
| tokenizers | 0.19.1 |
| accelerate | 0.29.1 |
| peft | 0.4.0 |
| bitsandbytes | 0.43.3 |
| timm | 0.9.16 |
| open-clip-torch | 2.24.0 |
| decord | 0.6.0 |
| numpy | 1.26.4 |
| 추가 설치 | matplotlib, av, omegaconf |

---

# 진행 현황

| # | 단계 | 상태 |
|---|------|------|
| 1 | git-lfs, ninja, build tools 설치 | ✅ |
| 2 | Python 3.10 venv 생성 | ✅ |
| 3 | base + torch cu121 + flash-attn + ML deps 설치 | ✅ |
| 4 | CUT3R 가중치 다운로드 (`/data/CUT3R/src/cut3r_512_dpt_4_64.pth`, 3.0 GB) | ✅ |
| 5 | VLM-3R LoRA 다운로드 (`/data/vlm-3r-llava-qwen2-lora/`, adapter 617 MB + non_lora 51 MB) | ✅ |
| 6 | LLaVA-NeXT-Video-7B-Qwen2 베이스 다운로드 (`/data/huggingface_cache/llava-next-video-7b-qwen2/`, 15 GB) | ✅ |
| 7 | CUT3R curope C++ 확장 빌드 (`curope.cpython-310-x86_64-linux-gnu.so`) | ✅ |
| 8 | `.env` 작성 (`STUB_MODELS=false`, MODEL_PATH 등) | ✅ |
| 9 | **V1: `load_cut3r()`** — Cut3rEncoder/cuda/fp16, VRAM 1.51 GB | ✅ **PASS** |
| 10 | V2: `process_video_with_decord` ↔ `_VideoArgs` 호환성 | ⏳ 다음 |
| 11 | V3: `Cut3rEncoder` forward 시그니처 (camera_tokens [32,1,768] / patch_tokens [32,729,768]) | ⏳ |
| 12 | V4: `/preprocess` e2e (RabbitMQ 발행 + S3 `.pt`) | ⏳ |
| 13 | V5: `/chat` e2e + 캐시 적중 | ⏳ |

`cut3r_runner.py::extract_spatial_features` 내부에서 `_VideoArgs(video_fps=1.0, frames_upbound=32, force_sample=True)` 세 필드로 `process_video_with_decord` 를 호출한다. V2 는 이 3필드만으로 함수가 동작하는지를 확인.

---

# 다음 단계 (V2 부터 이어 진행)

세션 재진입 시 다음 순서로 진행. 환경 변수는 모두 `.env` 에서 자동 로드되므로 별도 export 불필요.

## V2. `process_video_with_decord` 호환성

**준비**
샘플 영상이 없으면 ffmpeg 으로 생성:
```bash
apt-get install -y ffmpeg
ffmpeg -f lavfi -i testsrc=duration=10:size=640x480:rate=30 -pix_fmt yuv420p /tmp/sample.mp4
```

**검증 명령**
```bash
cd /root/AI_repo && source .venv/bin/activate && python -c "
from app.vlm.llava.utils import process_video_with_decord
from app.vlm.cut3r_runner import _VideoArgs
frames, vt, ft, n = process_video_with_decord('/tmp/sample.mp4', _VideoArgs(video_fps=1.0, frames_upbound=32, force_sample=True))
print('num_frames:', len(frames) if hasattr(frames, '__len__') else 'n/a')
print('first frame shape/type:', frames[0].shape if hasattr(frames[0], 'shape') else type(frames[0]))
print('video_time:', vt, 'frame_time:', ft, 'n:', n)
"
```

**실패 시**
`AttributeError` 가 뜨면 `process_video_with_decord` 가 참조하는 필드명을 추출하여 `cut3r_runner.py::_VideoArgs` 에 추가. 본 PC 자체 문제 아닌 프로젝트 문제로 분류해 install-issues.md A 섹션에 추가.

## V3. `Cut3rEncoder` forward 시그니처

```bash
source .venv/bin/activate && python -c "
import torch
from app.vlm.loader import load_cut3r
h = load_cut3r()
x = torch.randn(32, 3, 432, 432, device=h['device'], dtype=h['dtype'])
out = h['model'](x, point_cloud_output_paths=None)
print('output type:', type(out).__name__)
if isinstance(out, tuple):
    print('camera_tokens:', out[0].shape, '/ patch_tokens:', out[1].shape)
"
```

**기대**: `camera_tokens: torch.Size([32, 1, 768]) / patch_tokens: torch.Size([32, 729, 768])`

**실패 시**: 입력 차원 `(F, C, H, W)` 대신 `(B, F, C, H, W)` 요구 또는 `point_cloud_output_paths` 인자명 차이 가능 → `cut3r_runner.py::extract_spatial_features` 호출부 수정.

## V4. `/preprocess` e2e

전제: RabbitMQ 로컬 기동, S3 자격증명. (`docker run -d --name rmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management`)

```bash
# 서버 기동
cd /root/AI_repo && source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# 별도 셸에서
curl http://localhost:8000/health
# → ai_model_loaded=true, cut3r_loaded=true

curl -X POST http://localhost:8000/api/v1/preprocess \
  -H "X-Internal-Token: dev-internal-token-change-me" \
  -H "Content-Type: application/json" \
  -d '{"job_id":"v4-test","user_id":"u_demo","video_url":"<URL>"}'

curl http://localhost:8000/api/v1/jobs/v4-test
```

S3 가 아직 안 잡힌 상태면 `.env` 자격증명 추가 또는 stub S3 우회. RabbitMQ 가 안 잡히면 발행이 실패하여 `analysis.failed` 로 빠짐.

## V5. `/chat` e2e + 캐시 적중

V4 의 `job_id` 가 `completed` 상태가 된 후:
```bash
# 1회차 (캐시 미스)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "X-Internal-Token: dev-internal-token-change-me" \
  -H "Content-Type: application/json" \
  -d '{"job_id":"v4-test","user_id":"u_demo","question":"방 크기는?"}'

# 2회차 (캐시 적중)
# → metadata.spatial_cache_hit == true 확인
```

**기대**: `metadata.stub == false`, `metadata.inference_time_ms > 0`, 2회차 응답이 1회차보다 빠름.

---

# 정식 패치 권장 (요약)

V2~V5 통과 후 다음 변경을 정식 커밋으로 묶을 것을 권장한다 (A 항목 전체 + curope 빌드 환경 문서화):

1. **`requirements-ml.txt` 수정** (A1, A2, A3, A5)
   ```diff
   - transformers==4.40.0.dev0
   + transformers==4.40.0
   - tokenizers==0.15.2
   + tokenizers>=0.19,<0.20
   - open-clip-torch
   + open-clip-torch==2.24.0
   - timm
   + timm==0.9.16
   - bitsandbytes>=0.41.3
   + bitsandbytes>=0.41.3,<0.44
   + matplotlib
   + av
   + omegaconf
   ```

2. **세 파일 import 경로 패치** (A6)
   - `app/vlm/llava/model/multimodal_spatial_encoder/cut3r_spatial_encoder.py`
   - `app/vlm/llava/model/multimodal_spatial_encoder/cut3r_points.py`
   - `app/vlm/scripts/extract_spatial_features.py`

3. **README §2 / setup.md 업데이트**
   - Ubuntu 24.04 환경에서 추가로 필요한 단계 명시 (CUDA 12.1 toolkit apt 설치, gcc-11 사용)
   - curope 빌드 명령에 `CC=gcc-11 CXX=g++-11 CUDA_HOME=/usr/local/cuda-12.1` 포함
