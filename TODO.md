# AI Server TODO

> 우선순위·의존관계 기준 작업 리스트
> 골격 작업 완료 후 다음 단계 추적용

---

## A. 지금 바로 가능 (stub 모드, GPU 없이)

| # | 작업 | 위치 | 비고 |
|---|------|------|------|
| A1 | `.env.example` 만들기 | `.env.example` | RabbitMQ/S3/토큰 키 노출 없는 템플릿 |
| A2 | `internal_api_token` **실제 검증** 적용 | `routers/preprocess.py`, `routers/chat.py` | 현재 설정만 있고 미들웨어 없음 → BE가 보낼 헤더 키 합의 필요 |
| A3 | `video_downloader`가 `settings.download_dir` 사용 | `services/video_downloader.py` | 현재 하드코딩 `./temp/videos` |
| A4 | `/health`에 **실 상태** 반영 | `routers/health.py` | `app.state.vlm/cut3r`, RabbitMQ 연결 상태 |
| A5 | `/ready` 추가 | `routers/health.py` | 모델 로드 완료 + 큐 연결 확인 (k8s readiness probe용) |
| A6 | **stub 모드 e2e 테스트** | `tests/` 신설 | `/preprocess` → MQ 발행 확인 → `/chat` stub 응답 |
| A7 | **RabbitMQ 메시지 계약 문서** | `docs/` 또는 README | BE 팀과 routing key + payload 합의 |

---

## B. BE(Spring)와 실통합 (stub 모드로도 가능)

| # | 작업 | 비고 |
|---|------|------|
| B1 | RabbitMQ 로컬 띄우기 | `docker run rabbitmq:3-management` |
| B2 | BE 쪽 큐 구독 검증 | Spring이 `analysis.completed` 받아 처리하는지 |
| B3 | Pre-signed URL 다운로드 실테스트 | BE가 실제로 발급하는 URL 포맷 확인 |
| B4 | 에러 코드 합의 | `INVALID_VIDEO_URL`, `VIDEO_TOO_LARGE` 등 BE가 사용자에게 매핑할 코드 |
| B5 | `job_id` 중복/멱등 처리 정책 | 현재 409 반환 — BE 재시도 정책과 정합 확인 |

---

## C. VLM-3R 모델 실통합 (GPU 필요)

| # | 작업 | 의존 | 비고 |
|---|------|------|------|
| C1 | `video_demo.py` 분리 동작 먼저 CLI 검증 | 상위 폴더 VLM-3R 레포에서 | `--spatial-features-path` 옵션 추가 (seperation.md §5.2) |
| C2 | `app/vlm/llava`, `app/vlm/CUT3R` 이식 | C1 후 | 원본 → ai_repo로 코드 복사 + import 경로 일괄 치환 |
| C3 | `app/vlm/loader.py::load_vlm` 실구현 | C2 | `spatial_tower=None` 비활성화 검증 |
| C4 | `app/vlm/loader.py::load_cut3r` 실구현 | C2 | `scripts/extract_spatial_features.py` 참조 |
| C5 | `app/vlm/cut3r_runner.py::extract_spatial_features` 실구현 | C4 | forward 시그니처 매핑 |
| C6 | `stub_models=False` 전환 + 단일 영상 e2e | C3~C5 | VRAM 모니터링 (두 모델 합산) |
| C7 | 캐시 적중 검증 (`spatial_cache_hit=true`) | C6 | 같은 영상에 연속 질문 |

---

## D. 안정화

| # | 작업 |
|---|------|
| D1 | Dockerfile + docker-compose (GPU 할당) |
| D2 | 로깅 표준화 (job_id 컨텍스트, 구조화 로그) |
| D3 | 메트릭 (Prometheus): 추론 latency p50/p95, 캐시 hit rate, VRAM |
| D4 | 영상 파일 정리 정책 (TTL 또는 `/jobs/{id}/cleanup`) — 현재 preprocess 후 안 지우고 chat까지 보관 |
| D5 | OOM 가드 + 재시도 정책 |
| D6 | RabbitMQ publish 실패 시 outbox 패턴 (job_store에 pending 마킹 후 재시도) |

---

## E. 배포

| # | 작업 |
|---|------|
| E1 | GPU 인스턴스 선정 (A10G 24GB 또는 자체 8GB) |
| E2 | S3 IAM Role 구성 |
| E3 | ngrok → 실 도메인 + TLS |
| E4 | k8s 또는 ECS 매니페스트 |

---

## 추천 진행 순서

```
[이번 주]   A1~A5   →  A6 (stub e2e 테스트)
[다음 주]   B1~B5   →  BE 팀과 통합 검증 (모델 없이도 전 흐름 검증 끝)
[그 다음]   C1~C7   →  실모델 통합 (GPU 환경에서)
[안정화]    D1~D6
[배포]      E1~E4
```

**가장 큰 레버리지**: A2(인증) + A6(테스트) + B2(BE 큐 검증) 묶음을 먼저 끝내면,
**모델 통합과 BE 통합을 병렬화** 가능. 모델 작업이 길어져도 BE 팀은 stub 응답으로 자기 작업 계속 가능.
