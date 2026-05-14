"""
환경 설정 관리
.env 파일에서 환경변수 로드
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── 서버 ──
    app_name: str = "Spatial Analysis AI Server"
    debug: bool = True

    # ── 내부 API 인증 (Spring BE → AI) ──
    internal_api_token: str = "dev-internal-token-change-me"

    # ── VLM-3R 모델 ──
    model_path: str = "/data/vlm-3r-llava-qwen2-lora"
    model_base: str = "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"
    cut3r_weights: str = "/data/CUT3R/src/cut3r_512_dpt_4_64.pth"
    load_4bit: bool = True

    # VLM 로드 시 spatial_tower 비활성화 (CUT3R 중복 적재 방지)
    disable_spatial_tower_in_vlm: bool = True

    # 전처리 후 CUT3R를 GPU에서 해제할지 (저VRAM 환경에서 True 권장)
    unload_cut3r_after_preprocess: bool = False

    # 실모델 미적재 모드 (개발/CI에서 GPU 없이 실행할 때 True)
    stub_models: bool = True

    # ── GPU LRU 캐시 ──
    spatial_cache_size: int = 20
    video_cache_size: int = 10

    # ── 영상 다운로드 (Pre-signed URL → 로컬) ──
    download_dir: str = "./temp/videos"
    max_video_size_mb: int = 500
    download_timeout_sec: int = 300

    # ── S3 (spatial_features 저장용) ──
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "ap-northeast-2"
    s3_bucket: str = "vlm3r-bucket"
    s3_spatial_prefix: str = "spatial_features"

    # ── RabbitMQ ──
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_exchange: str = "analysis_exchange"
    rabbitmq_queue: str = "analysis_results"
    # BE와 계약된 라우팅 키 (변경 시 BE 큐 구독자도 함께 수정 필요)
    rabbitmq_routing_key_completed: str = "analysis.completed"
    rabbitmq_routing_key_failed: str = "analysis.failed"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


settings = Settings()
