"""
환경 설정 관리
.env 파일에서 환경변수 로드
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 서버 설정
    app_name: str = "Spatial Analysis AI Server"
    debug: bool = True

    # API 설정
    api_key: str = "dev-api-key-change-me"

    # AI 모델 설정 (추후 사용)
    model_path: str = "./models"

    # RabbitMQ 설정 ⭐
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_exchange: str = "analysis_exchange"
    rabbitmq_queue: str = "analysis_results"
    rabbitmq_routing_key_completed: str = "analysis.completed"
    rabbitmq_routing_key_failed: str = "analysis.failed"

    # S3 설정 (추후 사용)
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "ap-northeast-2"

    class Config:
        env_file = ".env"

settings = Settings()