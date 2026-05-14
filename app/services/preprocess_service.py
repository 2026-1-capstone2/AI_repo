"""
전처리 비즈니스 로직
영상 다운로드 → CUT3R 추출 → S3 업로드 → RabbitMQ 발행
"""
import logging
import time
from datetime import datetime

from app.core.config import settings
from app.services import cache_service
from app.services.job_store import job_store
from app.services.rabbitmq_publisher import rabbitmq_publisher
from app.services.video_downloader import (
    VideoDownloadError,
    delete_video,
    download_video,
)
from app.services.s3_service import build_spatial_s3_key, upload_spatial_features
from app.vlm import cut3r_runner, loader as vlm_loader

logger = logging.getLogger(__name__)


async def run_preprocess_job(
    job_id: str,
    user_id: str,
    video_url: str,
    cut3r_handle: dict | None,
):
    """BackgroundTask 진입점.

    Args:
        cut3r_handle: lifespan에서 로드된 CUT3R 핸들. stub 모드면 None일 수 있음.
    """
    start = time.time()
    video_path: str | None = None
    try:
        # 1) 영상 다운로드
        job_store.update_job(
            job_id=job_id,
            status="downloading",
            current_step="download_video",
            progress=10,
        )
        video_path = await download_video(video_url, job_id)
        cache_service.register_video(job_id, video_path)

        # 2) CUT3R 추출
        job_store.update_job(
            job_id=job_id,
            status="processing",
            current_step="cut3r_extract",
            progress=50,
            video_path=video_path,
        )
        if settings.stub_models or cut3r_handle is None:
            features = cut3r_runner.extract_spatial_features_stub(video_path)
        else:
            features = cut3r_runner.extract_spatial_features(cut3r_handle, video_path)

        # 3) S3 업로드
        job_store.update_job(
            job_id=job_id,
            status="processing",
            current_step="upload_features",
            progress=85,
        )
        s3_key = build_spatial_s3_key(user_id=user_id, job_id=job_id)
        if not settings.stub_models:
            upload_spatial_features(s3_key, features)
        else:
            logger.info(f"[stub] skip S3 upload, would write {s3_key}")
        cache_service.register_spatial(job_id, s3_key)

        # 4) 완료 — RabbitMQ 발행
        duration_ms = int((time.time() - start) * 1000)
        job_store.update_job(
            job_id=job_id,
            status="completed",
            current_step="publish_result",
            progress=100,
            completed_at=datetime.utcnow(),
            spatial_features_s3_key=s3_key,
        )
        rabbitmq_publisher.publish_preprocess_completed(
            job_id=job_id,
            user_id=user_id,
            spatial_features_s3_key=s3_key,
            duration_ms=duration_ms,
        )

    except VideoDownloadError as e:
        _fail(job_id, user_id, e.code, e.message)
    except Exception as e:
        logger.exception("preprocess failed")
        _fail(job_id, user_id, "INTERNAL_SERVER_ERROR", str(e))
    finally:
        # 영상 파일은 chat에서 video_tensor 만들 때까지 필요 → 정리 시점은 보류.
        # 후처리(예: TTL 또는 명시적 /jobs/{id}/cleanup)로 옮긴다.
        if settings.unload_cut3r_after_preprocess and cut3r_handle:
            vlm_loader.unload_cut3r(cut3r_handle)


def _fail(job_id: str, user_id: str, code: str, message: str):
    job_store.update_job(
        job_id=job_id,
        status="failed",
        error_code=code,
        error_message=message,
        failed_at=datetime.utcnow(),
    )
    rabbitmq_publisher.publish_preprocess_failed(
        job_id=job_id,
        user_id=user_id,
        error_code=code,
        error_message=message,
    )
