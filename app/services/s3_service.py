"""
S3 입출력 서비스 (spatial_features 저장/조회)
영상은 BE가 제공한 Pre-signed URL로 다운로드하므로 여기서는 features만 다룬다.
"""
import io
import os
import tempfile
from typing import Optional

from app.core.config import settings

# boto3는 GPU 환경 외에서도 import 시 무거우므로 lazy 사용
_s3_client = None


def _client():
    global _s3_client
    if _s3_client is None:
        import boto3
        kwargs = {"region_name": settings.aws_region}
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        _s3_client = boto3.client("s3", **kwargs)
    return _s3_client


def build_spatial_s3_key(user_id: str, job_id: str) -> str:
    return f"{settings.s3_spatial_prefix}/{user_id}/{job_id}.pt"


def upload_spatial_features(s3_key: str, features: dict) -> None:
    """features dict → in-memory .pt → S3 PUT."""
    import torch  # local import: torch가 없을 수도 있는 개발 환경 대비
    buf = io.BytesIO()
    torch.save(features, buf)
    buf.seek(0)
    _client().put_object(Bucket=settings.s3_bucket, Key=s3_key, Body=buf.getvalue())


def download_spatial_features(s3_key: str) -> str:
    """spatial_features.pt를 로컬 임시 파일로 다운로드. 캐시 적중 시 재사용."""
    local_name = s3_key.replace("/", "_")
    local_path = os.path.join(tempfile.gettempdir(), f"vlm3r_{local_name}")
    if os.path.exists(local_path):
        return local_path
    _client().download_file(settings.s3_bucket, s3_key, local_path)
    return local_path


def head_object(s3_key: str) -> Optional[dict]:
    try:
        return _client().head_object(Bucket=settings.s3_bucket, Key=s3_key)
    except Exception:
        return None
