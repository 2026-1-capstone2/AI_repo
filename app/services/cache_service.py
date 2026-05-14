"""
GPU LRU 캐시 — spatial_features / video_tensor
- spatial_features: S3 다운로드 후 GPU에 적재. 최근 N개 영상 유지.
- video_tensor: 영상 프레임을 image_processor에 통과시킨 텐서. 최근 N개 유지.

lifespan에서 init_video_cache(image_processor) 호출 필수.
"""
from functools import lru_cache
from typing import Optional

from app.core.config import settings

_video_processor = None
_job_to_spatial_key: dict[str, str] = {}
_job_to_video_path: dict[str, str] = {}


def init_video_cache(image_processor) -> None:
    """lifespan에서 1회 호출. image_processor를 모듈 클로저에 보관."""
    global _video_processor
    _video_processor = image_processor


def register_spatial(job_id: str, s3_key: str) -> None:
    """전처리 완료 시 job_id ↔ S3 key 매핑 등록."""
    _job_to_spatial_key[job_id] = s3_key


def register_video(job_id: str, local_video_path: str) -> None:
    """다운로드된 영상 로컬 경로 등록 (chat에서 텐서 만들 때 재사용)."""
    _job_to_video_path[job_id] = local_video_path


def get_spatial_s3_key(job_id: str) -> Optional[str]:
    return _job_to_spatial_key.get(job_id)


def get_video_path(job_id: str) -> Optional[str]:
    return _job_to_video_path.get(job_id)


@lru_cache(maxsize=settings.spatial_cache_size)
def get_spatial_features(job_id: str):
    """S3에서 spatial_features.pt 다운로드 → GPU 적재. LRU 자동 관리.

    캐시 키는 job_id. 호출 전 register_spatial(job_id, s3_key) 가 되어 있어야 함.
    """
    import torch
    from app.services.s3_service import download_spatial_features

    s3_key = _job_to_spatial_key.get(job_id)
    if not s3_key:
        raise KeyError(f"spatial_features not registered for job_id={job_id}")

    local_path = download_spatial_features(s3_key)
    feats = torch.load(local_path, map_location="cpu")
    return [{
        "camera_tokens": feats["camera_tokens"].to("cuda", dtype=torch.float16),
        "patch_tokens":  feats["patch_tokens"].to("cuda", dtype=torch.float16),
    }]


@lru_cache(maxsize=settings.video_cache_size)
def get_video_tensor_cached(job_id: str):
    """등록된 로컬 영상 경로 → 프레임 샘플링 + image_processor → GPU 텐서."""
    import torch
    from app.vlm.preprocessing import sample_video_frames

    assert _video_processor is not None, "init_video_cache() must be called in lifespan"
    local_path = _job_to_video_path.get(job_id)
    if not local_path:
        raise KeyError(f"video path not registered for job_id={job_id}")

    frames = sample_video_frames(local_path, num_frames=32, size=432)
    pixel_values = _video_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    return pixel_values.to("cuda", dtype=torch.float16)


def spatial_cache_info():
    return get_spatial_features.cache_info()
