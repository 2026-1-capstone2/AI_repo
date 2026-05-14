"""
CUT3R spatial feature 추출

extract_spatial_features:    실제 CUT3R 모델 호출 (TODO 구현)
extract_spatial_features_stub: 개발용 더미 데이터 생성
"""
import logging

logger = logging.getLogger(__name__)


def extract_spatial_features(cut3r_handle: dict, video_path: str) -> dict:
    """CUT3R 실행 → {"camera_tokens", "patch_tokens"} 반환.

    TODO: scripts/extract_spatial_features.py 의 forward 호출 시그니처에 맞춰 구현.
    포맷 (fp16):
      - camera_tokens: [F, 1, 768]
      - patch_tokens:  [F, 729, 768]
    """
    raise NotImplementedError(
        "CUT3R 추출 미구현. scripts/extract_spatial_features.py 의 forward 흐름을 이식할 것."
    )


def extract_spatial_features_stub(video_path: str) -> dict:
    """개발/CI용 더미 텐서. 모양만 맞춰 S3 업로드까지의 흐름 검증 가능."""
    import torch
    F = 32
    return {
        "camera_tokens": torch.zeros((F, 1, 768), dtype=torch.float16),
        "patch_tokens":  torch.zeros((F, 729, 768), dtype=torch.float16),
    }
