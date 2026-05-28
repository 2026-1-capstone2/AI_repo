"""
CUT3R spatial feature 추출

본 레포 scripts/extract_spatial_features.py 의 흐름을 단일 영상 추론용으로 축약:
  - 영상 → 프레임 샘플링 (process_video_with_decord)
  - SigLip image processor 통과
  - (432, 432) 리사이즈
  - Cut3rEncoder forward 호출
  - camera_tokens / patch_tokens 반환

extract_spatial_features:      실제 CUT3R 모델 호출
extract_spatial_features_stub: 개발용 더미 텐서
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# 본 레포의 train.train.DataArguments 대체 (해당 폴더는 이식 안 함)
# process_video_with_decord 가 참조하는 3개 필드만 정의한다.
# ──────────────────────────────────────────────────────────────────
@dataclass
class _VideoArgs:
    video_fps: float = 1.0
    frames_upbound: int = 32
    force_sample: bool = True


# image processor 캐시 (모듈 단위, 1회 생성)
_image_processor = None


def _get_image_processor():
    """CUT3R 입력용 SigLip image processor (LLaVA-NeXT-Video 표준값과 동일)."""
    global _image_processor
    if _image_processor is None:
        from app.vlm.llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor  # type: ignore
        _image_processor = SigLipImageProcessor(
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            size=(384, 384),
            resample=3,
            rescale_factor=1 / 255.0,
        )
    return _image_processor


def extract_spatial_features(cut3r_handle: dict, video_path: str) -> dict:
    """CUT3R 실행 → {"camera_tokens", "patch_tokens"} 반환.

    출력 포맷 (fp16, CPU 텐서):
      - camera_tokens: [F, 1, 768]
      - patch_tokens:  [F, 729, 768]
      F = 32 (frames_upbound)

    본 레포 extract_spatial_features.py 의 단일 영상 처리 시 동작과 동등.
    배치 처리는 하지 않는다 (운영 환경에서는 영상당 1회 호출).
    """
    import torch
    from app.vlm.llava.utils import process_video_with_decord  # type: ignore

    model = cut3r_handle["model"]
    device = cut3r_handle["device"]
    dtype = cut3r_handle["dtype"]
    processor = _get_image_processor()

    # 1) video → frame list (PIL/numpy) — 본 레포의 학습 시와 동일 함수 사용
    data_args = _VideoArgs(video_fps=1.0, frames_upbound=32, force_sample=True)
    video_frames, _video_time, _frame_time, _num_frames = process_video_with_decord(video_path, data_args)

    # 2) SigLip image processor 통과 → (F, C, H_proc, W_proc)
    processed = processor.preprocess(images=video_frames, return_tensors="pt")
    frames_tensor = processed["pixel_values"]

    # 3) CUT3R 요구 사이즈(432×432) 로 리사이즈
    target_size = (432, 432)
    if frames_tensor.shape[-2] != target_size[0] or frames_tensor.shape[-1] != target_size[1]:
        frames_tensor = torch.nn.functional.interpolate(
            frames_tensor,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    # 4) device/dtype 정합, forward 호출
    #    Cut3rEncoder 는 (F, C, H, W) 또는 (B, F, C, H, W) 입력 모두 지원.
    #    단일 영상이라 (F, C, H, W) 그대로 전달.
    frames_tensor = frames_tensor.to(device=device, dtype=dtype)

    with torch.inference_mode():
        camera_tokens, patch_tokens = model(
            frames_tensor,
            point_cloud_output_paths=None,
        )

    # 5) CPU + fp16 으로 직렬화 가능한 상태로 변환
    return {
        "camera_tokens": camera_tokens.detach().to("cpu", dtype=torch.float16),
        "patch_tokens":  patch_tokens.detach().to("cpu", dtype=torch.float16),
    }


def extract_spatial_features_stub(video_path: str) -> dict:
    """개발/CI용 더미 텐서. 모양만 맞춰 S3 업로드까지의 흐름 검증 가능."""
    import torch
    F = 32
    return {
        "camera_tokens": torch.zeros((F, 1, 768), dtype=torch.float16),
        "patch_tokens":  torch.zeros((F, 729, 768), dtype=torch.float16),
    }
