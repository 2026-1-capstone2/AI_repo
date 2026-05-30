"""
VLM-3R 모델 로더 — VLM과 CUT3R를 분리 로드

stub_models=True 인 동안은 실모델을 적재하지 않고 None 핸들을 반환한다.
실모델 통합은 다음 단계:
  1) VLM-3R 원본 레포의 llava/, CUT3R/ 를 app/vlm/llava, app/vlm/CUT3R 로 이식
  2) settings.stub_models=False 로 전환
"""
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


def load_vlm() -> dict | None:
    """VLM (SigLIP + Qwen2 + LoRA) 만 로드. CUT3R(spatial_tower) 비활성화.

    Returns:
        {"tokenizer", "model", "image_processor"} 또는 stub 모드면 None
    """
    if settings.stub_models:
        logger.warning("[loader] stub_models=True → VLM 미로드")
        return None

    import torch
    from app.vlm.llava.model.builder import load_pretrained_model        # type: ignore
    from app.vlm.llava.mm_utils import get_model_name_from_path           # type: ignore

    model_name = get_model_name_from_path(settings.model_path)
    overwrite_config = {
        "spatial_tower_select_feature": "all",
        "spatial_feature_dim": 768,
        "fusion_block": "cross_attention",
    }
    if settings.disable_spatial_tower_in_vlm:
        # CUT3R는 load_cut3r()로 별도 적재하고, 추론 시 precompute된 spatial_features를
        # 주입한다. 따라서 VLM 안에 CUT3R(spatial_tower) 가중치를 중복 적재할 필요가 없다.
        # 단, spatial_tower 설정 자체("cut3r")는 그대로 둬야 추론 분기(encode_images)가
        # precomputed features 경로를 탄다 (spatial_tower=None 으로 끄면 그 경로가 막힌다).
        # 아래 플래그를 보고 llava_qwen.from_pretrained 가 "무거운 가중치 로드만" 건너뛴다.
        overwrite_config["disable_spatial_tower_weights"] = True

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=settings.model_path,
        model_base=settings.model_base,
        model_name=model_name,
        load_4bit=settings.load_4bit,
        overwrite_config=overwrite_config,
    )
    model.eval()
    logger.info(f"[loader] VLM loaded. VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")
    return {"tokenizer": tokenizer, "model": model, "image_processor": image_processor}


def load_cut3r() -> dict | None:
    """CUT3R 모델만 별도 핸들로 로드. 전처리 경로 전용.

    본 레포 scripts/extract_spatial_features.py 의 process_videos_on_gpu()
    초기화 부분(line 196~212)을 그대로 옮긴 구현. Cut3rSpatialConfig 로
    가중치 경로 등을 설정하고 Cut3rEncoder 로 모델을 구성한 뒤 GPU 에 적재한다.

    Returns:
        {"model": Cut3rEncoder, "device": "cuda", "dtype": torch.float16} 또는 stub
    """
    if settings.stub_models:
        logger.warning("[loader] stub_models=True → CUT3R 미로드")
        return None

    import torch
    from app.vlm.llava.model.multimodal_spatial_encoder.cut3r_spatial_encoder import (
        Cut3rSpatialConfig,
        Cut3rEncoder,
    )  # type: ignore

    cut3r_config = Cut3rSpatialConfig(
        weights_path=settings.cut3r_weights,
        # 추론 경로에서는 point cloud 출력 불필요
        export_point_cloud=False,
        point_cloud_output_dir=None,
        point_cloud_voxel_size=None,
    )
    spatial_tower = Cut3rEncoder(config=cut3r_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    spatial_tower.to(device=device, dtype=dtype).eval()

    logger.info(f"[loader] CUT3R loaded on {device}. VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")
    return {"model": spatial_tower, "device": device, "dtype": dtype}


def unload_cut3r(cut3r_handle: dict | None) -> None:
    """CUT3R를 GPU에서 해제 (저VRAM 환경에서 전처리 후 호출)."""
    if not cut3r_handle:
        return
    if "model" in cut3r_handle:
        del cut3r_handle["model"]
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
