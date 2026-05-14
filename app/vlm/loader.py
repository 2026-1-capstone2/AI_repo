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
        overwrite_config["spatial_tower"] = None

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
    """CUT3R 모델만 별도 핸들로 로드. 전처리 경로 전용."""
    if settings.stub_models:
        logger.warning("[loader] stub_models=True → CUT3R 미로드")
        return None

    import torch
    # TODO: 실제 CUT3R weight 로딩 함수는 scripts/extract_spatial_features.py 참고
    # from app.vlm.CUT3R.src.dust3r.inference import load_model as load_cut3r_weights
    raise NotImplementedError(
        "CUT3R loader 미구현. extract_spatial_features.py를 참조해 로딩 코드를 채울 것."
    )


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
