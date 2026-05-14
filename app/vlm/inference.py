"""
VLM-3R 추론 — model.generate() 호출

run_inference 는 HTTP 무관 (테스트/CLI에서도 직접 호출 가능).
"""
import logging

from app.services import cache_service
from app.vlm.preprocessing import build_prompt

logger = logging.getLogger(__name__)


def run_inference(model, tokenizer, req) -> dict:
    """req: ChatRequest. 반환: {"answer": str, "tokens_used": int}."""
    import torch
    from app.vlm.llava.constants import IMAGE_TOKEN_INDEX           # type: ignore
    from app.vlm.llava.mm_utils import tokenizer_image_token        # type: ignore

    spatial_features = cache_service.get_spatial_features(req.job_id)
    video_tensor = cache_service.get_video_tensor_cached(req.job_id)

    prompt = build_prompt(req.history, req.question)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 151643
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=video_tensor,
            attention_mask=attention_mask,
            spatial_features=spatial_features,
            modalities="video",
            do_sample=req.temperature > 0,
            temperature=req.temperature,
            max_new_tokens=req.max_new_tokens,
            use_cache=True,
        )

    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return {"answer": answer, "tokens_used": int(output_ids.shape[1])}
