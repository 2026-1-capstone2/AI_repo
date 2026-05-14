"""
챗봇 추론 비즈니스 로직
spatial_features 캐시 + video_tensor 캐시 → VLM 추론
"""
import asyncio
import logging
import time

from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.chat import ChatMetadata, ChatRequest, ChatResponse
from app.services import cache_service
from app.services.job_store import job_store
from app.vlm import inference as vlm_inference

logger = logging.getLogger(__name__)


async def handle_chat(
    req: ChatRequest,
    vlm_handle: dict | None,
) -> ChatResponse:
    """질문 처리 진입점.

    - vlm_handle이 None이거나 settings.stub_models이면 mock 응답.
    - 실모델이면 spatial_features + video_tensor 캐시 → run_inference.
    """
    job = job_store.get_job(req.job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "JOB_NOT_FOUND", "message": "해당 job_id를 찾을 수 없습니다"},
        )
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail={
                "code": "PREPROCESS_NOT_READY",
                "message": f"전처리 미완료 (현재 status={job.get('status')})",
            },
        )

    if settings.stub_models or vlm_handle is None:
        return _stub_response(req)

    return await _real_response(req, vlm_handle)


def _stub_response(req: ChatRequest) -> ChatResponse:
    """실모델 없이도 흐름 검증 가능하도록 더미 응답."""
    answer = (
        f"[stub] '{req.question}'에 대한 답변입니다. "
        "현재는 실모델이 로드되지 않은 stub 모드입니다."
    )
    return ChatResponse(
        job_id=req.job_id,
        answer=answer,
        metadata=ChatMetadata(
            tokens_used=len(answer),
            inference_time_ms=0,
            spatial_cache_hit=False,
            stub=True,
        ),
    )


async def _real_response(req: ChatRequest, vlm_handle: dict) -> ChatResponse:
    start = time.time()
    hit_before = cache_service.spatial_cache_info().hits

    result = await asyncio.to_thread(
        vlm_inference.run_inference,
        vlm_handle["model"],
        vlm_handle["tokenizer"],
        req,
    )

    cache_hit = cache_service.spatial_cache_info().hits > hit_before
    return ChatResponse(
        job_id=req.job_id,
        answer=result["answer"],
        metadata=ChatMetadata(
            tokens_used=result["tokens_used"],
            inference_time_ms=int((time.time() - start) * 1000),
            spatial_cache_hit=cache_hit,
            stub=False,
        ),
    )
