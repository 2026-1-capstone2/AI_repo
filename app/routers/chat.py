"""
챗봇 추론 라우터
- POST /api/v1/chat  → 전처리 완료된 영상에 대해 자연어 질문 응답 (동기)
"""
from fastapi import APIRouter, Depends, Request

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import handle_chat

router = APIRouter()


def _get_vlm(request: Request):
    return getattr(request.app.state, "vlm", None)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="VLM-3R 챗봇 응답",
)
async def chat(
    req: ChatRequest,
    vlm = Depends(_get_vlm),
):
    return await handle_chat(req, vlm)
