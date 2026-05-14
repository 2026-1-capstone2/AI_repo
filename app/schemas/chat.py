"""
챗봇 추론(VLM-3R) 요청/응답 스키마
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """BE → AI: 사용자 질문"""
    job_id: str = Field(..., description="전처리 완료된 영상의 job_id")
    user_id: str
    question: str
    history: List[ChatMessage] = Field(default_factory=list)
    max_new_tokens: int = 512
    temperature: float = 0.0


class ChatMetadata(BaseModel):
    tokens_used: int
    inference_time_ms: int
    spatial_cache_hit: bool
    stub: bool = False


class ChatResponse(BaseModel):
    """AI → BE: 동기 응답"""
    job_id: str
    answer: str
    metadata: ChatMetadata
