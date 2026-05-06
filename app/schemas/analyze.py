"""
영상 분석 관련 스키마
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# 요청 모델
class VideoMetadata(BaseModel):
    """영상 메타데이터"""
    video_duration_sec: Optional[int] = None
    video_resolution: Optional[str] = None
    uploaded_at: Optional[datetime] = None


class AnalyzeRequest(BaseModel):
    """영상 분석 요청"""
    job_id: str = Field(..., description="작업 고유 ID (UUID)")
    user_id: str = Field(..., description="사용자 ID")
    video_url: str = Field(..., description="S3 Pre-signed URL")
    question: Optional[str] = Field(None, description="사용자 질문 (선택)")
    metadata: Optional[VideoMetadata] = None


# 응답 모델
class AnalyzeResponse(BaseModel):
    """영상 분석 응답"""
    job_id: str
    status: str = "accepted"
    estimated_time_sec: int = 45
    received_at: datetime


class JobStatusResponse(BaseModel):
    """작업 상태 응답"""
    job_id: str
    status: str
    progress: int = 0
    current_step: str
    started_at: datetime
    updated_at: datetime