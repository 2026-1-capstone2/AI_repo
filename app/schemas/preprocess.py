"""
전처리(CUT3R spatial feature 추출) 요청/응답 스키마
"""
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    video_duration_sec: Optional[int] = None
    video_resolution: Optional[str] = None
    uploaded_at: Optional[datetime] = None


class PreprocessRequest(BaseModel):
    """BE → AI: 영상 전처리 요청"""
    job_id: str = Field(..., description="작업 고유 ID (UUID)")
    user_id: str = Field(..., description="사용자 ID")
    video_url: str = Field(..., description="S3 Pre-signed GET URL")
    metadata: Optional[VideoMetadata] = None


class PreprocessAcceptedResponse(BaseModel):
    """AI → BE: 즉시 반환 (202 Accepted)"""
    job_id: str
    status: Literal["accepted"] = "accepted"
    estimated_time_sec: int = 45
    received_at: datetime


# RabbitMQ payload (AI → BE via MQ) ─────────────────────────────────
class PreprocessCompletedMessage(BaseModel):
    """analysis.completed 라우팅 키로 발행되는 메시지 본문"""
    job_id: str
    user_id: str
    status: Literal["completed"] = "completed"
    event_type: Literal["preprocess_completed"] = "preprocess_completed"
    spatial_features_s3_key: str
    duration_ms: int
    completed_at: datetime


class ErrorBody(BaseModel):
    code: str
    message: str


class PreprocessFailedMessage(BaseModel):
    """analysis.failed 라우팅 키로 발행되는 메시지 본문.

    실제 발행은 services/rabbitmq_publisher.py 의 publish_preprocess_failed 가
    dict를 직접 만들어 수행한다. 본 스키마는 BE와 공유하는 명세 역할이며,
    README §4.2 의 페이로드와 동일한 중첩 구조(`error: {code, message}`)를 정의한다.
    """
    job_id: str
    user_id: str
    status: Literal["failed"] = "failed"
    event_type: Literal["preprocess_failed"] = "preprocess_failed"
    error: ErrorBody
    failed_at: datetime
