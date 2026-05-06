"""
공통 응답 스키마
"""
from typing import Any, Optional
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """에러 상세 정보"""
    code: str
    message: str
    details: Optional[Any] = None


class SuccessResponse(BaseModel):
    """성공 응답 형식"""
    success: bool = True
    data: Any


class ErrorResponse(BaseModel):
    """실패 응답 형식"""
    success: bool = False
    error: ErrorDetail