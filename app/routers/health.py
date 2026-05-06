"""
헬스체크 라우터
"""
from datetime import datetime
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "version": "1.0.0",
            "checked_at": datetime.utcnow().isoformat() + "Z",
            "dependencies": {
                "ai_model_loaded": False,
                "rabbitmq_connected": False,
                "s3_accessible": False,
            },
        },
    }