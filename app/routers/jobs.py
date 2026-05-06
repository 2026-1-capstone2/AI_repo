"""
작업 상태 조회 라우터
"""
from fastapi import APIRouter, HTTPException, status

from app.services.job_store import job_store

router = APIRouter()


@router.get(
    "/jobs/{job_id}",
    summary="작업 상태 조회",
    description="특정 작업의 현재 진행 상태를 조회",
)
async def get_job_status(job_id: str):
    """작업 상태 조회"""
    job = job_store.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "success": False,
                "error": {
                    "code": "JOB_NOT_FOUND",
                    "message": "해당 job_id를 찾을 수 없습니다",
                },
            },
        )

    return {
        "success": True,
        "data": job,
    }