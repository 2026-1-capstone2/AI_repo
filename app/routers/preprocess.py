"""
영상 전처리 라우터
- POST /api/v1/preprocess  → CUT3R spatial feature 추출 작업 등록
- 완료/실패는 RabbitMQ로 Spring BE에 통보
"""
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from app.schemas.preprocess import PreprocessAcceptedResponse, PreprocessRequest
from app.services.job_store import job_store
from app.services.preprocess_service import run_preprocess_job

router = APIRouter()


def _get_cut3r(request: Request):
    return getattr(request.app.state, "cut3r", None)


@router.post(
    "/preprocess",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=PreprocessAcceptedResponse,
    summary="영상 전처리 요청 (CUT3R)",
)
async def preprocess(
    req: PreprocessRequest,
    background_tasks: BackgroundTasks,
    cut3r = Depends(_get_cut3r),
):
    if job_store.exists(req.job_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "DUPLICATE_JOB_ID",
                "message": "이미 처리 중인 job_id입니다",
                "details": f"job_id: {req.job_id}",
            },
        )

    job_store.create_job(job_id=req.job_id, user_id=req.user_id)

    background_tasks.add_task(
        run_preprocess_job,
        job_id=req.job_id,
        user_id=req.user_id,
        video_url=req.video_url,
        cut3r_handle=cut3r,
    )

    return PreprocessAcceptedResponse(
        job_id=req.job_id,
        status="accepted",
        estimated_time_sec=45,
        received_at=datetime.utcnow(),
    )
