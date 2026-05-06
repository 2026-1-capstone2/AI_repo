"""
영상 분석 라우터
"""
import asyncio
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.schemas.analyze import AnalyzeRequest
from app.services.job_store import job_store
from app.services.rabbitmq_publisher import rabbitmq_publisher
from app.services.video_downloader import (
    VideoDownloadError,
    delete_video,
    download_video,
)

router = APIRouter()


@router.post(
    "/analyze",
    status_code=status.HTTP_202_ACCEPTED,
    summary="영상 분석 요청",
    description="Spring 서버에서 영상 분석을 요청하는 엔드포인트",
)
async def analyze_video(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
):
    """
    영상 분석 작업을 받아서 백그라운드로 처리 시작
    """
    # 중복 작업 체크
    if job_store.exists(request.job_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "success": False,
                "error": {
                    "code": "DUPLICATE_JOB_ID",
                    "message": "이미 처리 중인 job_id입니다",
                    "details": f"job_id: {request.job_id}",
                },
            },
        )

    # 작업 생성
    job_store.create_job(
        job_id=request.job_id,
        user_id=request.user_id,
    )

    # 백그라운드에서 영상 다운로드 + 분석 시작
    background_tasks.add_task(
        process_video_analysis,
        job_id=request.job_id,
        user_id=request.user_id,
        video_url=request.video_url,
        question=request.question,
    )

    return {
        "success": True,
        "data": {
            "job_id": request.job_id,
            "status": "accepted",
            "estimated_time_sec": 45,
            "received_at": datetime.utcnow().isoformat() + "Z",
        },
    }


async def process_video_analysis(
    job_id: str,
    user_id: str,
    video_url: str,
    question: str | None = None,
):
    """
    백그라운드에서 영상 분석 전체 프로세스 수행
    """
    video_path = None

    try:
        # 1단계: 영상 다운로드
        job_store.update_job(
            job_id=job_id,
            status="downloading",
            current_step="download_video",
            progress=10,
        )

        video_path = await download_video(video_url, job_id)

        job_store.update_job(
            job_id=job_id,
            status="processing",
            current_step="ai_inference",
            progress=30,
            video_path=video_path,
        )

        # 2단계: AI 분석 (TODO: 실제 모델 연동)
        await asyncio.sleep(2)

        # Mock 분석 결과
        mock_analysis = {
            "spatial_features": {
                "room_type": "거실",
                "estimated_area_sqm": 25.5,
                "objects_detected": ["소파", "TV", "테이블"],
            },
            "depth_estimation": "성공",
        }

        job_store.update_job(
            job_id=job_id,
            status="generating_response",
            current_step="langchain_response",
            progress=70,
        )

        # 3단계: 자연어 응답 생성 (TODO: LangChain 연동)
        await asyncio.sleep(1)

        mock_response = (
            f"분석된 공간은 약 {mock_analysis['spatial_features']['estimated_area_sqm']}㎡ 크기의 "
            f"{mock_analysis['spatial_features']['room_type']}으로 보입니다. "
            f"주요 가구로는 {', '.join(mock_analysis['spatial_features']['objects_detected'])}이 감지되었습니다."
        )

        # 4단계: RabbitMQ로 결과 발행 ⭐
        publish_success = rabbitmq_publisher.publish_completed(
            job_id=job_id,
            user_id=user_id,
            analysis_result=mock_analysis,
            natural_language_response=mock_response,
        )

        if not publish_success:
            raise Exception("RabbitMQ 메시지 발행 실패")

        job_store.update_job(
            job_id=job_id,
            status="completed",
            current_step="publish_result",
            progress=100,
            completed_at=datetime.utcnow(),
            analysis_result=mock_analysis,
            natural_language_response=mock_response,
        )

    except VideoDownloadError as e:
        job_store.update_job(
            job_id=job_id,
            status="failed",
            error_code=e.code,
            error_message=e.message,
            failed_at=datetime.utcnow(),
        )

        # RabbitMQ로 실패 메시지 발행
        rabbitmq_publisher.publish_failed(
            job_id=job_id,
            user_id=user_id,
            error_code=e.code,
            error_message=e.message,
        )

    except Exception as e:
        job_store.update_job(
            job_id=job_id,
            status="failed",
            error_code="INTERNAL_SERVER_ERROR",
            error_message=str(e),
            failed_at=datetime.utcnow(),
        )

        # RabbitMQ로 실패 메시지 발행
        rabbitmq_publisher.publish_failed(
            job_id=job_id,
            user_id=user_id,
            error_code="INTERNAL_SERVER_ERROR",
            error_message=str(e),
        )

    finally:
        # 다운로드된 영상 파일 정리
        if video_path:
            delete_video(video_path)