"""
영상 다운로드 서비스
Pre-signed URL을 통해 S3 영상을 로컬로 다운로드
"""
import os
import uuid
from pathlib import Path
from typing import Optional

import httpx


# 다운로드 임시 저장 폴더
DOWNLOAD_DIR = Path("./temp/videos")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 다운로드 제약
MAX_VIDEO_SIZE_MB = 500
DOWNLOAD_TIMEOUT_SEC = 300  # 5분
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}


class VideoDownloadError(Exception):
    """영상 다운로드 관련 에러"""

    def __init__(self, code: str, message: str, details: Optional[str] = None):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


async def download_video(video_url: str, job_id: str) -> str:
    """
    Pre-signed URL에서 영상을 다운로드해서 로컬에 저장

    Args:
        video_url: S3 Pre-signed URL
        job_id: 작업 ID (파일명에 사용)

    Returns:
        저장된 로컬 파일 경로

    Raises:
        VideoDownloadError: 다운로드 실패 시
    """
    # 파일 확장자 추출
    extension = _extract_extension(video_url)

    # 저장 경로 생성
    file_name = f"{job_id}{extension}"
    save_path = DOWNLOAD_DIR / file_name

    try:
        # httpx로 스트리밍 다운로드 (큰 파일도 메모리 효율적)
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_SEC) as client:
            async with client.stream("GET", video_url) as response:
                # HTTP 상태 코드 체크
                if response.status_code != 200:
                    raise VideoDownloadError(
                        code="VIDEO_DOWNLOAD_FAILED",
                        message=f"영상 다운로드 실패 (HTTP {response.status_code})",
                        details=f"URL response status: {response.status_code}",
                    )

                # 파일 크기 체크 (Content-Length 헤더)
                content_length = response.headers.get("content-length")
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > MAX_VIDEO_SIZE_MB:
                        raise VideoDownloadError(
                            code="VIDEO_TOO_LARGE",
                            message=f"영상 크기 초과: {size_mb:.1f}MB (최대 {MAX_VIDEO_SIZE_MB}MB)",
                        )

                # 청크 단위로 다운로드 (8KB씩)
                with open(save_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)

        # 다운로드 완료 후 파일 크기 검증
        actual_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        if actual_size_mb > MAX_VIDEO_SIZE_MB:
            os.remove(save_path)
            raise VideoDownloadError(
                code="VIDEO_TOO_LARGE",
                message=f"다운로드된 영상 크기 초과: {actual_size_mb:.1f}MB",
            )

        return str(save_path)

    except httpx.TimeoutException as e:
        raise VideoDownloadError(
            code="VIDEO_DOWNLOAD_FAILED",
            message="영상 다운로드 시간 초과",
            details=str(e),
        )

    except httpx.RequestError as e:
        raise VideoDownloadError(
            code="INVALID_VIDEO_URL",
            message="영상 URL에 접근할 수 없음",
            details=str(e),
        )

    except VideoDownloadError:
        raise

    except Exception as e:
        raise VideoDownloadError(
            code="VIDEO_DOWNLOAD_FAILED",
            message="영상 다운로드 중 알 수 없는 오류 발생",
            details=str(e),
        )


def delete_video(file_path: str) -> bool:
    """
    다운로드된 영상 파일 삭제 (분석 완료 후 정리용)

    Args:
        file_path: 삭제할 파일 경로

    Returns:
        삭제 성공 여부
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception:
        return False


def _extract_extension(video_url: str) -> str:
    """
    URL에서 파일 확장자 추출
    Pre-signed URL은 쿼리 파라미터가 붙어있을 수 있음
    """
    # 쿼리 파라미터 제거
    url_without_query = video_url.split("?")[0]

    # 확장자 추출
    extension = Path(url_without_query).suffix.lower()

    # 지원 포맷 체크
    if extension not in ALLOWED_EXTENSIONS:
        raise VideoDownloadError(
            code="UNSUPPORTED_VIDEO_FORMAT",
            message=f"지원하지 않는 영상 포맷: {extension}",
            details=f"지원 포맷: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    return extension