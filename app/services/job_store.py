"""
Mock 작업 저장소 (인메모리)
실제 서비스에선 Redis나 DB 사용
"""
from datetime import datetime
from typing import Optional


class JobStore:
    """간단한 인메모리 작업 저장소"""

    def __init__(self):
        self._jobs: dict = {}

    def create_job(self, job_id: str, user_id: str) -> dict:
        """새 작업 생성"""
        now = datetime.utcnow()
        job = {
            "job_id": job_id,
            "user_id": user_id,
            "status": "accepted",
            "progress": 0,
            "current_step": "download_video",
            "started_at": now,
            "updated_at": now,
        }
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[dict]:
        """작업 조회"""
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs) -> Optional[dict]:
        """작업 업데이트"""
        if job_id not in self._jobs:
            return None
        self._jobs[job_id].update(kwargs)
        self._jobs[job_id]["updated_at"] = datetime.utcnow()
        return self._jobs[job_id]

    def exists(self, job_id: str) -> bool:
        """작업 존재 여부"""
        return job_id in self._jobs


# 싱글톤 인스턴스
job_store = JobStore()