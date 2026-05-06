"""
공간 분석 AI 챗봇 서비스 - AI 서버
FastAPI 기반 백엔드 메인 진입점
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import analyze, jobs, health

# FastAPI 앱 생성
app = FastAPI(
    title="Spatial Analysis AI Server",
    description="단안 카메라 영상 -> 3D 공간 분석 + 자연어 응답",
    version="1.0.0",
)

# CORS 설정 (개발 단계에서는 모두 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(health.router, tags=["Health"])
app.include_router(analyze.router, prefix="/api/v1", tags=["Analyze"])
app.include_router(jobs.router, prefix="/api/v1", tags=["Jobs"])


@app.get("/")
async def root():
    """루트 페이지"""
    return {
        "service": "Spatial Analysis AI Server",
        "version": "1.0.0",
        "docs_url": "/docs",
    }