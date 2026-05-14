"""
FastAPI 라이프사이클
- 시작 시 VLM + CUT3R 핸들 로드 → app.state에 보관
- 종료 시 정리

stub_models=True 인 동안은 핸들이 None이고, 라우터/서비스가 stub 분기로 동작.
"""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.core.config import settings
from app.services import cache_service
from app.vlm import loader as vlm_loader

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI server (stub_models=%s)", settings.stub_models)

    # 1) VLM
    try:
        app.state.vlm = vlm_loader.load_vlm()
    except Exception as e:
        logger.exception("VLM load failed; running without VLM")
        app.state.vlm = None

    # 2) CUT3R
    try:
        app.state.cut3r = vlm_loader.load_cut3r()
    except Exception as e:
        logger.exception("CUT3R load failed; running without CUT3R")
        app.state.cut3r = None

    # 3) video tensor 캐시에 image_processor 주입 (VLM이 있을 때만)
    if app.state.vlm is not None:
        cache_service.init_video_cache(app.state.vlm["image_processor"])

    yield

    logger.info("Shutting down.")
    app.state.vlm = None
    app.state.cut3r = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
