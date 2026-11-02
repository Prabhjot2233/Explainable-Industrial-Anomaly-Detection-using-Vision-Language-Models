"""Aggregates all API sub-routers."""
from fastapi import APIRouter

from app.api.images import router as images_router
from app.api.detection import router as detection_router

router = APIRouter(prefix="/api")
router.include_router(images_router)
router.include_router(detection_router)