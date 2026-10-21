"""Aggregates all API sub-routers. Add new routers (detection, explainability) here."""
from fastapi import APIRouter

from app.api.images import router as images_router

router = APIRouter(prefix="/api")
router.include_router(images_router)