"""API routes for industrial anomaly detection and explanation."""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.services.detector_service import inference_service

router = APIRouter(prefix="/detection", tags=["detection"])

@router.get("/samples")
async def list_sample_images():
    """Returns available built-in demo industrial samples."""
    return {"samples": inference_service.get_sample_list()}

@router.post("/analyze")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    category: str = Form("bottle"),
    detector: str = Form("winclip"),
    k_shots: int = Form(2),
    explainer: str = Form("vit_gradcam"),
):
    """Analyzes an uploaded image or sample for industrial defects with visual explanations."""
    file_bytes = None
    if file is not None:
        file_bytes = await file.read()
    
    if not file_bytes and not sample_id:
        raise HTTPException(status_code=400, detail="Must provide an image upload or a sample_id.")
        
    try:
        img = inference_service.load_image(file_bytes=file_bytes, sample_id=sample_id)
        result = inference_service.analyze(
            img=img,
            category=category,
            detector_type=detector,
            k_shots=k_shots,
            explainer_type=explainer
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
