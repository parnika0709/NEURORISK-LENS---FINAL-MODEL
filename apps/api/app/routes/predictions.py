"""Prediction endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.services import CaseService, InferenceService
from app.schemas import PredictionRead

router = APIRouter(prefix="/cases", tags=["Predictions"])
case_service = CaseService()
inference_service = InferenceService()


class PredictRequest(BaseModel):
    model_name: str | None = None


@router.post("/{case_id}/predict", status_code=202)
async def run_prediction(
    case_id: int,
    model_name: str | None = None,
    body: PredictRequest | None = None,
):
    """Trigger ML inference for a case with selectable model architecture.

    Supported model_name values:
      - 'attn_unet_2d' (Baseline 2D Attention U-Net, Default)
      - 'unet_3d' (Independent 3D Volumetric U-Net, Experimental)
    """
    case = case_service.get_case_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")

    # The volume is stored in the upload_dir specified in Case.nifti_path
    case_dir = Path(case.nifti_path)

    # Resolve requested model: body takes precedence, then query param, then default
    selected_model = (
        (body.model_name if body and body.model_name else None)
        or model_name
        or settings.active_predictor
    )

    if selected_model not in ("attn_unet_2d", "unet_3d"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model_name '{selected_model}'. Supported models: ['attn_unet_2d', 'unet_3d']",
        )

    try:
        prediction = inference_service.run_inference(case_id, case_dir, model_name=selected_model)
        return {
            "message": "Inference completed successfully",
            "prediction_id": prediction.id,
            "model_name": prediction.model_name,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/{case_id}/prediction", response_model=PredictionRead | None)
async def get_latest_prediction(case_id: int):
    """Get the most recent prediction for a case."""
    prediction = case_service.get_latest_case_prediction(case_id)
    return prediction
