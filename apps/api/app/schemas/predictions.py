"""Prediction schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, model_validator

from braintumor.schemas.contracts import TumorFeaturesBundle


class PredictionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    case_id: int
    model_name: str
    model_version: str
    inference_seconds: float
    mask_path: str
    uncertainty_path: str | None = None
    saliency_path: str | None = None
    mesh_paths: dict[str, Any]
    features: TumorFeaturesBundle | dict[str, Any]
    metrics: dict[str, Any] | None = None
    risk_regions: list[dict[str, Any]] | None = None
    risk_report: str | None = None
    tumor_detection_status: str | None = None
    is_no_tumor: bool | None = None
    analysis_details: dict[str, Any] | None = None
    created_at: datetime

    @model_validator(mode="after")
    def populate_detection_status(self) -> PredictionRead:
        if self.tumor_detection_status is None:
            if self.risk_regions and all(
                r.get("tumor_detection_status") == "NO_TUMOR_DETECTED"
                or (
                    r.get("risk_tier") in ("NOT_APPLICABLE", "No tumor detected")
                    and r.get("mapping_status") != "FAILED_REGISTRATION"
                    and r.get("registration_quality") != "FAILED"
                )
                for r in self.risk_regions
            ):
                self.tumor_detection_status = "NO_TUMOR_DETECTED"
            else:
                self.tumor_detection_status = "TUMOR_DETECTED"
        if self.is_no_tumor is None:
            self.is_no_tumor = self.tumor_detection_status == "NO_TUMOR_DETECTED"
        return self
