"""Case schemas."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, ConfigDict


class CaseBase(BaseModel):
    case_name: str
    patient_id: str | None = None
    modality_summary: str | None = None


class CaseCreate(CaseBase):
    pass


class CaseUpdate(BaseModel):
    case_name: str | None = None
    patient_id: str | None = None
    modality_summary: str | None = None


class CaseRead(CaseBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    nifti_path: str
    created_at: datetime
