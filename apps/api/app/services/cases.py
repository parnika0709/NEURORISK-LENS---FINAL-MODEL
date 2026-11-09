"""Service for managing MRI cases and their associated predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import settings
from app.db import Case, Prediction, get_session, list_cases, get_case, get_latest_prediction, to_dict
from app.schemas import CaseCreate, CaseRead

class CaseService:
    @staticmethod
    def create_case(case_data: CaseCreate, nifti_path: str) -> CaseRead:
        with get_session() as session:
            case = Case(
                case_name=case_data.case_name,
                patient_id=case_data.patient_id,
                modality_summary=case_data.modality_summary,
                nifti_path=nifti_path
            )
            session.add(case)
            session.commit()
            session.refresh(case)
            return CaseRead.model_validate(case)

    @staticmethod
    def list_all_cases() -> list[CaseRead]:
        cases = list_cases()
        return [CaseRead.model_validate(c) for c in cases]

    @staticmethod
    def get_case_by_id(case_id: int) -> CaseRead | None:
        case = get_case(case_id)
        return CaseRead.model_validate(case) if case else None

    @staticmethod
    def get_latest_case_prediction(case_id: int) -> Prediction | None:
        pred = get_latest_prediction(case_id)
        return pred
