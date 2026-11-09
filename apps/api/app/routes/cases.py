"""Case endpoints."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.config import settings
from app.schemas import CaseCreate, CaseRead
from app.services import CaseService

router = APIRouter(prefix="/cases", tags=["Cases"])
case_service = CaseService()


def _save_modality(upload_dir: Path, case_name: str, suffix: str, file: UploadFile) -> None:
    """Save an uploaded modality file with the suffix the loader expects.

    The inference loader looks for a file whose stem *ends with* '_t1ce',
    '_t2', or '_flair', so we normalize the filename here regardless of what
    the user originally named it.
    """
    ext = ".nii.gz" if file.filename and file.filename.endswith(".nii.gz") else ".nii"
    dest = upload_dir / f"{case_name}_{suffix}{ext}"
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


@router.post("/", response_model=CaseRead, status_code=201)
async def create_case(
    case_name: Annotated[str, Form()],
    patient_id: Annotated[str | None, Form()] = None,
    modality_summary: Annotated[str | None, Form()] = None,
    t1ce_file: Annotated[UploadFile, File()] = None,
    t2_file: Annotated[UploadFile, File()] = None,
    flair_file: Annotated[UploadFile, File()] = None,
):
    """Upload the three MRI modalities (T1ce, T2, FLAIR) required for a case."""
    if not (t1ce_file and t2_file and flair_file):
        raise HTTPException(
            status_code=400,
            detail="All three files are required: t1ce_file, t2_file, flair_file.",
        )

    case_id_tmp = str(uuid.uuid4())
    upload_dir = settings.upload_dir / case_id_tmp
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = case_name.replace(" ", "_") or "case"
    _save_modality(upload_dir, safe_name, "t1ce", t1ce_file)
    _save_modality(upload_dir, safe_name, "t2", t2_file)
    _save_modality(upload_dir, safe_name, "flair", flair_file)

    case_data = CaseCreate(
        case_name=case_name,
        patient_id=patient_id,
        modality_summary=modality_summary,
    )

    return case_service.create_case(case_data, str(upload_dir))


@router.get("/", response_model=list[CaseRead])
async def list_cases():
    """List all MRI cases."""
    return case_service.list_all_cases()


@router.get("/{case_id}", response_model=CaseRead)
async def get_case(case_id: int):
    """Get details for a specific case."""
    case = case_service.get_case_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    return case
