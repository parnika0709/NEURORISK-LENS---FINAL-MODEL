"""Four-modality upload and trained-checkpoint segmentation API."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from matplotlib.lines import Line2D
from pydantic import BaseModel, Field

from app.config import settings
from app.db import Prediction, get_case, get_prediction, get_session
from app.schemas.cases import CaseCreate
from app.services.cases import CaseService
from app.services.model_runtime import (
    MODEL_MODALITIES,
    ModelRuntime,
    estimate_resampled_shape,
)
from braintumor.risk import (
    build_report_text,
    compute_risk_regions,
    generate_registration_qc_image,
    register_patient_to_mni,
)
from braintumor.schemas.contracts import VolumeModality

router = APIRouter(prefix="/api", tags=["Trained BraTS Models"])
case_service = CaseService()
model_runtime = ModelRuntime()
_safe_case_name = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
_NATIVE_LABELS = (0, 1, 2, 4)


class PredictionRequest(BaseModel):
    case_id: int = Field(gt=0)
    model_name: str = Field(pattern=r"^(attn_unet_2d|unet_3d)$")


async def _save_upload(
    upload: UploadFile,
    destination: Path,
    total_bytes: list[int],
) -> int:
    filename = (upload.filename or "").lower()
    if not filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=415, detail="Only .nii and .nii.gz files are accepted."
        )
    written = 0
    with destination.open("wb") as target:
        while chunk := await upload.read(_CHUNK_BYTES):
            written += len(chunk)
            total_bytes[0] += len(chunk)
            if total_bytes[0] > _MAX_UPLOAD_BYTES:
                target.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Combined upload exceeds {settings.max_upload_mb} MiB.",
                )
            target.write(chunk)
        target.flush()
    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded NIfTI file is empty.")
    return written


def _validate_nifti(path: Path, label_file: bool = False) -> nib.spatialimages.SpatialImage:
    try:
        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise ValueError(f"Expected a 3D NIfTI volume, got {image.shape}.")
        affine = np.asarray(image.affine, dtype=np.float64)
        if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
            raise ValueError("NIfTI affine is invalid or singular.")
        spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("NIfTI voxel spacing is invalid.")
        data = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(data).all() or not np.any(data):
            raise ValueError("NIfTI contains no finite nonzero voxel data.")
        if label_file:
            unique_labels = np.unique(data)
            if not np.isin(unique_labels, _NATIVE_LABELS).all():
                raise ValueError(
                    f"Ground truth contains labels outside {_NATIVE_LABELS}."
                )
        return image
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid NIfTI file {path.name}: {error}",
        ) from error


def _validate_alignment(
    images: list[nib.spatialimages.SpatialImage],
    names: list[str],
) -> None:
    reference = images[0]
    reference_affine = np.asarray(reference.affine, dtype=np.float64)
    reference_spacing = np.asarray(reference.header.get_zooms()[:3])
    for name, image in zip(names[1:], images[1:]):
        if image.shape != reference.shape:
            raise HTTPException(
                status_code=422,
                detail=f"{name} shape {image.shape} does not match FLAIR {reference.shape}.",
            )
        if not np.allclose(image.affine, reference_affine, rtol=0.0, atol=1e-4):
            raise HTTPException(
                status_code=422,
                detail=f"{name} affine is not spatially aligned with FLAIR.",
            )
        if not np.allclose(
            image.header.get_zooms()[:3],
            reference_spacing,
            rtol=0.0,
            atol=1e-4,
        ):
            raise HTTPException(
                status_code=422,
                detail=f"{name} voxel spacing does not match FLAIR.",
            )


def _find_modalities(case_dir: Path) -> dict[VolumeModality, Path]:
    files: dict[VolumeModality, Path] = {}
    for modality in MODEL_MODALITIES["attn_unet_2d"]:
        suffix = f"_{modality.value}"
        matches = sorted(
            path
            for path in case_dir.iterdir()
            if path.is_file()
            and path.name.lower().endswith((suffix + ".nii", suffix + ".nii.gz"))
        )
        if len(matches) != 1:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Expected exactly one {modality.value.upper()} NIfTI file; "
                    f"found {len(matches)}."
                ),
            )
        files[modality] = matches[0]
    return files


def _case_groundtruth(case_dir: Path) -> Path | None:
    matches = sorted(
        path
        for path in case_dir.iterdir()
        if path.is_file()
        and path.name.lower().endswith(("_groundtruth.nii", "_groundtruth.nii.gz"))
    )
    if len(matches) > 1:
        raise HTTPException(
            status_code=422, detail="Only one optional ground-truth mask is supported."
        )
    return matches[0] if matches else None


def _groundtruth_metrics(
    prediction: np.ndarray,
    groundtruth: np.ndarray,
    voxel_volume_mm3: float,
) -> dict[str, Any]:
    regions = {
        "WT": (1, 2, 4),
        "TC": (1, 4),
        "ET": (4,),
    }
    results: dict[str, Any] = {}
    for name, labels in regions.items():
        predicted = np.isin(prediction, labels)
        target = np.isin(groundtruth, labels)
        true_positive = int(np.count_nonzero(predicted & target))
        false_positive = int(np.count_nonzero(predicted & ~target))
        false_negative = int(np.count_nonzero(~predicted & target))
        denominator = 2 * true_positive + false_positive + false_negative
        results[name] = {
            "dice": (
                1.0 if denominator == 0 else 2 * true_positive / denominator
            ),
            "iou": (
                1.0
                if true_positive + false_positive + false_negative == 0
                else true_positive
                / (true_positive + false_positive + false_negative)
            ),
            "precision": (
                1.0
                if true_positive + false_positive == 0
                else true_positive / (true_positive + false_positive)
            ),
            "recall": (
                1.0
                if true_positive + false_negative == 0
                else true_positive / (true_positive + false_negative)
            ),
            "predicted_volume_mm3": float(
                np.count_nonzero(predicted) * voxel_volume_mm3
            ),
            "ground_truth_volume_mm3": float(
                np.count_nonzero(target) * voxel_volume_mm3
            ),
        }
    results["aggregation"] = "single uploaded case, voxel-wise within each region"
    return results


def _save_mask(mask: np.ndarray, reference: nib.spatialimages.SpatialImage, path: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header=header)
    output.set_qform(reference.affine, code=1)
    output.set_sform(reference.affine, code=1)
    nib.save(output, str(path))


def _safe_result_path(path: str) -> Path:
    root = settings.results_dir.resolve()
    candidate = Path(path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Prediction artifact was not found.")
    return candidate


def _build_analysis_details(
    case_name: str,
    case_id: int,
    prediction_id: int | None,
    model_name: str,
    checkpoint_epoch: int,
    device: str,
    inference_seconds: float,
    parameter_count: int,
    modalities: list[str],
    volume_dimensions: list[int],
    voxel_spacing_mm: list[float],
    voxel_volume_mm3: float,
    tumor_regions: dict[str, Any],
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    model_display_name = (
        "2D Attention U-Net"
        if model_name == "attn_unet_2d"
        else ("3D U-Net" if model_name == "unet_3d" else model_name)
    )
    return {
        "case_name": case_name,
        "case_id": case_id,
        "prediction_id": prediction_id,
        "model_name": model_name,
        "model_display_name": model_display_name,
        "checkpoint_epoch": checkpoint_epoch,
        "device": device,
        "inference_seconds": inference_seconds,
        "parameter_count": parameter_count,
        "modalities": [m.upper() for m in modalities],
        "volume_dimensions": volume_dimensions,
        "voxel_spacing_mm": voxel_spacing_mm,
        "voxel_volume_mm3": voxel_volume_mm3,
        "tumor_regions": tumor_regions,
        "has_ground_truth": metrics is not None,
        "ground_truth_metrics": metrics,
    }


def _prediction_payload(prediction: Prediction) -> dict[str, Any]:
    metadata_path = prediction.mesh_paths.get("report")
    if not metadata_path:
        raise HTTPException(status_code=404, detail="Prediction report is unavailable.")
    path = _safe_result_path(metadata_path)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500, detail="Saved prediction metadata could not be read."
        ) from error
    if "analysis_details" not in metadata and "tumor_regions" in metadata:
        case_id = metadata.get("case_id", prediction.case_id)
        metadata["analysis_details"] = _build_analysis_details(
            case_name=f"case_{case_id}",
            case_id=case_id,
            prediction_id=prediction.id,
            model_name=metadata.get("model_name", prediction.model_name),
            checkpoint_epoch=metadata.get("checkpoint_epoch", 0),
            device=metadata.get("device", "cpu"),
            inference_seconds=metadata.get("inference_seconds", prediction.inference_seconds),
            parameter_count=metadata.get("parameter_count", 0),
            modalities=metadata.get("modalities", ["flair", "t1", "t1ce", "t2"]),
            volume_dimensions=metadata.get("input_shape_xyz", []),
            voxel_spacing_mm=metadata.get("voxel_spacing_xyz_mm", []),
            voxel_volume_mm3=metadata.get("voxel_volume_mm3", 1.0),
            tumor_regions=metadata.get("tumor_regions", {}),
            metrics=metadata.get("ground_truth_metrics"),
        )
    return metadata


@router.get("/health")
def health() -> dict[str, Any]:
    return model_runtime.health_info()


@router.get("/models")
def models() -> dict[str, Any]:
    return {"models": model_runtime.model_status(), **model_runtime.device_info()}


@router.post("/upload", status_code=201)
async def upload_case(
    case_name: Annotated[str, Form(min_length=1, max_length=120)],
    flair_file: Annotated[UploadFile, File()],
    t1ce_file: Annotated[UploadFile, File()],
    t2_file: Annotated[UploadFile, File()],
    t1_file: Annotated[UploadFile | None, File()] = None,
    segmentation_file: Annotated[UploadFile | None, File()] = None,
) -> dict[str, Any]:
    safe_name = _safe_case_name.sub("_", case_name).strip("_-")[:80] or "case"
    upload_dir = settings.upload_dir / str(uuid.uuid4())
    upload_dir.mkdir(parents=True, exist_ok=False)
    primary_uploads = {
        VolumeModality.FLAIR: flair_file,
        VolumeModality.T1CE: t1ce_file,
        VolumeModality.T2: t2_file,
    }
    total_bytes = [0]
    paths: dict[VolumeModality, Path] = {}
    try:
        for modality, upload in primary_uploads.items():
            filename = (upload.filename or "").lower()
            extension = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
            destination = upload_dir / f"{safe_name}_{modality.value}{extension}"
            await _save_upload(upload, destination, total_bytes)
            paths[modality] = destination

        # Handle T1: If t1_file is explicitly provided, save it; otherwise duplicate the uploaded T1CE file
        # so the user never has to upload T1CE twice.
        if t1_file is not None and t1_file.filename:
            t1_name = t1_file.filename.lower()
            t1_ext = ".nii.gz" if t1_name.endswith(".nii.gz") else ".nii"
            t1_dest = upload_dir / f"{safe_name}_t1{t1_ext}"
            await _save_upload(t1_file, t1_dest, total_bytes)
            paths[VolumeModality.T1] = t1_dest
        else:
            t1ce_path = paths[VolumeModality.T1CE]
            t1_ext = ".nii.gz" if t1ce_path.name.endswith(".nii.gz") else ".nii"
            t1_dest = upload_dir / f"{safe_name}_t1{t1_ext}"
            shutil.copyfile(t1ce_path, t1_dest)
            paths[VolumeModality.T1] = t1_dest

        images = [
            _validate_nifti(paths[modality])
            for modality in MODEL_MODALITIES["attn_unet_2d"]
        ]
        _validate_alignment(
            images, [modality.value.upper() for modality in MODEL_MODALITIES["attn_unet_2d"]]
        )
        if int(np.prod(images[0].shape)) > settings.max_volume_voxels:
            raise HTTPException(
                status_code=413,
                detail=f"Volume exceeds the {settings.max_volume_voxels}-voxel limit.",
            )
        resampled_shape = estimate_resampled_shape(
            tuple(int(value) for value in images[0].shape),
            np.asarray(images[0].affine, dtype=np.float64),
        )
        if int(np.prod(resampled_shape)) > settings.max_volume_voxels:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Resampled RAS volume exceeds the "
                    f"{settings.max_volume_voxels}-voxel limit."
                ),
            )
        if segmentation_file is not None and segmentation_file.filename:
            filename = segmentation_file.filename.lower()
            extension = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
            destination = upload_dir / f"{safe_name}_groundtruth{extension}"
            await _save_upload(segmentation_file, destination, total_bytes)
            groundtruth = _validate_nifti(destination, label_file=True)
            _validate_alignment(
                [images[0], groundtruth],
                ["FLAIR", "ground-truth segmentation"],
            )
        case = case_service.create_case(
            CaseCreate(
                case_name=case_name,
                patient_id=None,
                modality_summary="flair,t1,t1ce,t2",
            ),
            str(upload_dir),
        )
        return {
            "case_id": case.id,
            "case_name": case.case_name,
            "modalities": ["flair", "t1", "t1ce", "t2"],
            "shape_xyz": list(images[0].shape),
            "spacing_xyz_mm": [
                float(value) for value in images[0].header.get_zooms()[:3]
            ],
            "ground_truth_available": segmentation_file is not None
            and bool(segmentation_file.filename),
        }
    except HTTPException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    except (OSError, ValueError) as error:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    finally:
        for upload in primary_uploads.values():
            await upload.close()
        if t1_file is not None:
            await t1_file.close()
        if segmentation_file is not None:
            await segmentation_file.close()


@router.post("/predict", status_code=200)
def predict(request: PredictionRequest) -> dict[str, Any]:
    case = get_case(request.case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    case_dir = Path(case.nifti_path)
    if not case_dir.is_dir():
        raise HTTPException(status_code=404, detail="Uploaded MRI case directory was not found.")
    modality_paths = _find_modalities(case_dir)
    try:
        inference = model_runtime.predict(
            request.model_name, modality_paths
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (ValueError, nib.filebasedimages.ImageFileError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except torch.cuda.OutOfMemoryError as error:
        raise HTTPException(
            status_code=503,
            detail="GPU and CPU inference ran out of memory; reduce MRI volume dimensions.",
        ) from error
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise HTTPException(
                status_code=503,
                detail="GPU and CPU inference ran out of memory; reduce MRI volume dimensions.",
            ) from error
        raise HTTPException(status_code=500, detail=f"Inference failed: {error}") from error

    reference = inference.source_image
    voxel_volume_mm3 = float(abs(np.linalg.det(inference.affine[:3, :3])))
    region_masks = {
        "wt": np.isin(inference.mask, (1, 2, 4)),
        "tc": np.isin(inference.mask, (1, 4)),
        "et": inference.mask == 4,
    }
    region_volumes = {
        name.upper(): {
            "voxel_count": int(mask.sum()),
            "volume_mm3": float(mask.sum() * voxel_volume_mm3),
            "volume_cm3": float(mask.sum() * voxel_volume_mm3 / 1000.0),
        }
        for name, mask in region_masks.items()
    }
    groundtruth_path = _case_groundtruth(case_dir)
    metrics = None
    if groundtruth_path is not None:
        groundtruth_image = _validate_nifti(groundtruth_path, label_file=True)
        _validate_alignment([reference, groundtruth_image], ["FLAIR", "ground truth"])
        groundtruth = np.asarray(groundtruth_image.dataobj, dtype=np.uint8)
        metrics = _groundtruth_metrics(
            inference.mask, groundtruth, voxel_volume_mm3
        )

    artifact_dir = settings.results_dir / f"case_{case.id}" / str(uuid.uuid4())
    artifact_dir.mkdir(parents=True, exist_ok=False)
    try:
        mask_paths = {"mask": artifact_dir / "segmentation.nii.gz"}
        _save_mask(inference.mask, reference, mask_paths["mask"])
        for name, binary in region_masks.items():
            mask_paths[name] = artifact_dir / f"{name}.nii.gz"
            _save_mask(binary.astype(np.uint8), reference, mask_paths[name])
        case_path = modality_paths[VolumeModality.T1CE]
        source_t1ce = nib.load(str(case_path))
        source_data = np.asarray(source_t1ce.dataobj, dtype=np.float32)
        visualization_path = artifact_dir / "visualization.png"
        slice_indices = (
            int(np.argmax(region_masks["wt"].sum(axis=(0, 1)))),
            int(np.argmax(region_masks["wt"].sum(axis=(0, 2)))),
            int(np.argmax(region_masks["wt"].sum(axis=(1, 2)))),
        )
        views = (
            (
                "Axial",
                source_data[:, :, slice_indices[0]].T,
                inference.mask[:, :, slice_indices[0]].T,
            ),
            (
                "Coronal",
                source_data[:, slice_indices[1], :].T,
                inference.mask[:, slice_indices[1], :].T,
            ),
            (
                "Sagittal",
                source_data[slice_indices[2], :, :].T,
                inference.mask[slice_indices[2], :, :].T,
            ),
        )
        region_colors = {"WT": "#36d9e8", "TC": "#a78bfa", "ET": "#ffb45e"}
        figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for axis, (plane, source_slice, mask_slice), slice_index in zip(
            axes, views, slice_indices
        ):
            axis.imshow(source_slice, cmap="gray", origin="lower")
            for name, labels in (("WT", (1, 2, 4)), ("TC", (1, 4)), ("ET", (4,))):
                region = np.isin(mask_slice, labels)
                if region.any():
                    axis.contour(
                        region,
                        levels=[0.5],
                        colors=[region_colors[name]],
                        linewidths=0.8,
                        origin="lower",
                    )
            axis.set_title(f"{plane} · slice {slice_index}")
            axis.axis("off")
        figure.suptitle("T1CE source MRI with predicted tumor class boundaries")
        figure.legend(
            handles=[
                Line2D([0], [0], color=color, lw=2, label=name)
                for name, color in region_colors.items()
            ],
            loc="lower center",
            ncol=3,
        )
        figure.savefig(visualization_path, dpi=140)
        plt.close(figure)

        # Check the actual segmentation mask before calculating eloquent brain structure risk
        wt_mask = region_masks["wt"]
        is_empty_mask = not bool(wt_mask.any())
        tumor_detection_status = (
            "NO_TUMOR_DETECTED" if is_empty_mask else "TUMOR_DETECTED"
        )
        is_no_tumor = (tumor_detection_status == "NO_TUMOR_DETECTED")

        voxel_spacing = tuple(
            float(value) for value in nib.affines.voxel_sizes(inference.affine)
        )
        t1_ref = modality_paths.get(VolumeModality.T1) or modality_paths.get(VolumeModality.T1CE)
        t1ce_ref = modality_paths.get(VolumeModality.T1CE) or modality_paths.get(VolumeModality.T1)

        reg_res = None
        if t1_ref and not is_empty_mask:
            try:
                reg_res = register_patient_to_mni(
                    patient_image=str(t1_ref),
                    patient_affine=inference.affine,
                    patient_spacing=voxel_spacing,
                    tumor_mask=wt_mask,
                    cost_function_masking=True,
                )
            except Exception as reg_err:
                logger.warning("Pipeline registration attempt failed: %s", reg_err)

        risk_regions_objs = compute_risk_regions(
            tumor_mask=wt_mask,
            affine=inference.affine,
            spacing=voxel_spacing,
            t1_image=str(t1_ref) if t1_ref else None,
            registration_result=reg_res,
            cost_function_masking=True,
        )

        qc_registration_path = artifact_dir / "qc_registration.png"
        if reg_res and reg_res.success and reg_res.warped_atlas is not None and t1ce_ref:
            try:
                generate_registration_qc_image(
                    patient_image=str(t1ce_ref),
                    tumor_mask=wt_mask,
                    warped_atlas=reg_res.warped_atlas,
                    output_path=qc_registration_path,
                )
            except Exception as qc_err:
                logger.warning("Could not generate registration QC image: %s", qc_err)

        wt_volume_mm3 = 0.0 if is_no_tumor else float(region_volumes["WT"]["volume_mm3"])

        risk_report_text = build_report_text(
            case_name=case.case_name or f"case_{case.id}",
            regions=risk_regions_objs,
            tumor_volume_mm3=wt_volume_mm3,
        )
        risk_regions_payload = [
            {
                "region": r.region,
                "function": r.function,
                "overlap_pct": float(r.overlap_pct) if r.overlap_pct is not None else None,
                "distance_mm": float(r.distance_mm) if r.distance_mm is not None else None,
                "risk_tier": r.risk_tier,
                "eloquent_structure_risk": r.eloquent_structure_risk,
                "tumor_detection_status": r.tumor_detection_status,
                "region_overlap_pct": float(r.region_overlap_pct) if r.region_overlap_pct is not None else None,
                "overlap_volume_mm3": float(r.overlap_volume_mm3),
                "overlap_voxels": int(getattr(r, "overlap_voxels", 0)),
                "contact_type": getattr(r, "contact_type", "none"),
                "contact_label": getattr(r, "contact_label", "none"),
                "region_volume_mm3": float(r.region_volume_mm3),
                "tumor_volume_mm3": float(r.tumor_volume_mm3),
                "mapping_status": r.mapping_status,
                "registration_quality": getattr(r, "registration_quality", "PASSED"),
                "mapping_reason": getattr(r, "mapping_reason", None),
            }
            for r in risk_regions_objs
        ]

        plain_report_path = artifact_dir / "report.txt"
        plain_report_path.write_text(risk_report_text + "\n", encoding="utf-8")

        metadata_path = artifact_dir / "report.json"
        analysis_details = _build_analysis_details(
            case_name=case.case_name or f"case_{case.id}",
            case_id=case.id,
            prediction_id=None,
            model_name=inference.model_name,
            checkpoint_epoch=inference.checkpoint_epoch,
            device=inference.device,
            inference_seconds=inference.inference_seconds,
            parameter_count=inference.parameter_count,
            modalities=[modality.value for modality in MODEL_MODALITIES[request.model_name]],
            volume_dimensions=list(inference.mask.shape),
            voxel_spacing_mm=[
                float(value) for value in nib.affines.voxel_sizes(inference.affine)
            ],
            voxel_volume_mm3=voxel_volume_mm3,
            tumor_regions=region_volumes,
            metrics=metrics,
        )
        metadata = {
            "status": "completed",
            "tumor_detection_status": tumor_detection_status,
            "overall_status": "No tumor detected by the model" if is_no_tumor else "completed",
            "is_no_tumor": is_no_tumor,
            "prediction_id": None,
            "case_id": case.id,
            "model_name": inference.model_name,
            "checkpoint_epoch": inference.checkpoint_epoch,
            "parameter_count": inference.parameter_count,
            "device": inference.device,
            "inference_seconds": inference.inference_seconds,
            "gpu_memory": inference.gpu_memory,
            "modalities": [modality.value for modality in MODEL_MODALITIES[request.model_name]],
            "input_shape_xyz": list(inference.mask.shape),
            "voxel_spacing_xyz_mm": [
                float(value) for value in nib.affines.voxel_sizes(inference.affine)
            ],
            "voxel_volume_mm3": voxel_volume_mm3,
            "segmentation_labels": {
                "0": "background",
                "1": "NCR/NET",
                "2": "edema",
                "4": "enhancing_tumor",
            },
            "tumor_regions": region_volumes,
            "ground_truth_metrics": metrics,
            "risk_regions": risk_regions_payload,
            "risk_report": risk_report_text,
            "analysis_details": analysis_details,
            "metric_scope": (
                "single uploaded case with ground truth"
                if metrics is not None
                else "not calculated; no ground-truth segmentation uploaded"
            ),
            "disclaimer": (
                "Research/educational segmentation only. A model’s negative prediction is not proof that a patient has no tumor. "
                "Tumor volume is not a clinical risk score or medical diagnosis."
            ),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        features = {"tumor_regions": region_volumes, "analysis_details": analysis_details}
        mesh_paths = {
            **{f"{name}_mask": str(path) for name, path in mask_paths.items()},
            "visualization": str(visualization_path),
            "report": str(metadata_path),
            "plain_report": str(plain_report_path),
        }
        if qc_registration_path.is_file():
            mesh_paths["qc_registration"] = str(qc_registration_path)
        with get_session() as session:
            prediction = Prediction(
                case_id=case.id,
                model_name=inference.model_name,
                model_version="trained-checkpoint",
                inference_seconds=inference.inference_seconds,
                mask_path=str(mask_paths["mask"]),
                uncertainty_path=None,
                saliency_path=None,
                mesh_paths=mesh_paths,
                features=features,
                metrics=metrics,
                risk_regions=risk_regions_payload,
                risk_report=risk_report_text,
                tumor_detection_status=tumor_detection_status,
            )
            session.add(prediction)
            session.commit()
            session.refresh(prediction)
            metadata["prediction_id"] = prediction.id
            analysis_details["prediction_id"] = prediction.id
            metadata["analysis_details"] = analysis_details
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return {
            "status": "completed",
            "tumor_detection_status": tumor_detection_status,
            "is_no_tumor": is_no_tumor,
            "prediction_id": prediction.id,
            "case_id": case.id,
            "model_name": inference.model_name,
            "device": inference.device,
            "inference_seconds": inference.inference_seconds,
            "gpu_memory": inference.gpu_memory,
            "tumor_regions": region_volumes,
            "ground_truth_metrics": metrics,
            "risk_regions": risk_regions_payload,
            "risk_report": risk_report_text,
            "analysis_details": analysis_details,
            "result_url": f"/api/results/{prediction.id}",
            "visualization_url": f"/api/results/{prediction.id}/visualization",
            "mask_url": f"/api/results/{prediction.id}/mask",
            "report_url": f"/api/results/{prediction.id}/report",
            "plain_report_url": f"/api/results/{prediction.id}/plain-report",
        }
    except Exception:
        shutil.rmtree(artifact_dir, ignore_errors=True)
        raise


@router.get("/predict/{prediction_id}")
def prediction_status(prediction_id: int) -> dict[str, Any]:
    prediction = get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    payload = _prediction_payload(prediction)
    detection_status = getattr(prediction, "tumor_detection_status", None) or payload.get("tumor_detection_status")
    return {
        "prediction_id": prediction.id,
        "status": "completed",
        "tumor_detection_status": detection_status,
        "model_name": prediction.model_name,
        "created_at": prediction.created_at.isoformat(),
        "result_url": f"/api/results/{prediction.id}",
    }


@router.get("/results/{prediction_id}")
def get_result(prediction_id: int) -> dict[str, Any]:
    prediction = get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    payload = _prediction_payload(prediction)
    if getattr(prediction, "tumor_detection_status", None):
        payload["tumor_detection_status"] = prediction.tumor_detection_status
    elif "tumor_detection_status" not in payload:
        payload["tumor_detection_status"] = "NO_TUMOR_DETECTED" if payload.get("is_no_tumor") else "TUMOR_DETECTED"
    return payload


@router.get("/results/{prediction_id}/mask")
def download_mask(prediction_id: int, region: str = "mask") -> FileResponse:
    if region not in ("mask", "wt", "tc", "et"):
        raise HTTPException(
            status_code=422, detail="region must be one of mask, wt, tc, or et."
        )
    prediction = get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    key = "mask" if region == "mask" else f"{region}_mask"
    path = prediction.mask_path if region == "mask" else prediction.mesh_paths.get(key)
    if not path:
        raise HTTPException(status_code=404, detail="Requested mask is unavailable.")
    return FileResponse(
        _safe_result_path(path),
        media_type="application/gzip",
        filename=f"prediction_{prediction_id}_{region}.nii.gz",
    )


@router.get("/results/{prediction_id}/visualization")
def prediction_visualization(prediction_id: int) -> FileResponse:
    prediction = get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    path = prediction.mesh_paths.get("visualization")
    if not path:
        raise HTTPException(status_code=404, detail="Visualization is unavailable.")
    return FileResponse(
        _safe_result_path(path),
        media_type="image/png",
        filename=f"prediction_{prediction_id}_visualization.png",
    )


@router.get("/results/{prediction_id}/report")
def download_report(prediction_id: int) -> FileResponse:
    prediction = get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    path = prediction.mesh_paths.get("report")
    if not path:
        raise HTTPException(status_code=404, detail="Report is unavailable.")
    return FileResponse(
        _safe_result_path(path),
        media_type="application/json",
        filename=f"prediction_{prediction_id}_report.json",
    )


@router.get("/results/{prediction_id}/plain-report")
def download_plain_report(prediction_id: int) -> FileResponse:
    prediction = get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    path = prediction.mesh_paths.get("plain_report")
    if not path or not Path(path).is_file():
        # Fallback: check if report.txt exists in artifact dir or generate from prediction
        report_json_path = prediction.mesh_paths.get("report")
        if report_json_path:
            artifact_dir = Path(report_json_path).parent
            candidate = artifact_dir / "report.txt"
            if candidate.is_file():
                path = str(candidate)
            elif prediction.risk_report:
                candidate.write_text(prediction.risk_report + "\n", encoding="utf-8")
                path = str(candidate)
    if not path:
        raise HTTPException(status_code=404, detail="Plain language report is unavailable.")
    return FileResponse(
        _safe_result_path(path),
        media_type="text/plain; charset=utf-8",
        filename=f"prediction_{prediction_id}_plain_report.txt",
    )


@router.get("/results/{prediction_id}/qc-registration")
def download_qc_registration(prediction_id: int) -> FileResponse:
    prediction = get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    path = prediction.mesh_paths.get("qc_registration")
    if not path or not Path(path).is_file():
        # Fallback: check if qc_registration.png exists in artifact dir
        report_json_path = prediction.mesh_paths.get("report")
        if report_json_path:
            candidate = Path(report_json_path).parent / "qc_registration.png"
            if candidate.is_file():
                path = str(candidate)
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="Registration QC image is unavailable.")
    return FileResponse(
        _safe_result_path(path),
        media_type="image/png",
        filename=f"prediction_{prediction_id}_qc_registration.png",
    )
