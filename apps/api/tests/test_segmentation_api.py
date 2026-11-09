from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.routes import segmentation
from app.services.model_runtime import ModelRuntime, prepare_volume
from braintumor.data.brats_loader import BratsCase
from braintumor.data.preprocessing import fourclass_to_brats
from braintumor.schemas.contracts import VolumeModality


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _nifti_bytes(path: Path, affine: np.ndarray | None = None) -> bytes:
    data = np.zeros((16, 16, 16), dtype=np.float32)
    data[3:13, 3:13, 3:13] = 100
    nib.save(
        nib.Nifti1Image(data, np.eye(4) if affine is None else affine),
        str(path),
    )
    return path.read_bytes()


def _modality_files(tmp_path: Path, affine: np.ndarray | None = None) -> dict:
    names = ("flair", "t1", "t1ce", "t2")
    return {
        f"{name}_file": (
            f"sample_{name}.nii.gz",
            _nifti_bytes(tmp_path / f"{name}.nii.gz", affine),
            "application/gzip",
        )
        for name in names
    }


def test_health_and_models_report_both_trained_models(client: TestClient) -> None:
    health = client.get("/api/health")
    models = client.get("/api/models")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert "torch_version" in health.json()
    assert models.status_code == 200
    assert {model["name"] for model in models.json()["models"]} == {
        "attn_unet_2d",
        "unet_3d",
    }
    assert all(len(model["input_modalities"]) == 4 for model in models.json()["models"])
    assert all(model["checkpoint_available"] for model in models.json()["models"])


def test_upload_requires_all_four_modalities(client: TestClient) -> None:
    response = client.post(
        "/api/upload",
        data={"case_name": "missing-modality"},
        files={
            "flair_file": ("flair.nii.gz", b"not-a-volume", "application/gzip"),
            "t1_file": ("t1.nii.gz", b"not-a-volume", "application/gzip"),
            "t1ce_file": ("t1ce.nii.gz", b"not-a-volume", "application/gzip"),
        },
    )

    assert response.status_code == 422


def test_upload_validates_geometry_and_saves_four_modalities(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    monkeypatch.setattr(settings, "upload_dir", upload_root)
    monkeypatch.setattr(
        segmentation.case_service,
        "create_case",
        lambda case, path: SimpleNamespace(id=42, case_name=case.case_name),
    )
    files = _modality_files(tmp_path)

    response = client.post(
        "/api/upload",
        data={"case_name": "smoke-case"},
        files=files,
    )

    assert response.status_code == 201
    result = response.json()
    assert result["modalities"] == ["flair", "t1", "t1ce", "t2"]
    assert result["shape_xyz"] == [16, 16, 16]
    saved_files = list((upload_root / next(iter(upload_root.iterdir())).name).glob("*.nii.gz"))
    assert len(saved_files) == 4


def test_upload_accepts_optional_aligned_ground_truth(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    monkeypatch.setattr(settings, "upload_dir", upload_root)
    monkeypatch.setattr(
        segmentation.case_service,
        "create_case",
        lambda case, path: SimpleNamespace(id=43, case_name=case.case_name),
    )
    labels = np.zeros((16, 16, 16), dtype=np.uint8)
    labels[3:8, 3:8, 3:8] = 1
    labels_path = tmp_path / "ground_truth.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.eye(4)), str(labels_path))
    files = _modality_files(tmp_path)
    files["segmentation_file"] = (
        "sample_segmentation.nii.gz",
        labels_path.read_bytes(),
        "application/gzip",
    )

    response = client.post(
        "/api/upload",
        data={"case_name": "with-ground-truth"},
        files=files,
    )

    assert response.status_code == 201
    assert response.json()["ground_truth_available"] is True
    saved_files = list((upload_root / next(iter(upload_root.iterdir())).name).glob("*.nii.gz"))
    assert len(saved_files) == 5


def test_upload_rejects_misaligned_modality(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    monkeypatch.setattr(settings, "upload_dir", upload_root)
    shifted_affine = np.eye(4)
    shifted_affine[0, 3] = 2
    files = _modality_files(tmp_path)
    files["t1_file"] = (
        "sample_t1.nii.gz",
        _nifti_bytes(tmp_path / "shifted.nii.gz", shifted_affine),
        "application/gzip",
    )

    response = client.post(
        "/api/upload",
        data={"case_name": "misaligned"},
        files=files,
    )

    assert response.status_code == 422
    assert "affine is not spatially aligned" in response.json()["detail"]
    assert list(upload_root.iterdir()) == []


def test_upload_enforces_combined_size_limit(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    monkeypatch.setattr(settings, "upload_dir", upload_root)
    monkeypatch.setattr(segmentation, "_MAX_UPLOAD_BYTES", 10)

    response = client.post(
        "/api/upload",
        data={"case_name": "too-large"},
        files=_modality_files(tmp_path),
    )

    assert response.status_code == 413
    assert list(upload_root.iterdir()) == []


@pytest.mark.parametrize("model_name", ["attn_unet_2d", "unet_3d"])
def test_training_preprocessing_preserves_four_channel_geometry(
    model_name: str, tmp_path: Path
) -> None:
    modalities = {
        VolumeModality.FLAIR: tmp_path / "flair.nii.gz",
        VolumeModality.T1: tmp_path / "t1.nii.gz",
        VolumeModality.T1CE: tmp_path / "t1ce.nii.gz",
        VolumeModality.T2: tmp_path / "t2.nii.gz",
    }
    data = np.zeros((16, 16, 16), dtype=np.float32)
    data[2:14, 2:14, 2:14] = 100
    for path in modalities.values():
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))

    prepared = prepare_volume(model_name, modalities)

    assert prepared.image.shape == (4, 16, 16, 16)
    assert np.isfinite(prepared.image).all()
    assert np.allclose(prepared.affine, np.eye(4))


def test_training_classes_map_to_native_brats_labels() -> None:
    native = fourclass_to_brats(np.array([0, 1, 2, 3], dtype=np.uint8))

    assert native.tolist() == [0, 1, 2, 4]


@pytest.mark.integration
def test_actual_trained_models_infer_validation_case() -> None:
    if os.getenv("BT_RUN_MODEL_SMOKE") != "1":
        pytest.skip("Set BT_RUN_MODEL_SMOKE=1 to run GPU checkpoint inference.")
    data_root_value = os.getenv("BT_DATA_ROOT")
    if not data_root_value:
        pytest.fail("BT_DATA_ROOT is required when BT_RUN_MODEL_SMOKE=1.")

    repo_root = Path(__file__).resolve().parents[3]
    data_root = Path(data_root_value)
    split = json.loads(
        (repo_root / "data" / "splits" / "brats2021_3d_80_20.json").read_text(
            encoding="utf-8"
        )
    )
    available_validation_cases = sorted(
        set(split["val"]).intersection(os.listdir(data_root))
    )
    if not available_validation_cases:
        pytest.fail("No case from the saved validation split exists in BT_DATA_ROOT.")

    case = BratsCase.from_dir(data_root / available_validation_cases[0])
    required = (
        VolumeModality.FLAIR,
        VolumeModality.T1,
        VolumeModality.T1CE,
        VolumeModality.T2,
    )
    if not all(modality in case.modality_paths for modality in required):
        pytest.fail("Selected validation case does not contain all four modalities.")
    paths = {modality: case.modality_paths[modality] for modality in required}

    runtime = ModelRuntime()
    for model_name in ("attn_unet_2d", "unet_3d"):
        result = runtime.predict(model_name, paths)
        assert result.mask.shape == result.source_image.shape[:3]
        assert set(np.unique(result.mask)).issubset({0, 1, 2, 4})


def test_upload_rejects_excessive_resampled_volume(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    monkeypatch.setattr(settings, "upload_dir", upload_root)
    monkeypatch.setattr(settings, "max_volume_voxels", 20_000)
    coarse_affine = np.diag([4.0, 4.0, 4.0, 1.0])
    files = _modality_files(tmp_path, coarse_affine)

    response = client.post(
        "/api/upload",
        data={"case_name": "resampling-limit"},
        files=files,
    )

    assert response.status_code == 413
    assert "Resampled RAS volume exceeds" in response.json()["detail"]
    assert list(upload_root.iterdir()) == []


def test_result_metadata_and_mask_downloads_are_available(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_root = tmp_path / "results"
    artifact_dir = result_root / "case_42" / "prediction_7"
    artifact_dir.mkdir(parents=True)
    mask_path = artifact_dir / "segmentation.nii.gz"
    mask_path.write_bytes(b"nifti-mask")
    report_path = artifact_dir / "report.json"
    report_path.write_text(json.dumps({"prediction_id": 7}), encoding="utf-8")
    visualization_path = artifact_dir / "visualization.png"
    visualization_path.write_bytes(b"png-image")
    monkeypatch.setattr(settings, "results_dir", result_root)
    monkeypatch.setattr(
        segmentation,
        "get_prediction",
        lambda prediction_id: SimpleNamespace(
            id=prediction_id,
            mask_path=str(mask_path),
            mesh_paths={
                "report": str(report_path),
                "visualization": str(visualization_path),
                "wt_mask": str(mask_path),
                "tc_mask": str(mask_path),
                "et_mask": str(mask_path),
            },
            model_name="attn_unet_2d",
            created_at=SimpleNamespace(isoformat=lambda: "2025-01-01T00:00:00"),
        ),
    )

    status = client.get("/api/predict/7")
    result = client.get("/api/results/7")
    mask = client.get("/api/results/7/mask")
    region_masks = [
        client.get(f"/api/results/7/mask?region={region}")
        for region in ("wt", "tc", "et")
    ]
    report = client.get("/api/results/7/report")
    visualization = client.get("/api/results/7/visualization")

    assert status.status_code == 200 and status.json()["status"] == "completed"
    assert result.status_code == 200
    assert result.json()["prediction_id"] == 7
    assert mask.status_code == 200 and mask.content == b"nifti-mask"
    assert all(response.status_code == 200 for response in region_masks)
    assert report.status_code == 200
    assert report.json() == {"prediction_id": 7}
    assert visualization.status_code == 200
    assert visualization.content == b"png-image"


def test_upload_without_explicit_t1_automatically_reuses_t1ce(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that uploading only FLAIR, T1CE, and T2 works without requiring a 2nd T1CE upload."""
    upload_root = tmp_path / "uploads_3mod"
    upload_root.mkdir()
    monkeypatch.setattr(settings, "upload_dir", upload_root)
    monkeypatch.setattr(
        segmentation.case_service,
        "create_case",
        lambda case, path: SimpleNamespace(id=88, case_name=case.case_name),
    )
    all_files = _modality_files(tmp_path)
    # Remove t1_file to simulate single T1CE upload
    files_3 = {k: v for k, v in all_files.items() if k != "t1_file"}

    response = client.post(
        "/api/upload",
        data={"case_name": "single-t1ce-upload"},
        files=files_3,
    )

    assert response.status_code == 201
    result = response.json()
    assert result["modalities"] == ["flair", "t1", "t1ce", "t2"]


def test_no_tumor_detected_never_reports_low_tier() -> None:
    """Verify that empty mask gives 'No tumor detected' rather than 'Low' risk."""
    from braintumor.risk import NO_TUMOR_TIER, build_report_text, compute_risk_regions

    empty_mask = np.zeros((32, 32, 32), dtype=np.uint8)
    affine = np.eye(4)
    spacing = (1.0, 1.0, 1.0)

    regions = compute_risk_regions(empty_mask, affine=affine, spacing=spacing)
    assert len(regions) > 0
    assert all(r.risk_tier == NO_TUMOR_TIER for r in regions)
    assert not any(r.risk_tier == "Low" for r in regions)

    report_text = build_report_text("case_no_tumor", regions, tumor_volume_mm3=0.0)
    assert "NO TUMOR DETECTED" in report_text
    assert "Low" not in report_text


def test_analysis_details_payload_structure(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that analysis_details contains all required fields, tumor regions, and geometry."""
    result_root = tmp_path / "results_ad"
    artifact_dir = result_root / "case_10" / "prediction_55"
    artifact_dir.mkdir(parents=True)
    report_path = artifact_dir / "report.json"

    ad_data = {
        "case_name": "TestPatient_01",
        "case_id": 10,
        "prediction_id": 55,
        "model_name": "attn_unet_2d",
        "model_display_name": "2D Attention U-Net",
        "checkpoint_epoch": 15,
        "device": "cpu",
        "inference_seconds": 1.25,
        "parameter_count": 2029816,
        "modalities": ["FLAIR", "T1", "T1CE", "T2"],
        "volume_dimensions": [240, 240, 155],
        "voxel_spacing_mm": [1.0, 1.0, 1.0],
        "voxel_volume_mm3": 1.0,
        "tumor_regions": {
            "WT": {"voxel_count": 5000, "volume_mm3": 5000.0, "volume_cm3": 5.0},
            "TC": {"voxel_count": 3000, "volume_mm3": 3000.0, "volume_cm3": 3.0},
            "ET": {"voxel_count": 1000, "volume_mm3": 1000.0, "volume_cm3": 1.0},
        },
        "has_ground_truth": False,
        "ground_truth_metrics": None,
    }
    payload = {
        "status": "completed",
        "prediction_id": 55,
        "case_id": 10,
        "model_name": "attn_unet_2d",
        "tumor_regions": ad_data["tumor_regions"],
        "analysis_details": ad_data,
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(settings, "results_dir", result_root)
    monkeypatch.setattr(
        segmentation,
        "get_prediction",
        lambda pid: SimpleNamespace(
            id=pid,
            case_id=10,
            model_name="attn_unet_2d",
            inference_seconds=1.25,
            mask_path=str(report_path),
            mesh_paths={"report": str(report_path)},
            created_at=SimpleNamespace(isoformat=lambda: "2025-01-01T00:00:00"),
        ),
    )

    resp = client.get("/api/results/55")
    assert resp.status_code == 200
    data = resp.json()
    assert "analysis_details" in data
    ad = data["analysis_details"]

    # 1. Analysis details
    assert ad["case_name"] == "TestPatient_01"
    assert ad["case_id"] == 10
    assert ad["prediction_id"] == 55
    assert ad["model_display_name"] == "2D Attention U-Net"
    assert ad["checkpoint_epoch"] == 15
    assert ad["device"] == "cpu"
    assert ad["inference_seconds"] == 1.25
    assert ad["parameter_count"] == 2029816

    # 2. Input and geometry
    assert ad["modalities"] == ["FLAIR", "T1", "T1CE", "T2"]
    assert ad["volume_dimensions"] == [240, 240, 155]
    assert ad["voxel_spacing_mm"] == [1.0, 1.0, 1.0]
    assert ad["voxel_volume_mm3"] == 1.0

    # 3. Predicted tumor-region volumes
    for key in ("WT", "TC", "ET"):
        assert key in ad["tumor_regions"]
        assert "voxel_count" in ad["tumor_regions"][key]
        assert "volume_mm3" in ad["tumor_regions"][key]
        assert "volume_cm3" in ad["tumor_regions"][key]

    # 4. Ground truth block condition
    assert ad["has_ground_truth"] is False
    assert ad["ground_truth_metrics"] is None


def test_analysis_details_ground_truth_metrics(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that when ground truth is provided, comparison metrics are populated."""
    result_root = tmp_path / "results_gt"
    artifact_dir = result_root / "case_11" / "prediction_56"
    artifact_dir.mkdir(parents=True)
    report_path = artifact_dir / "report.json"

    gt_metrics = {
        "WT": {"dice": 0.912, "iou": 0.838, "precision": 0.92, "recall": 0.904, "ground_truth_volume_mm3": 5100.0},
        "TC": {"dice": 0.875, "iou": 0.778, "precision": 0.88, "recall": 0.87, "ground_truth_volume_mm3": 3050.0},
        "ET": {"dice": 0.821, "iou": 0.696, "precision": 0.83, "recall": 0.812, "ground_truth_volume_mm3": 1020.0},
    }
    ad_data = {
        "case_name": "GT_Patient",
        "case_id": 11,
        "prediction_id": 56,
        "model_name": "unet_3d",
        "model_display_name": "3D U-Net",
        "checkpoint_epoch": 17,
        "device": "cuda:0",
        "inference_seconds": 2.1,
        "parameter_count": 1191413,
        "modalities": ["FLAIR", "T1", "T1CE", "T2"],
        "volume_dimensions": [155, 240, 240],
        "voxel_spacing_mm": [1.0, 1.0, 1.0],
        "voxel_volume_mm3": 1.0,
        "tumor_regions": {
            "WT": {"voxel_count": 5000, "volume_mm3": 5000.0, "volume_cm3": 5.0},
            "TC": {"voxel_count": 3000, "volume_mm3": 3000.0, "volume_cm3": 3.0},
            "ET": {"voxel_count": 1000, "volume_mm3": 1000.0, "volume_cm3": 1.0},
        },
        "has_ground_truth": True,
        "ground_truth_metrics": gt_metrics,
    }
    payload = {
        "status": "completed",
        "prediction_id": 56,
        "case_id": 11,
        "model_name": "unet_3d",
        "tumor_regions": ad_data["tumor_regions"],
        "ground_truth_metrics": gt_metrics,
        "analysis_details": ad_data,
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(settings, "results_dir", result_root)
    monkeypatch.setattr(
        segmentation,
        "get_prediction",
        lambda pid: SimpleNamespace(
            id=pid,
            case_id=11,
            model_name="unet_3d",
            inference_seconds=2.1,
            mask_path=str(report_path),
            mesh_paths={"report": str(report_path)},
            created_at=SimpleNamespace(isoformat=lambda: "2025-01-01T00:00:00"),
        ),
    )

    resp = client.get("/api/results/56")
    assert resp.status_code == 200
    ad = resp.json()["analysis_details"]
    assert ad["has_ground_truth"] is True
    assert ad["ground_truth_metrics"] is not None
    for key in ("WT", "TC", "ET"):
        m = ad["ground_truth_metrics"][key]
        assert "dice" in m
        assert "iou" in m
        assert "precision" in m
        assert "recall" in m
        assert "ground_truth_volume_mm3" in m


def test_analysis_details_frontend_elements_and_graceful_missing(client: TestClient) -> None:
    """Verify frontend HTML contains all analysis details elements and handles missing details gracefully."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # Section card and headings
    assert 'id="analysis-details-card"' in html
    assert "Analysis details" in html
    assert "Input and geometry" in html
    assert "Predicted tumor-region volumes" in html
    assert "Comparison with uploaded ground truth" in html

    # Section 1 elements
    assert 'id="ad-case"' in html
    assert 'id="ad-case-id"' in html
    assert 'id="ad-prediction-id"' in html
    assert 'id="ad-model"' in html
    assert 'id="ad-epoch"' in html
    assert 'id="ad-device"' in html
    assert 'id="ad-inference-time"' in html
    assert 'id="ad-parameters"' in html

    # Section 2 elements
    assert 'id="ad-modalities"' in html
    assert 'id="ad-dimensions"' in html
    assert 'id="ad-spacing"' in html
    assert 'id="ad-voxel-volume"' in html

    # Section 3 elements
    assert 'id="analysis-regions-table"' in html
    assert 'id="analysis-regions-tbody"' in html

    # Section 4 elements
    assert 'id="analysis-gt-block"' in html
    assert 'id="analysis-gt-table"' in html
    assert 'id="analysis-gt-tbody"' in html

    # Graceful missing check: card is hidden if details is null
    assert "!details" in html
    assert 'analysisCard.style.display = "none"' in html

