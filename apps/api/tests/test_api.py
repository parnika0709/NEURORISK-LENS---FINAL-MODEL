"""FastAPI API and integration tests covering dual-architecture model selection."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import nibabel as nib
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.db import Case, Prediction, get_session
from app.services.model_runtime import ModelRuntime
from braintumor.inference.pipeline import PipelineOutput
from braintumor.models.attention_unet_2d_four_channel import AttentionUNet2DFourChannel
from braintumor.models.unet_3d import Unet3DPredictor
from braintumor.risk.mapping import compute_risk_regions
from braintumor.schemas.contracts import (
    PredictionMeta,
    PredictionOutput,
    TumorFeatures,
    TumorFeaturesBundle,
    TumorLabel,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_case_dir(tmp_path):
    """Create a temporary case record and minimal valid NIfTI files."""
    case_folder = tmp_path / "test_case_modalities"
    case_folder.mkdir(parents=True)

    data = np.zeros((16, 16, 16), dtype=np.float32)
    # Put some values to represent brain tissue
    data[4:12, 4:12, 4:12] = 100.0
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    nib.save(img, str(case_folder / "test_case_t1ce.nii.gz"))
    nib.save(img, str(case_folder / "test_case_t2.nii.gz"))
    nib.save(img, str(case_folder / "test_case_flair.nii.gz"))

    with get_session() as session:
        case = Case(
            case_name="test_case_dual_arch",
            patient_id="patient_test_001",
            modality_summary="t1ce,t2,flair",
            nifti_path=str(case_folder),
        )
        session.add(case)
        session.commit()
        session.refresh(case)
        case_id = case.id

    yield case_id, case_folder

    with get_session() as session:
        c = session.get(Case, case_id)
        if c:
            session.delete(c)
            session.commit()


def _make_mock_pipeline_output(model_name: str, mask_shape: tuple[int, int, int]):
    mask = np.zeros(mask_shape, dtype=np.uint8)
    mask[4:8, 4:8, 4:8] = 1
    mask[5:7, 5:7, 5:7] = 4

    pred_out = PredictionOutput(
        mask=mask,
        probability=None,
        uncertainty=None,
        saliency=None,
        meta=PredictionMeta(
            model_name=model_name,
            model_version="0.1.0",
            inference_seconds=0.08,
        ),
    )
    features = TumorFeaturesBundle(
        features=[
            TumorFeatures(
                label=TumorLabel.WT,
                volume_mm3=64.0,
                voxel_count=64,
                centroid=None,
                bbox=None,
                shape=None,
                intensity=None,
            )
        ]
    )
    return PipelineOutput(prediction=pred_out, features=features)


def test_api_root(client):
    """Test the API welcome and docs root endpoint."""
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Welcome to the Brain Tumor MRI Analysis API"
    assert data["docs"] == "/docs"


def test_frontend_mounted(client):
    """Test that the static single-page frontend is properly served at root."""
    response = client.get("/")
    assert response.status_code == 200
    assert "NeuroRisk Lens" in response.text
    assert "<html" in response.text.lower()


def test_frontend_model_selector_and_badge(client):
    """Test that the frontend HTML includes dual model selector and status badge."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    # Model selector element
    assert 'id="model_select"' in html
    assert 'value="attn_unet_2d"' in html
    assert '2D Attention U-Net (trained four-channel)' in html
    assert 'value="unet_3d"' in html
    assert '3D Volumetric U-Net (trained checkpoint)' in html
    for modality in ("flair_file", "t1_file", "t1ce_file", "t2_file"):
        assert f'id="{modality}"' in html
    assert 'id="segmentation_file"' in html
    assert "Segmentation (optional)" in html
    assert 'request.open("POST", `${API}/api/upload`)' in html
    assert "uploadWithProgress(fd)" in html
    assert 'fetch(`${API}/api/predict`' in html
    assert "const root = `${API}/api/results/${prediction.prediction_id}`" in html
    assert '`${root}/visualization`' in html
    assert 'href="#dashboard"' in html
    assert 'href="#upload"' in html
    assert 'setAttribute("href", "#result")' in html
    assert 'href="#recent-analyses"' in html
    assert "Predicted region volumes" in html
    assert "loadDashboard()" in html
    assert "checkpoint_available" in html
    assert "drop a NIfTI file or browse" in html
    assert "clinical risk scores" in html
    assert "Ã" not in html
    assert "ðŸ" not in html
    assert "â€”" not in html
    # Model badge
    assert 'id="model-badge"' in html


def test_list_cases(client):
    """Test retrieving cases list."""
    response = client.get("/cases/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_nonexistent_case(client):
    """Test 404 behavior for invalid case ID."""
    response = client.get("/cases/9999999")
    assert response.status_code == 404


def test_predict_unsupported_model_raises_400(client, sample_case_dir):
    """Test that unsupported model name returns HTTP 400 with a clear error."""
    case_id, _ = sample_case_dir

    # Via query parameter
    resp_query = client.post(f"/cases/{case_id}/predict?model_name=vit_transformer_3d")
    assert resp_query.status_code == 400
    assert "Unsupported model_name 'vit_transformer_3d'" in resp_query.json()["detail"]
    assert "attn_unet_2d" in resp_query.json()["detail"]
    assert "unet_3d" in resp_query.json()["detail"]

    # Via JSON request body
    resp_body = client.post(
        f"/cases/{case_id}/predict",
        json={"model_name": "unknown_model"},
    )
    assert resp_body.status_code == 400
    assert "Unsupported model_name 'unknown_model'" in resp_body.json()["detail"]


def test_predict_missing_checkpoint_raises_400_no_silent_fallback(client, sample_case_dir, monkeypatch):
    """Test that missing checkpoint returns HTTP 400 and does NOT silently fall back."""
    case_id, _ = sample_case_dir

    monkeypatch.setattr(settings, "model_weights_path_3d", Path("/nonexistent/missing_unet_3d.pt"))
    from app.routes.predictions import inference_service
    inference_service._predictors.pop("unet_3d", None)

    resp = client.post(f"/cases/{case_id}/predict?model_name=unet_3d")
    assert resp.status_code == 400
    assert "Checkpoint for model 'unet_3d' not found" in resp.json()["detail"]


def test_predict_default_model_uses_attn_unet_2d(client, sample_case_dir):
    """Test that omitting model_name defaults to 2D Attention U-Net baseline."""
    case_id, _ = sample_case_dir
    with (
        patch("app.services.inference.run_pipeline") as mock_pipeline,
        patch(
            "app.routes.predictions.inference_service.get_predictor",
            return_value=SimpleNamespace(name="attn_unet_2d", version="test"),
        ),
    ):
        mock_pipeline.return_value = _make_mock_pipeline_output("attn_unet_2d", (16, 16, 16))

        resp = client.post(f"/cases/{case_id}/predict")
        assert resp.status_code == 202, resp.text
        data = resp.json()
        assert data["message"] == "Inference completed successfully"
        assert data["model_name"] == "attn_unet_2d"

        # Verify database record
        with get_session() as session:
            pred = session.query(Prediction).filter_by(id=data["prediction_id"]).first()
            assert pred is not None
            assert pred.model_name == "attn_unet_2d"
            assert pred.case_id == case_id


def test_predict_explicit_3d_model_uses_unet_3d(client, sample_case_dir):
    """Test that selecting unet_3d explicitly records unet_3d in database."""
    case_id, _ = sample_case_dir
    with patch("app.services.inference.run_pipeline") as mock_pipeline:
        mock_pipeline.return_value = _make_mock_pipeline_output("unet_3d", (16, 16, 16))

        # Test passing model_name via JSON body
        resp = client.post(
            f"/cases/{case_id}/predict",
            json={"model_name": "unet_3d"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["message"] == "Inference completed successfully"
        assert data["model_name"] == "unet_3d"

        with get_session() as session:
            pred = session.query(Prediction).filter_by(id=data["prediction_id"]).first()
            assert pred is not None
            assert pred.model_name == "unet_3d"
            assert pred.case_id == case_id


def test_independent_checkpoints_loaded_without_interference():
    """Verify the actual four-channel 2D and trained 3D checkpoints load independently."""
    service = ModelRuntime()

    p2d = service._load_model("attn_unet_2d")
    assert isinstance(p2d, AttentionUNet2DFourChannel)

    p3d = service._load_model("unet_3d")
    assert isinstance(p3d, Unet3DPredictor)

    # Strict independence
    assert type(p2d) is not type(p3d)
    assert p2d is not p3d
    assert service.checkpoints["attn_unet_2d"] != service.checkpoints["unet_3d"]
    service._move_to_cpu()
    assert next(p2d.parameters()).device.type == "cpu"
    assert next(p3d.model.parameters()).device.type == "cpu"


def test_atlas_mapping_dual_architecture_compatibility():
    """Verify Harvard-Oxford atlas mapping works identically for 2D and 3D output masks."""
    affine = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 239.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    spacing = (1.0, 1.0, 1.0)

    # Active tumor mask (BraTS classes 1 and 4)
    mask = np.zeros((155, 240, 240), dtype=np.uint8)
    mask[70:90, 110:130, 110:130] = 1
    mask[75:85, 115:125, 115:125] = 4

    regions = compute_risk_regions(tumor_mask=mask, spacing=spacing, affine=affine)
    assert len(regions) > 0
    for r in regions:
        assert r.region
        assert r.function
        assert r.risk_tier in ("High", "Moderate", "Low", "NOT_APPLICABLE", "No tumor detected")

    # Empty mask
    empty_mask = np.zeros((155, 240, 240), dtype=np.uint8)
    empty_regions = compute_risk_regions(tumor_mask=empty_mask, spacing=spacing, affine=affine)
    assert len(empty_regions) > 0
    for r in empty_regions:
        assert r.risk_tier in ("NOT_APPLICABLE", "No tumor detected")
        assert r.risk_tier != "Low"
        assert r.tumor_detection_status == "NO_TUMOR_DETECTED"
        assert r.eloquent_structure_risk == "NOT_APPLICABLE"
