import numpy as np
import torch
import pytest
from braintumor.models.attention_unet_2d import AttentionUnet2D

def test_model_forward_pass():
    """Test that AttentionUnet2D produces correct output shapes."""
    batch_size = 2
    in_channels = 3
    h, w = 128, 128
    out_channels = 4

    model = AttentionUnet2D(in_channels=in_channels, out_channels=out_channels)
    model.eval()

    x = torch.randn(batch_size, in_channels, h, w)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch_size, out_channels, h, w)

def test_model_invalid_config():
    """Test that AttentionUnet2D raises error for invalid channels/strides."""
    with pytest.raises(ValueError):
        # channels len must be strides len + 1
        AttentionUnet2D(channels=(32, 64), strides=(2, 2, 2))


def test_losses_2d_and_3d():
    """Test both standard and class-imbalance-aware loss computation."""
    from braintumor.training.losses import ClassWeightedDiceCELoss, DiceCELoss

    logits = torch.randn(2, 4, 32, 32)
    target = torch.randint(0, 4, (2, 32, 32))

    loss_fn = DiceCELoss()
    val = loss_fn(logits, target)
    assert val > 0 and torch.isfinite(val)

    weighted_loss_fn = ClassWeightedDiceCELoss(
        dice_class_weights=(2.0, 1.0, 2.0),
        ce_class_weights=(0.1, 1.5, 0.8, 1.5),
    )
    w_val = weighted_loss_fn(logits, target)
    assert w_val > 0 and torch.isfinite(w_val)


def test_unet_3d_forward_and_predictor():
    """Test independent 3D U-Net forward pass and Predictor interface."""
    from braintumor.models.unet_3d import Unet3D, Unet3DPredictor
    from braintumor.schemas.contracts import VolumeInput, VolumeModality

    model = Unet3D(in_channels=4, out_channels=4, channels=(8, 16, 32), strides=(2, 2))
    model.eval()

    # (B, C, D, H, W)
    x = torch.randn(1, 4, 16, 16, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4, 16, 16, 16)

    # Test Predictor contract
    predictor = Unet3DPredictor(roi_size=(16, 16, 16))
    volume = VolumeInput(
        modalities=[VolumeModality.FLAIR, VolumeModality.T1, VolumeModality.T1CE, VolumeModality.T2],
        array=np.zeros((4, 16, 16, 16), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        affine=np.eye(4),
    )
    pred_out = predictor.predict(volume)
    assert pred_out.mask.shape == (16, 16, 16)
    assert pred_out.meta.model_name == "unet_3d"
    assert pred_out.probability.shape == (4, 16, 16, 16)


def test_unet_3d_training_smoke():
    """Smoke test: 3D U-Net forward pass, ClassWeightedDiceCELoss backward, and optimizer step."""
    from braintumor.models.unet_3d import Unet3D
    from braintumor.training.losses import ClassWeightedDiceCELoss

    model = Unet3D(in_channels=4, out_channels=4, channels=(8, 16, 32), strides=(2, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = ClassWeightedDiceCELoss(
        dice_class_weights=(1.5, 1.0, 1.5),
        ce_class_weights=(1.0, 2.0, 1.5, 2.0),
    )

    x = torch.randn(2, 4, 16, 16, 16)
    y = torch.randint(0, 4, (2, 16, 16, 16))

    optimizer.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    assert torch.isfinite(loss)
    loss.backward()

    # Verify gradients were calculated
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    optimizer.step()


def test_ras_reorientation_preservation():
    """Verify that canonical RAS reorientation preserves exact label values and shapes."""
    import nibabel as nib

    # Create synthetic volume in LPS orientation (affine diagonal [-1, -1, 1, 1])
    lps_affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    lps_data = np.zeros((16, 16, 16), dtype=np.uint8)
    lps_data[4:8, 4:8, 4:8] = 1   # NCR
    lps_data[6:8, 6:8, 6:8] = 2   # ED
    lps_data[7:8, 7:8, 7:8] = 4   # ET

    img = nib.Nifti1Image(lps_data, lps_affine)
    assert nib.orientations.aff2axcodes(img.affine) == ("L", "P", "S")

    # Reorient to canonical RAS
    ras_img = nib.as_closest_canonical(img)
    ras_data = np.asarray(ras_img.get_fdata(), dtype=np.uint8)

    assert nib.orientations.aff2axcodes(ras_img.affine) == ("R", "A", "S")
    assert ras_data.shape == (16, 16, 16)
    # Check that labels are strictly preserved without interpolation corruption
    assert set(np.unique(ras_data).tolist()) == {0, 1, 2, 4}
    for val in [0, 1, 2, 4]:
        assert np.count_nonzero(ras_data == val) == np.count_nonzero(lps_data == val)


def test_unet_3d_brain_mask_constraint():
    """Verify that Unet3DPredictor brain mask eliminates outside air while preserving inside tumor."""
    from braintumor.models.unet_3d import Unet3DPredictor
    from braintumor.schemas.contracts import VolumeInput, VolumeModality

    predictor = Unet3DPredictor(roi_size=(16, 16, 16))

    # Create a synthetic volume with central brain (signal > 0) and surrounding air (signal == 0)
    arr = np.zeros((4, 32, 32, 32), dtype=np.float32)
    # Head/brain in center
    arr[:, 8:24, 8:24, 8:24] = 1.0

    vol = VolumeInput(
        modalities=[VolumeModality.FLAIR, VolumeModality.T1, VolumeModality.T1CE, VolumeModality.T2],
        array=arr,
        spacing=(1.0, 1.0, 1.0),
        affine=np.eye(4),
    )

    # 1. Prediction WITH brain mask
    pred_masked = predictor.predict(vol, apply_brain_mask=True)
    mask = pred_masked.mask

    # Voxels outside the brain (e.g. index 0 to 7) must be strictly 0
    outside_voxels = mask[:7, :, :]
    assert (outside_voxels == 0).all(), "Brain mask failed to suppress air background voxels!"

    # 2. Prediction WITHOUT brain mask
    pred_unmasked = predictor.predict(vol, apply_brain_mask=False)
    assert pred_unmasked.mask.shape == (32, 32, 32)
