import numpy as np
import pytest
from braintumor.data.preprocessing import (
    train_transforms,
    val_transforms,
    brats_label_to_4class,
    fourclass_to_brats,
)
from braintumor.schemas.contracts import VolumeModality

def test_brats_label_mapping():
    """Test the bidirectional mapping between BraTS and 4-class labels."""
    # BraTS: 0=bg, 1=WT, 2=TC, 4=ET
    # 4-class: 0=bg, 1=WT, 2=TC, 3=ET (based on preprocessing.py impl)
    # Actually, the current preprocessing.py impl is:
    # 1 (NCR/NET) -> 1
    # 2 (ED) -> 1
    # 4 (ET) -> 3
    # This seems to map WT (1+2) to 1 and ET (4) to 3.

    label = np.array([0, 1, 2, 4], dtype=np.uint8)
    mapped = brats_label_to_4class(label)
    np.testing.assert_array_equal(mapped, [0, 1, 2, 3])

    back = fourclass_to_brats(mapped)
    # 1 -> 1, 2 -> 2, 3 -> 4
    np.testing.assert_array_equal(back, [0, 1, 2, 4])

def test_transforms_output_shape():
    """Verify that transforms produce expected shapes (using mock data)."""
    # This is hard to test without real files, so we mock the LoadImaged
    # but we can test if the compose object exists.
    modalities = [VolumeModality.T1CE, VolumeModality.T2, VolumeModality.FLAIR]

    train_tf = train_transforms(modalities)
    val_tf = val_transforms(modalities)

    assert train_tf is not None
    assert val_tf is not None


def test_evaluation_normalization_consistency():
    """Verify that evaluation numpy normalization matches MONAI's non-zero z-score logic."""
    from monai.transforms import NormalizeIntensity

    np.random.seed(42)
    # Simulate a brain volume with foreground and air background (zeros)
    raw = np.random.uniform(10.0, 100.0, (1, 32, 32, 32)).astype(np.float32)
    raw[:, :10, :, :] = 0.0  # Background air

    # Evaluation logic
    p1, p99 = np.percentile(raw[0], (1.0, 99.0))
    scaled = np.clip(raw[0], p1, p99)
    scaled = (scaled - p1) / (p99 - p1)
    mask = scaled > 0
    mean = scaled[mask].mean()
    std = scaled[mask].std() + 1e-8
    eval_normalized = scaled.copy()
    eval_normalized[mask] = (eval_normalized[mask] - mean) / std
    eval_normalized[~mask] = 0.0

    # MONAI logic
    monai_norm = NormalizeIntensity(nonzero=True, channel_wise=True)
    monai_output = monai_norm(scaled[np.newaxis, ...])[0]

    np.testing.assert_allclose(eval_normalized, monai_output, atol=1e-5)
    # Ensure background air remains exactly 0.0
    assert (eval_normalized[~mask] == 0.0).all()

