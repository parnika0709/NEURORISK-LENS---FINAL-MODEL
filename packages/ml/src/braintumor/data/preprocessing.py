"""Preprocessing transforms for BraTS volumes.

Training uses augmentation; validation and testing are deterministic.
BraTS native labels are preserved as four contiguous training classes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)

from braintumor.schemas.contracts import VolumeModality

KEYS = ["image", "label"]
IMAGE_KEYS = ["image"]


def _select_keys(modalities: list[VolumeModality]) -> list[str]:
    """Map VolumeModality to the string keys used by MONAI."""
    return [m.value for m in modalities]


def _build_base_transforms(
    modalities: list[VolumeModality],
    keys: list[str],
) -> list:
    """Create the common loading and spatial preprocessing steps."""
    img_keys = _select_keys(modalities)
    all_keys = img_keys + (["label"] if "label" in keys else [])

    return [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(
            keys=all_keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear",) * len(img_keys)
            + (("nearest",) if "label" in keys else ()),
        ),
        ScaleIntensityRangePercentilesd(
            keys=img_keys,
            lower=1.0,
            upper=99.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]


def train_transforms(
    modalities: list[VolumeModality],
    keys: list[str] = KEYS,
) -> Callable:
    """Augmented training pipeline."""
    img_keys = _select_keys(modalities)
    all_keys = img_keys + (["label"] if "label" in keys else [])

    transforms = _build_base_transforms(modalities, keys)
    transforms.extend([
        RandFlipd(
            keys=all_keys,
            prob=0.5,
            spatial_axis=0,
        ),
        RandRotate90d(
            keys=all_keys,
            prob=0.5,
            spatial_axes=(0, 1),
        ),
        RandGaussianNoised(
            keys=img_keys,
            prob=0.15,
            mean=0.0,
            std=0.1,
        ),
        RandShiftIntensityd(
            keys=img_keys,
            offsets=0.1,
            prob=0.5,
        ),
        RandScaleIntensityd(
            keys=img_keys,
            factors=0.1,
            prob=0.5,
        ),
        RandAdjustContrastd(
            keys=img_keys,
            prob=0.3,
            gamma=(0.7, 1.5),
        ),
        NormalizeIntensityd(
            keys=img_keys,
            nonzero=True,
            channel_wise=True,
        ),
    ])

    return Compose(transforms)


def val_transforms(
    modalities: list[VolumeModality],
    keys: list[str] = KEYS,
) -> Callable:
    """Deterministic validation and testing pipeline."""
    img_keys = _select_keys(modalities)

    transforms = _build_base_transforms(modalities, keys)
    transforms.append(
        NormalizeIntensityd(
            keys=img_keys,
            nonzero=True,
            channel_wise=True,
        )
    )

    return Compose(transforms)


def brats_label_to_4class(label: np.ndarray) -> np.ndarray:
    """Convert native BraTS labels {0, 1, 2, 4} to training classes {0, 1, 2, 3}.

    0 = background
    1 = NCR/NET
    2 = edema
    3 = enhancing tumor
    """
    label = np.asarray(label)

    valid_labels = np.isin(label, [0, 1, 2, 4])
    if not np.all(valid_labels):
        invalid = np.unique(label[~valid_labels])
        raise ValueError(f"Unexpected BraTS segmentation labels: {invalid}")

    out = np.zeros(label.shape, dtype=np.uint8)
    out[label == 1] = 1
    out[label == 2] = 2
    out[label == 4] = 3

    return out


def fourclass_to_brats(label: np.ndarray) -> np.ndarray:
    """Convert training classes {0, 1, 2, 3} to native BraTS labels {0, 1, 2, 4}."""
    label = np.asarray(label)

    valid_labels = np.isin(label, [0, 1, 2, 3])
    if not np.all(valid_labels):
        invalid = np.unique(label[~valid_labels])
        raise ValueError(f"Unexpected training class labels: {invalid}")

    out = np.zeros(label.shape, dtype=np.uint8)
    out[label == 1] = 1
    out[label == 2] = 2
    out[label == 3] = 4

    return out