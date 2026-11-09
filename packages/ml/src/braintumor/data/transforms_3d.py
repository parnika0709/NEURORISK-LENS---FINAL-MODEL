"""Deterministic shared preprocessing for 3D BraTS training and inference."""

from __future__ import annotations

import numpy as np
import torch
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)

from braintumor.data.preprocessing import brats_label_to_4class


class ConvertBraTSLabelsd:
    """Map native BraTS labels {0, 1, 2, 4} to contiguous classes {0, 1, 2, 3}."""

    def __init__(self, keys: tuple[str, ...] = ("label",)) -> None:
        self.keys = keys

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        for key in self.keys:
            if key in result:
                result[key] = brats_label_to_4class(np.asarray(result[key]))
        return result


def build_3d_volume_transform(include_label: bool = True) -> Compose:
    """Load, align, normalize, and type MRI volumes using the training pipeline."""
    keys = ["image", "label"] if include_label else ["image"]
    modes = ("bilinear", "nearest") if include_label else "bilinear"
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=modes),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1.0,
            upper=99.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]
    if include_label:
        transforms.append(ConvertBraTSLabelsd())
    transforms.extend(
        [
            EnsureTyped(keys=keys),
            CastToTyped(keys=["image"], dtype=torch.float32),
        ]
    )
    if include_label:
        transforms.append(CastToTyped(keys=["label"], dtype=torch.long))
    return Compose(transforms)
