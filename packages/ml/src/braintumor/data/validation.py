"""Dataset and input validation for BraTS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from braintumor.data.brats_loader import BratsCase, discover_cases
from braintumor.schemas.contracts import VolumeInput

def validate_brats_case(case: BratsCase) -> None:
    """Validate that a BratsCase has all required modalities and files exist."""
    for modality, path in case.modality_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Modality file missing: {path}")

    if case.seg_path and not case.seg_path.exists():
        raise FileNotFoundError(f"Segmentation file missing: {case.seg_path}")

def validate_volume_input(volume: VolumeInput) -> None:
    """Validate the shape and type of VolumeInput.array."""
    arr = volume.array
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr)}")

    if arr.ndim != 4:
        raise ValueError(f"Expected array shape (C, D, H, W), got {arr.shape}")

    if len(volume.modalities) != arr.shape[0]:
        raise ValueError(
            f"Number of modalities ({len(volume.modalities)}) "
            f"must match number of channels ({arr.shape[0]})"
        )

def check_dataset_integrity(root: Path) -> dict[str, Any]:
    """Scan dataset root and report missing files or inconsistent cases."""
    cases = discover_cases(root)
    errors = []

    for case in cases:
        try:
            validate_brats_case(case)
        except Exception as e:
            errors.append({"case": case.case_id, "error": str(e)})

    return {
        "total_cases": len(cases),
        "valid_cases": len(cases) - len(errors),
        "errors": errors,
    }
