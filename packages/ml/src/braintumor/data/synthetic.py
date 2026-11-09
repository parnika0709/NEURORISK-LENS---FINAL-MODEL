"""Synthetic data generator for pipeline verification.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import nibabel as nib

from braintumor.schemas.contracts import VolumeModality

def create_synthetic_case(case_id: str, root: Path) -> Path:
    """Create a synthetic BraTS case directory with fake NIfTI files.

    Structure:
      - t1ce.nii.gz
      - t2.nii.gz
      - flair.nii.gz
      - seg.nii.gz
    """
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # Volume dimensions (C, D, H, W)
    shape = (128, 128, 128)
    affine = np.eye(4)

    # Modalities
    modalities = ["t1ce", "t2", "flair"]
    for mod in modalities:
        data = np.random.randn(*shape).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(case_dir / f"{case_id}_{mod}.nii.gz"))

    # Segmentation: 0=bg, 1=WT, 2=TC, 4=ET
    seg = np.zeros(shape, dtype=np.uint8)
    # Create a random blob
    z, y, x = np.random.randint(20, 100, size=3)
    seg[z-10:z+10, y-10:y+10, x-10:x+10] = 1
    seg[z-5:z+5, y-5:y+5, x-5:x+5] = 2
    seg[z-2:z+2, y-2:y+2, x-2:x+2] = 4

    seg_img = nib.Nifti1Image(seg, affine)
    nib.save(seg_img, str(case_dir / f"{case_id}_seg.nii.gz"))

    return case_dir

def setup_synthetic_dataset(root: Path, n_cases: int = 10) -> None:
    """Setup a small synthetic dataset for verification."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_cases):
        create_synthetic_case(f"BraTS2021_{i:05d}_001", root)
