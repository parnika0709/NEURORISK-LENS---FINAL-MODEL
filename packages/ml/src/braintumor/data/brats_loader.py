"""BraTS dataset loader.

Directory layout expected::

    data/raw/BraTS2021_Training/
        BraTS2021_00001/
            BraTS2021_00001_t1.nii.gz
            BraTS2021_00001_t1ce.nii.gz
            BraTS2021_00001_t2.nii.gz
            BraTS2021_00001_flair.nii.gz
            BraTS2021_00001_seg.nii.gz     # only in training set

Each case is a "patient" — all slices share the same patient_id. Splitting
MUST be done at the case level, not the slice level, to avoid leakage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from braintumor.schemas.contracts import VolumeInput, VolumeModality

# BraTS modality filenames → enum value.
MODALITY_FILENAME = {
    "t1": VolumeModality.T1,
    "t1ce": VolumeModality.T1CE,
    "t2": VolumeModality.T2,
    "flair": VolumeModality.FLAIR,
}

CASE_PATTERN = re.compile(r"^(BraTS\d+).*$")


@dataclass
class BratsCase:
    """One BraTS case directory."""

    case_id: str
    patient_id: str
    root: Path
    modality_paths: dict[VolumeModality, Path]
    seg_path: Path | None

    @classmethod
    def from_dir(cls, case_dir: Path) -> "BratsCase":
        """Discover modalities by filename convention."""
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")

        # Infer patient_id from directory name.
        match = CASE_PATTERN.match(case_dir.name)
        if not match:
            raise ValueError(f"Unexpected BraTS case name: {case_dir.name}")
        # Use the full directory name as patient_id to ensure correct splitting
        # in cases where multiple cases belong to the same patient.
        patient_id = case_dir.name
        case_id = case_dir.name

        modalities: dict[VolumeModality, Path] = {}
        seg_path: Path | None = None

        for path in case_dir.iterdir():
            if not path.is_file():
                continue
            stem = path.name.replace(".nii.gz", "").replace(".nii", "")
            tokens = stem.split("_")
            if len(tokens) < 2:
                continue
            tag = tokens[-1].lower()

            if tag in MODALITY_FILENAME:
                modalities[MODALITY_FILENAME[tag]] = path
            elif tag == "seg":
                seg_path = path

        if not modalities:
            raise ValueError(f"No modality files found in {case_dir}")

        return cls(
            case_id=case_id,
            patient_id=patient_id,
            root=case_dir,
            modality_paths=modalities,
            seg_path=seg_path,
        )

    def has_segmentation(self) -> bool:
        return self.seg_path is not None

    def load_volume(self) -> VolumeInput:
        """Load all available modalities into a (C, D, H, W) array."""
        # Sort by modality enum for deterministic channel order.
        ordered = sorted(self.modality_paths.items(), key=lambda kv: kv[0].value)
        arrays = []
        spacing: tuple[float, float, float] | None = None
        affine: np.ndarray | None = None
        modalities: list[VolumeModality] = []

        for modality, path in ordered:
            img = nib.load(str(path))
            arr = img.get_fdata().astype(np.float32)
            arrays.append(arr)
            if spacing is None:
                spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
            if affine is None:
                affine = img.affine
            modalities.append(modality)

        stacked = np.stack(arrays, axis=0)  # (C, D, H, W)
        assert spacing is not None
        assert affine is not None

        return VolumeInput(
            modalities=modalities,
            array=stacked,
            spacing=spacing,
            affine=affine,
            case_id=self.case_id,
            patient_id=self.patient_id,
        )

    def load_segmentation(self) -> np.ndarray | None:
        """Load segmentation mask, or None if unavailable."""
        if self.seg_path is None:
            return None
        img = nib.load(str(self.seg_path))
        return img.get_fdata().astype(np.uint8)


def discover_cases(root: Path) -> list[BratsCase]:
    """Discover all BraTS cases under a directory."""
    cases: list[BratsCase] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            try:
                cases.append(BratsCase.from_dir(entry))
            except (FileNotFoundError, ValueError):
                continue
    return cases
