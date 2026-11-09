
"""Computational feature extraction from BraTS segmentation masks.

Native BraTS labels:
    0 = Background
    1 = NCR/NET
    2 = Edema
    4 = Enhancing Tumor

Derived regions:
    WT = labels 1, 2, 4
    TC = labels 1, 4
    ET = label 4

Computes per-region:
    - Volume (mm³)
    - Voxel count
    - Centroid (mm)
    - Bounding box (mm)
    - Surface area (mm²)
    - Sphericity
    - Principal axes (mm)
    - T1ce intensity statistics

No clinical interpretation is performed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure

from braintumor.schemas.contracts import (
    BoundingBox,
    Centroid,
    IntensityStats,
    ShapeFeatures,
    TumorFeatures,
    TumorLabel,
    VolumeInput,
    VolumeModality,
)


# Derived overlapping regions from native BraTS labels.
REGION_TO_ENUM = {
    TumorLabel.WT: (1, 2, 4),
    TumorLabel.TC: (1, 4),
    TumorLabel.ET: (4,),
}


def _volume_mm3(
    voxel_count: int,
    spacing: tuple[float, float, float],
) -> float:
    return float(voxel_count * spacing[0] * spacing[1] * spacing[2])


def _centroid_mm(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> Centroid | None:
    if not mask.any():
        return None

    coords = np.argwhere(mask).astype(np.float64)
    weighted = coords.mean(axis=0)

    return Centroid(
        d=float(weighted[0] * spacing[0]),
        h=float(weighted[1] * spacing[1]),
        w=float(weighted[2] * spacing[2]),
    )


def _bbox_mm(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> BoundingBox | None:
    if not mask.any():
        return None

    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    return BoundingBox(
        min_d=float(mins[0] * spacing[0]),
        max_d=float(maxs[0] * spacing[0]),
        min_h=float(mins[1] * spacing[1]),
        max_h=float(maxs[1] * spacing[1]),
        min_w=float(mins[2] * spacing[2]),
        max_w=float(maxs[2] * spacing[2]),
    )


def _principal_axes_mm(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Approximate principal axis lengths using PCA."""
    if mask.sum() < 3:
        return (0.0, 0.0, 0.0)

    coords = np.argwhere(mask).astype(np.float64)
    coords *= np.asarray(spacing)

    coords -= coords.mean(axis=0)

    cov = np.cov(coords, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)

    return tuple(
        float(4.0 * np.sqrt(max(ev, 0.0)))
        for ev in sorted(eigenvalues, reverse=True)
    )


def _surface_area_mm2(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Calculate surface area using marching cubes."""
    if mask.sum() < 6:
        return 0.0

    try:
        verts, faces, _normals, _values = measure.marching_cubes(
            mask.astype(np.float32),
            level=0.5,
            spacing=spacing,
        )

        if len(faces) == 0:
            return 0.0

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        area = 0.5 * np.linalg.norm(
            np.cross(v1 - v0, v2 - v0),
            axis=1,
        )

        return float(area.sum())

    except (ValueError, RuntimeError):
        return 0.0


def _sphericity(
    volume_mm3: float,
    surface_area_mm2: float,
) -> float:
    """Calculate sphericity. A perfect sphere has value 1."""
    if volume_mm3 <= 0 or surface_area_mm2 <= 0:
        return 0.0

    return float(
        (np.pi ** (1.0 / 3.0))
        * ((6.0 * volume_mm3) ** (2.0 / 3.0))
        / surface_area_mm2
    )


def _intensity_stats(
    mask: np.ndarray,
    modality: np.ndarray | None,
) -> IntensityStats | None:
    """Calculate intensity statistics within a matching 3D region."""
    if modality is None or not mask.any():
        return None

    modality = np.asarray(modality)

    if modality.shape != mask.shape:
        raise ValueError(
            "Mask and modality dimensions do not match: "
            f"mask shape={mask.shape}, "
            f"modality shape={modality.shape}. "
            "Check prediction and input volume orientation."
        )

    vals = modality[mask]

    if vals.size == 0:
        return None

    return IntensityStats(
        mean=float(vals.mean()),
        std=float(vals.std()),
        median=float(np.median(vals)),
        min=float(vals.min()),
        max=float(vals.max()),
    )


def _select_modality(
    volume: VolumeInput,
    target: VolumeModality,
) -> np.ndarray | None:
    """Select a modality from a (C,D,H,W) input volume."""
    if target not in volume.modalities:
        return None

    arr = np.asarray(volume.array)

    if arr.ndim != 4:
        raise ValueError(
            f"Expected volume shape (C,D,H,W), got {arr.shape}"
        )

    if arr.shape[0] != len(volume.modalities):
        raise ValueError(
            "Volume channel count does not match modality list: "
            f"channels={arr.shape[0]}, "
            f"modalities={len(volume.modalities)}"
        )

    idx = volume.modalities.index(target)

    return arr[idx]


def extract_features(
    mask: np.ndarray,
    volume: VolumeInput,
) -> "TumorFeaturesBundle":
    """Compute the full feature bundle for a segmentation prediction."""
    from braintumor.schemas.contracts import TumorFeaturesBundle

    mask = np.asarray(mask)

    if mask.ndim != 3:
        raise ValueError(
            f"Expected a 3D native segmentation mask, got {mask.shape}"
        )

    valid_labels = (0, 1, 2, 4)

    if not np.isin(mask, valid_labels).all():
        raise ValueError(
            f"Unexpected segmentation labels: {np.unique(mask).tolist()}"
        )

    spacing = volume.spacing
    t1ce = _select_modality(volume, VolumeModality.T1CE)

    if t1ce is not None and t1ce.shape != mask.shape:
        raise ValueError(
            "Prediction mask and input volume dimensions do not match: "
            f"mask shape={mask.shape}, "
            f"volume shape={t1ce.shape}. "
            "Check inference preprocessing and volume orientation."
        )

    features: list[TumorFeatures] = []

    for tumor_enum, native_labels in REGION_TO_ENUM.items():

        # Derive WT, TC and ET from native BraTS tissue labels.
        binary = np.isin(mask, native_labels)

        voxel_count = int(binary.sum())

        if voxel_count == 0:
            continue

        volume_mm3 = _volume_mm3(voxel_count, spacing)
        bbox = _bbox_mm(binary, spacing)
        centroid = _centroid_mm(binary, spacing)
        axes = _principal_axes_mm(binary, spacing)
        surface = _surface_area_mm2(binary, spacing)
        sphericity = _sphericity(volume_mm3, surface)
        intensity = _intensity_stats(binary, t1ce)

        features.append(
            TumorFeatures(
                label=tumor_enum,
                volume_mm3=volume_mm3,
                voxel_count=voxel_count,
                centroid=centroid,
                bbox=bbox,
                shape=ShapeFeatures(
                    surface_area_mm2=surface,
                    sphericity=sphericity,
                    principal_axes_mm=axes,
                ),
                intensity=intensity,
            )
        )

    return TumorFeaturesBundle(features=features)
