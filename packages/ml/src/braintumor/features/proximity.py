"""Anatomical Proximity Analysis — exploratory spatial analysis.

PURPOSE
=======
Compute the distance from a predicted tumor surface to nearby anatomical
structures segmented by an atlas-based method. Used to render an
exploratory spatial-analysis panel.

WHAT THIS IS NOT
================
This is NOT a surgical risk predictor. It is NOT clinically validated.
It is NOT a substitute for radiological assessment.

The BraTS dataset does NOT include anatomical structure labels. We use an
atlas-based segmentation (SynthSeg / MONAI WholeBrainSeg) to obtain
approximate structure boundaries, then compute simple Euclidean distances.
Distances are reported in millimeters, but they depend on the accuracy of
the atlas registration, which is not guaranteed across the population.

Output is a list of (structure_name, distance_mm, point_mm) tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

# Anatomical structures of interest for proximity analysis.
# Numbers correspond to SynthSeg label IDs (FreeSurfer color LUT).
# Source: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
STRUCTURES_OF_INTEREST = {
    4: "lateral-ventricle",
    5: "lateral-ventricle",
    10: "thalamus",
    11: "thalamus",
    14: "third-ventricle",
    17: "hippocampus",
    18: "hippocampus",
    24: "csf",
    31: "choroid-plexus",
    43: "lateral-ventricle",
    44: "lateral-ventricle",
    49: "thalamus",
    50: "thalamus",
    51: "thalamus",
    52: "thalamus",
    53: "caudate",
    54: "caudate",
    58: "putamen",
    59: "putamen",
    60: "putamen",
    63: "pallidum",
    64: "pallidum",
}


@dataclass
class ProximityResult:
    structure: str
    distance_mm: float
    closest_point_mm: tuple[float, float, float]
    tumor_point_mm: tuple[float, float, float]


def _structure_mask(atlas: np.ndarray, label_id: int) -> np.ndarray:
    return atlas == label_id


def _surface_points(mask: np.ndarray) -> np.ndarray:
    """Extract surface voxel coordinates from a binary mask."""
    if not mask.any():
        return np.empty((0, 3), dtype=np.int32)
    eroded = ndimage.binary_erosion(mask)
    surface = mask & ~eroded
    return np.argwhere(surface)


def compute_proximity(
    tumor_mask: np.ndarray,
    atlas_segmentation: np.ndarray,
    spacing: tuple[float, float, float],
    k: int = 5,
) -> list[ProximityResult]:
    """Compute distances from tumor surface to each structure.

    Parameters
    ----------
    tumor_mask : (D, H, W) bool/int
        Predicted tumor mask (binary or BraTS-labelled).
    atlas_segmentation : (D, H, W) int
        Atlas-based structure segmentation (SynthSeg labels).
    spacing : (d, h, w) in mm
        Voxel spacing.
    k : int
        How many closest structures to return (sorted by distance).

    Returns
    -------
    list[ProximityResult]
        Sorted by ascending distance.
    """
    tumor_binary = tumor_mask > 0
    tumor_surface = _surface_points(tumor_binary)
    if tumor_surface.size == 0:
        return []

    # Map each tumor surface voxel to physical space (mm).
    spacing_arr = np.asarray(spacing)
    tumor_pts_mm = tumor_surface.astype(np.float64) * spacing_arr

    results: list[ProximityResult] = []

    # Aggregate structures by name (e.g., left/right lateral ventricle).
    by_name: dict[str, np.ndarray] = {}
    for label_id, name in STRUCTURES_OF_INTEREST.items():
        struct_mask = _structure_mask(atlas_segmentation, label_id)
        if not struct_mask.any():
            continue
        surf_pts = _surface_points(struct_mask)
        if surf_pts.size == 0:
            continue
        by_name.setdefault(name, []).append(surf_pts)

    for name, parts in by_name.items():
        struct_pts = np.concatenate(parts, axis=0)
        struct_pts_mm = struct_pts.astype(np.float64) * spacing_arr

        # KD-tree nearest-neighbor in physical space.
        tree = cKDTree(struct_pts_mm)
        dists, idxs = tree.query(tumor_pts_mm, k=1)
        nearest_idx = int(np.argmin(dists))
        nearest_struct_pt = struct_pts_mm[idxs[nearest_idx]]
        nearest_tumor_pt = tumor_pts_mm[nearest_idx]

        results.append(
            ProximityResult(
                structure=name,
                distance_mm=float(dists[nearest_idx]),
                closest_point_mm=(
                    float(nearest_struct_pt[0]),
                    float(nearest_struct_pt[1]),
                    float(nearest_struct_pt[2]),
                ),
                tumor_point_mm=(
                    float(nearest_tumor_pt[0]),
                    float(nearest_tumor_pt[1]),
                    float(nearest_tumor_pt[2]),
                ),
            )
        )

    results.sort(key=lambda r: r.distance_mm)
    return results[:k]


def render_proximity_warning() -> str:
    """Standardized disclaimer text rendered alongside proximity results."""
    return (
        "EXPLORATORY SPATIAL ANALYSIS — NOT A SURGICAL RISK PREDICTOR. "
        "Structure boundaries are estimated via atlas-based segmentation and "
        "may be inaccurate. Do not use for clinical decisions. Consult a "
        "qualified radiologist for any clinical interpretation."
    )
