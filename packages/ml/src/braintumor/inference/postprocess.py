
"""Post-processing of BraTS 2021 native segmentation labels.

Native labels:
    0 = Background
    1 = NCR/NET
    2 = Edema
    4 = Enhancing Tumor

Each tissue class is cleaned independently.
WT, TC and ET are derived regions, not native voxel labels.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


BRATS_NATIVE_LABELS = (1, 2, 4)
BRATS_BACKGROUND = 0


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    """Validate a native BraTS segmentation mask."""
    mask = np.asarray(mask)

    if mask.ndim != 3:
        raise ValueError(
            f"Expected a 3D segmentation mask, got shape {mask.shape}"
        )

    valid_labels = (0, 1, 2, 4)

    if not np.isin(mask, valid_labels).all():
        invalid = np.unique(mask[~np.isin(mask, valid_labels)])
        raise ValueError(
            f"Invalid BraTS labels found: {invalid.tolist()}"
        )

    return mask


def remove_small_components(
    mask: np.ndarray,
    label: int,
    min_voxels: int = 10,
) -> np.ndarray:
    """Remove small connected components of one native tissue class."""
    mask = _validate_mask(mask)

    if label not in BRATS_NATIVE_LABELS:
        raise ValueError(f"Invalid native BraTS label: {label}")

    if min_voxels < 0:
        raise ValueError("min_voxels must be non-negative")

    binary = mask == label

    if not binary.any() or min_voxels <= 1:
        return mask.copy()

    structure = ndimage.generate_binary_structure(3, 1)
    labeled, component_count = ndimage.label(
        binary,
        structure=structure,
    )

    if component_count == 0:
        return mask.copy()

    component_sizes = np.bincount(labeled.ravel())
    small_components = np.where(
        (component_sizes < min_voxels)
        & (np.arange(len(component_sizes)) != 0)
    )[0]

    out = mask.copy()

    if small_components.size:
        remove_mask = np.isin(labeled, small_components)
        out[remove_mask] = BRATS_BACKGROUND

    return out


def postprocess(
    mask: np.ndarray,
    min_voxels: int = 10,
) -> np.ndarray:
    """Remove tiny isolated components while preserving native labels.

    No WT/TC hierarchy is imposed because the input contains
    mutually exclusive native BraTS tissue labels.
    """
    out = _validate_mask(mask).copy()

    for label in BRATS_NATIVE_LABELS:
        out = remove_small_components(
            out,
            label=label,
            min_voxels=min_voxels,
        )

    return out
