
"""Slice exporters for the 2D viewer.

Converts (D, H, W) arrays into PNG slices at axial/coronal/sagittal planes.
Used by the backend to serve preview images and by the report generator.

BraTS native segmentation labels:
    0 = background
    1 = NCR/NET
    2 = edema
    4 = enhancing tumor

Displayed region overlays are derived:
    WT = {1, 2, 4}
    TC = {1, 4}
    ET = {4}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


# Native BraTS labels used to derive each displayed tumor region.
REGION_NATIVE_LABELS = {
    "WT": (1, 2, 4),
    "TC": (1, 4),
    "ET": (4,),
}

# Display colors: WT orange, TC red, ET purple.
REGION_COLORS = {
    "WT": np.array([255, 165, 60], dtype=np.float32),
    "TC": np.array([220, 30, 30], dtype=np.float32),
    "ET": np.array([140, 0, 140], dtype=np.float32),
}


def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    """Robust percentile-based normalization to uint8."""
    arr = np.asarray(arr, dtype=np.float32)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    lo, hi = np.percentile(finite, [1, 99])

    if hi - lo < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)

    safe = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
    out = np.clip((safe - lo) / (hi - lo), 0.0, 1.0)

    return (out * 255).astype(np.uint8)


def _validate_volume(volume: np.ndarray) -> np.ndarray:
    """Validate a four-dimensional (C, D, H, W) MRI volume."""
    volume = np.asarray(volume)

    if volume.ndim != 4:
        raise ValueError(
            f"Expected volume shape (C, D, H, W), got {volume.shape}"
        )

    if any(size <= 0 for size in volume.shape):
        raise ValueError(f"Volume contains an empty dimension: {volume.shape}")

    return volume


def _validate_mask(
    mask: np.ndarray | None,
    expected_shape: tuple[int, int, int],
) -> np.ndarray | None:
    """Validate a native BraTS segmentation mask."""
    if mask is None:
        return None

    mask = np.asarray(mask)

    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match volume spatial shape "
            f"{expected_shape}"
        )

    valid = np.isin(mask, [0, 1, 2, 4])
    if not np.all(valid):
        invalid = np.unique(mask[~valid])
        raise ValueError(
            f"Unexpected BraTS segmentation labels: {invalid}"
        )

    return mask


def _apply_region_overlays(
    rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay derived WT, TC and ET regions on an RGB volume.

    More specific regions are applied last so ET remains visible over TC/WT.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    for region_name in ("WT", "TC", "ET"):
        native_labels = REGION_NATIVE_LABELS[region_name]
        color = REGION_COLORS[region_name]

        region_mask = np.isin(mask, native_labels)

        if region_mask.any():
            rgb[region_mask] = (
                (1.0 - alpha) * rgb[region_mask].astype(np.float32)
                + alpha * color
            ).astype(np.uint8)

    return rgb


def export_slice_pngs(
    volume: np.ndarray,
    mask: np.ndarray | None,
    output_dir: Path,
    *,
    indices: list[int] | None = None,
    modality_index: int = 0,
) -> list[Path]:
    """Export axial PNG slices with optional native BraTS region overlays.

    Args:
        volume: MRI volume with shape (C, D, H, W).
        mask: Native BraTS segmentation with shape (D, H, W), or None.
        output_dir: Directory where PNG files are saved.
        indices: Axial slice indices. Defaults to eight evenly spaced slices.
        modality_index: Channel index to display.

    Returns:
        Paths of the generated PNG files.
    """
    volume = _validate_volume(volume)

    channels, depth, height, width = volume.shape

    if not 0 <= modality_index < channels:
        raise ValueError(
            f"modality_index must be between 0 and {channels - 1}; "
            f"got {modality_index}"
        )

    mask = _validate_mask(mask, (depth, height, width))

    if indices is None:
        indices = np.linspace(
            0,
            depth - 1,
            num=min(8, depth),
            dtype=int,
        ).tolist()

    indices = list(indices)

    for index in indices:
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"Slice index must be an integer, got {index!r}")

        if not 0 <= index < depth:
            raise IndexError(
                f"Slice index {index} is outside valid range 0 to {depth - 1}"
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = volume[modality_index]
    base_norm = normalize_uint8(base)

    # Convert grayscale MRI volume to RGB.
    rgb = np.stack([base_norm, base_norm, base_norm], axis=-1)

    if mask is not None:
        rgb = _apply_region_overlays(rgb, mask)

    out_paths: list[Path] = []

    for index in indices:
        slice_rgb = rgb[index]

        img = Image.fromarray(slice_rgb)
        path = output_dir / f"slice_axial_{index:04d}.png"

        img.save(path)
        out_paths.append(path)

    return out_paths
