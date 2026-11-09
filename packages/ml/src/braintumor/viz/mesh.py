
"""3D mesh generation using PyVista/VTK.

Produces GLB-format meshes for the frontend @react-three/fiber viewer.

BraTS 2021 native labels:
    0 = Background
    1 = NCR/NET
    2 = Edema
    4 = Enhancing Tumor

Derived visualization regions:
    WT = native labels 1, 2, 4
    TC = native labels 1, 4
    ET = native label 4
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


# Region identifiers retained for frontend compatibility.
REGION_WT = 1
REGION_TC = 2
REGION_ET = 4

# Colors per derived region (RGB, 0-1).
REGION_COLORS = {
    REGION_WT: (0.93, 0.49, 0.20),
    REGION_TC: (0.85, 0.10, 0.10),
    REGION_ET: (0.55, 0.00, 0.55),
}

# Derived regions from native BraTS labels.
REGION_NATIVE_LABELS = {
    REGION_WT: (1, 2, 4),
    REGION_TC: (1, 4),
    REGION_ET: (4,),
}


@dataclass
class MeshResult:
    """Mesh file paths produced by generate_meshes."""

    brain_path: Path
    tumor_paths: dict[int, Path]


def _smooth(
    mesh: pv.PolyData,
    iterations: int = 50,
) -> pv.PolyData:
    """Light smoothing while preserving anatomical detail."""
    try:
        return mesh.smooth(
            n_iter=iterations,
            relaxation_factor=0.1,
        )
    except Exception:
        return mesh


def _marching_mesh(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    level: float = 0.5,
    smoothing: bool = True,
) -> pv.PolyData | None:
    """Generate a surface mesh from a binary 3D volume."""
    if not volume.any():
        return None

    try:
        grid = pv.wrap(volume.astype(np.float32))
        grid.spacing = spacing

        scalars = mesh_or_array(grid)
        mesh = grid.contour(
            isosurfaces=[level],
            scalars="values",
        )

        if mesh.n_points == 0:
            return None

        return _smooth(mesh) if smoothing else mesh

    except Exception:
        return None


def mesh_or_array(grid: pv.DataSet) -> np.ndarray:
    """Ensure the PyVista dataset has a named scalar array."""
    if "values" not in grid.array_names:
        grid["values"] = np.asarray(grid).flatten(order="F")

    return grid["values"]


def generate_tumor_meshes(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    output_dir: Path,
    regions: tuple[int, ...] = (
        REGION_WT,
        REGION_TC,
        REGION_ET,
    ),
) -> dict[int, Path]:
    """Generate one mesh per requested derived BraTS region.

    Returns:
        Mapping of region identifier to generated mesh path.
        WT, TC and ET use derived masks, not native label equality.
    """
    mask = np.asarray(mask)

    if mask.ndim != 3:
        raise ValueError(
            f"Expected a 3D segmentation mask, got {mask.shape}"
        )

    if not np.isin(mask, [0, 1, 2, 4]).all():
        raise ValueError(
            f"Unexpected BraTS labels: {np.unique(mask).tolist()}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    out: dict[int, Path] = {}

    for region in regions:
        if region not in REGION_NATIVE_LABELS:
            raise ValueError(
                f"Unknown derived region identifier: {region}"
            )

        native_labels = REGION_NATIVE_LABELS[region]

        binary = np.isin(mask, native_labels).astype(np.uint8)

        mesh = _marching_mesh(
            binary,
            spacing=spacing,
            level=0.5,
        )

        if mesh is None:
            continue

        mesh.translate(
            -np.asarray(mesh.center),
            inplace=True,
        )

        color = REGION_COLORS.get(
            region,
            (0.5, 0.5, 0.5),
        )

        path = output_dir / f"tumor_region_{region}.glb"

        mesh.point_data["color"] = np.tile(
            np.asarray(color),
            (mesh.n_points, 1),
        )

        try:
            mesh.save(str(path))
            out[region] = path

        except Exception:
            fallback = path.with_suffix(".ply")
            mesh.save(str(fallback))
            out[region] = fallback

    return out


def generate_brain_mesh(
    modality: np.ndarray,
    spacing: tuple[float, float, float],
    output_dir: Path,
    intensity_percentile: float = 30.0,
) -> Path | None:
    """Generate a brain surface mesh from a structural MRI modality."""
    modality = np.asarray(modality)

    if modality.ndim != 3:
        raise ValueError(
            f"Expected a 3D MRI modality, got {modality.shape}"
        )

    positive = modality[modality > 0]

    if positive.size == 0:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(
        np.percentile(positive, intensity_percentile)
    )

    binary = (modality > threshold).astype(np.uint8)

    from scipy import ndimage

    binary = ndimage.binary_fill_holes(binary).astype(np.uint8)

    mesh = _marching_mesh(
        binary,
        spacing=spacing,
        level=0.5,
    )

    if mesh is None:
        return None

    mesh.translate(
        -np.asarray(mesh.center),
        inplace=True,
    )

    mesh.point_data["color"] = np.tile(
        np.array([0.85, 0.85, 0.92]),
        (mesh.n_points, 1),
    )

    path = output_dir / "brain_surface.glb"

    try:
        mesh.save(str(path))
        return path

    except Exception:
        fallback = path.with_suffix(".ply")
        mesh.save(str(fallback))
        return fallback


def generate_meshes(
    mask: np.ndarray,
    modality: np.ndarray,
    spacing: tuple[float, float, float],
    output_dir: Path,
) -> MeshResult:
    """Generate brain and tumor meshes for a case."""
    brain_path = generate_brain_mesh(
        modality,
        spacing=spacing,
        output_dir=output_dir,
    )

    tumor_paths = generate_tumor_meshes(
        mask,
        spacing=spacing,
        output_dir=output_dir,
    )

    return MeshResult(
        brain_path=brain_path if brain_path else Path(""),
        tumor_paths=tumor_paths,
    )
