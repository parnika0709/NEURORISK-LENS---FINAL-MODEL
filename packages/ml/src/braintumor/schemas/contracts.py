
"""Stable schemas for the Predictor interface.

These Pydantic models define the contract between the ML layer and the
backend/frontend. Changes here are visible across the stack — keep them
additive and versioned via SCHEMA_VERSION.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Tumor label structure
# ---------------------------------------------------------------------------


class TumorLabel(str, enum.Enum):
    """Derived BraTS tumor regions.

    BraTS 2021 native segmentation labels:
      - 0 = BACKGROUND
      - 1 = NCR/NET (necrotic and non-enhancing tumor)
      - 2 = EDEMA
      - 4 = ENHANCING TUMOR

    Derived overlapping regions:
      - WT (WHOLE TUMOR) = native labels 1, 2, 4
      - TC (TUMOR CORE) = native labels 1, 4
      - ET (ENHANCING TUMOR) = native label 4
    """

    BACKGROUND = "background"
    WT = "whole_tumor"
    TC = "tumor_core"
    ET = "enhancing_tumor"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class VolumeModality(str, enum.Enum):
    """Imaging modalities expected by the pipeline.

    BraTS provides T1, T1ce (post-contrast T1), T2, FLAIR.
    Our MVP uses T1ce + T2 + FLAIR (3 channels).
    """

    T1 = "t1"
    T1CE = "t1ce"
    T2 = "t2"
    FLAIR = "flair"


class VolumeInput(BaseModel):
    """A multi-modal MRI volume presented to a Predictor.

    The shape convention is (C, D, H, W) where:
      - C = number of modality channels (typically 3 or 4)
      - D = depth (slices along axial axis)
      - H = height (rows)
      - W = width (columns)
    """

    model_config = {"arbitrary_types_allowed": True}

    modalities: list[VolumeModality]
    array: Any = Field(description="numpy.ndarray, shape (C, D, H, W)")
    spacing: tuple[float, float, float] = Field(
        description="Voxel spacing in mm (D, H, W).",
    )
    affine: Any = Field(description="numpy.ndarray, shape (4, 4) — RAS+ affine.")
    case_id: str | None = Field(default=None)
    patient_id: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in physical (mm) space."""

    min_d: float
    max_d: float
    min_h: float
    max_h: float
    min_w: float
    max_w: float


class Centroid(BaseModel):
    """Centroid in physical (mm) space."""

    d: float
    h: float
    w: float


class IntensityStats(BaseModel):
    mean: float
    std: float
    median: float
    min: float
    max: float


class ShapeFeatures(BaseModel):
    surface_area_mm2: float
    sphericity: float
    principal_axes_mm: tuple[float, float, float]


class TumorFeatures(BaseModel):
    """Computational, non-clinical features per derived BraTS region.

    None of these fields constitute a medical interpretation.
    """

    label: TumorLabel
    volume_mm3: float
    voxel_count: int
    centroid: Centroid | None
    bbox: BoundingBox | None
    shape: ShapeFeatures | None
    intensity: IntensityStats | None


class TumorFeaturesBundle(BaseModel):
    features: list[TumorFeatures]
    schema_version: str = SCHEMA_VERSION


class PredictionMeta(BaseModel):
    """Provenance for a prediction."""

    model_name: str
    model_version: str
    schema_version: str = SCHEMA_VERSION
    inference_seconds: float
    notes: str | None = None


class PredictionOutput(BaseModel):
    """Output of a Predictor.

    The mask array uses native BraTS 2021 tissue labels:
      0 = background
      1 = NCR/NET (necrotic and non-enhancing tumor)
      2 = edema
      4 = enhancing tumor

    WT, TC and ET are derived overlapping regions:
      WT = labels 1, 2, 4
      TC = labels 1, 4
      ET = label 4
    """

    model_config = {"arbitrary_types_allowed": True}

    mask: Any = Field(
        description="numpy.ndarray, shape (D, H, W), native labels 0, 1, 2, 4"
    )
    probability: Any | None = Field(
        default=None,
        description=(
            "numpy.ndarray, shape (4, D, H, W), float16 — "
            "softmax probabilities in class order [0, 1, 2, 4]."
        ),
    )
    uncertainty: Any | None = Field(
        default=None,
        description="numpy.ndarray, shape (D, H, W), float16 — voxel-wise std.",
    )
    saliency: Any | None = Field(
        default=None,
        description="numpy.ndarray, shape (D, H, W), float16 — encoder saliency.",
    )
    meta: PredictionMeta


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PredictorError(RuntimeError):
    """Raised when a Predictor cannot produce an output."""
