"""Segmentation evaluation metrics.

Reported per BraTS region (WT, TC, ET):
  - Dice coefficient (F1 of overlap)
  - IoU (Jaccard)
  - Hausdorff distance (95th percentile, mm) — surface distance, in mm
  - Sensitivity (recall)
  - Precision

All metrics are computed in physical space using the volume's voxel spacing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy import ndimage

# BraTS native regions (see schemas/contracts.py).
BRATS_REGIONS = {
    "WT": (1, 2, 4),     # whole tumor
    "TC": (1, 4),        # tumor core
    "ET": (4,),          # enhancing tumor
}

# Individual BraTS sub-compartment classes
BRATS_CLASSES = {
    "NCR_NET": (1,),     # necrotic / non-enhancing tumor
    "EDEMA": (2,),       # peritumoral edema
    "ET": (4,),          # enhancing tumor
}


@dataclass
class RegionMetrics:
    dice: float
    iou: float
    hausdorff95_mm: float
    sensitivity: float
    precision: float
    specificity: float = float("nan")
    recall: float | None = None

    def __post_init__(self) -> None:
        if self.recall is None:
            self.recall = self.sensitivity


def _binary_mask(label: np.ndarray, region: Iterable[int]) -> np.ndarray:
    out = np.zeros_like(label, dtype=bool)
    for v in region:
        out |= label == v
    return out


def dice(
    pred: np.ndarray,
    target: np.ndarray,
    region: Iterable[int],
    empty_val: float | None = None,
) -> float:
    p = _binary_mask(pred, region)
    t = _binary_mask(target, region)
    intersection = np.logical_and(p, t).sum()
    denom = p.sum() + t.sum()
    if denom == 0:
        return empty_val if empty_val is not None else float("nan")
    return float(2.0 * intersection / denom)


def iou(
    pred: np.ndarray,
    target: np.ndarray,
    region: Iterable[int],
    empty_val: float | None = None,
) -> float:
    p = _binary_mask(pred, region)
    t = _binary_mask(target, region)
    intersection = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    if union == 0:
        return empty_val if empty_val is not None else float("nan")
    return float(intersection / union)


def sensitivity(
    pred: np.ndarray,
    target: np.ndarray,
    region: Iterable[int],
    empty_val: float | None = None,
) -> float:
    p = _binary_mask(pred, region)
    t = _binary_mask(target, region)
    if t.sum() == 0:
        return empty_val if empty_val is not None else float("nan")
    return float(np.logical_and(p, t).sum() / t.sum())


def recall(
    pred: np.ndarray,
    target: np.ndarray,
    region: Iterable[int],
    empty_val: float | None = None,
) -> float:
    """Recall metric (clinically synonymous with Sensitivity)."""
    return sensitivity(pred, target, region, empty_val=empty_val)


def specificity(
    pred: np.ndarray,
    target: np.ndarray,
    region: Iterable[int],
    empty_val: float | None = None,
) -> float:
    """True Negative Rate (TNR): TN / (TN + FP)."""
    p = _binary_mask(pred, region)
    t = _binary_mask(target, region)
    neg_t = ~t
    if neg_t.sum() == 0:
        return empty_val if empty_val is not None else float("nan")
    true_neg = np.logical_and(~p, neg_t).sum()
    return float(true_neg / neg_t.sum())


def precision(
    pred: np.ndarray,
    target: np.ndarray,
    region: Iterable[int],
    empty_val: float | None = None,
) -> float:
    p = _binary_mask(pred, region)
    t = _binary_mask(target, region)
    if p.sum() == 0:
        return empty_val if empty_val is not None else float("nan")
    return float(np.logical_and(p, t).sum() / p.sum())


def hausdorff95_mm(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    region: Iterable[int],
    empty_val: float | None = None,
) -> float:
    """95th-percentile Hausdorff distance in mm, with physical spacing.

    Returns ``inf`` if either mask is empty (a meaningful boundary does not
    exist) and ``nan`` (or empty_val if provided) if both are empty (region absent in both).
    """
    p = _binary_mask(pred, region)
    t = _binary_mask(target, region)
    if not p.any() and not t.any():
        return empty_val if empty_val is not None else float("nan")
    if not p.any() or not t.any():
        return float("inf")

    # Boundary voxels = surface of each mask (boundary - eroded).
    p_boundary = p ^ ndimage.binary_erosion(p)
    t_boundary = t ^ ndimage.binary_erosion(t)

    p_coords = np.argwhere(p_boundary) * np.asarray(spacing)
    t_coords = np.argwhere(t_boundary) * np.asarray(spacing)
    if p_coords.size == 0 or t_coords.size == 0:
        return empty_val if empty_val is not None else float("nan")

    # Compute pairwise distances in chunks to avoid huge allocations.
    # For 95th percentile we don't need all pairs — random sample cap.
    max_pairs = 5_000_000
    if len(p_coords) * len(t_coords) > max_pairs:
        rng = np.random.default_rng(0)
        p_idx = rng.choice(len(p_coords), size=max_pairs // len(t_coords), replace=False)
        p_coords = p_coords[p_idx]

    diff = p_coords[:, None, :] - t_coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    forward = np.percentile(dist.min(axis=1), 95)
    backward = np.percentile(dist.min(axis=0), 95)
    return float(max(forward, backward))


def evaluate_case(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    regions: dict[str, tuple[int, ...]] | None = None,
    empty_val: float | None = None,
) -> dict[str, RegionMetrics]:
    """Compute metrics for all BraTS regions or custom classes on a single case."""
    target_regions = regions or BRATS_REGIONS
    out: dict[str, RegionMetrics] = {}
    for region_name, region_labels in target_regions.items():
        sens = sensitivity(pred, target, region_labels, empty_val=empty_val)
        out[region_name] = RegionMetrics(
            dice=dice(pred, target, region_labels, empty_val=empty_val),
            iou=iou(pred, target, region_labels, empty_val=empty_val),
            hausdorff95_mm=hausdorff95_mm(pred, target, spacing, region_labels, empty_val=empty_val),
            sensitivity=sens,
            precision=precision(pred, target, region_labels, empty_val=empty_val),
            specificity=specificity(pred, target, region_labels, empty_val=empty_val),
            recall=sens,
        )
    return out


def aggregate_metrics(
    per_case: list[dict[str, RegionMetrics]],
    regions: Iterable[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate per-case metrics into summary statistics per region, ignoring NaNs."""
    target_regions = list(regions) if regions is not None else list(BRATS_REGIONS.keys())
    aggregated: dict[str, dict[str, float]] = {}
    metric_keys = (
        "dice",
        "iou",
        "hausdorff95_mm",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
    )

    macro_accumulators: dict[str, list[float]] = {m: [] for m in metric_keys}

    for region_name in target_regions:
        region_vals: dict[str, list[float]] = {m: [] for m in metric_keys}
        for case in per_case:
            if region_name not in case:
                continue
            rm = case[region_name]
            for m in metric_keys:
                value = getattr(rm, m, float("nan"))
                if not np.isnan(value) and not np.isinf(value):
                    region_vals[m].append(float(value))

        region_summary: dict[str, Any] = {}
        for m, vals in region_vals.items():
            if vals:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                med_val = float(np.median(vals))
                q25, q75 = np.percentile(vals, [25, 75])
                iqr_val = float(q75 - q25)
                macro_accumulators[m].extend(vals)
            else:
                mean_val = float("nan")
                std_val = float("nan")
                med_val = float("nan")
                iqr_val = float("nan")

            region_summary[m] = {
                "mean": mean_val,
                "std": std_val,
                "median": med_val,
                "iqr": iqr_val,
                "n": len(vals),
            }

        aggregated[region_name] = region_summary

    # Overall macro average across all regions
    overall_summary: dict[str, Any] = {}
    for m, vals in macro_accumulators.items():
        if vals:
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            med_val = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25, 75])
            iqr_val = float(q75 - q25)
        else:
            mean_val = float("nan")
            std_val = float("nan")
            med_val = float("nan")
            iqr_val = float("nan")

        overall_summary[m] = {
            "mean": mean_val,
            "std": std_val,
            "median": med_val,
            "iqr": iqr_val,
            "n": len(vals),
        }
    aggregated["overall_macro"] = overall_summary

    return aggregated
