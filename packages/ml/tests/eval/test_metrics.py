import numpy as np
import pytest
from braintumor.eval.metrics import (
    BRATS_CLASSES,
    BRATS_REGIONS,
    aggregate_metrics,
    dice,
    evaluate_case,
    hausdorff95_mm,
    iou,
    precision,
    recall,
    sensitivity,
    specificity,
)

def test_metrics_perfect_overlap():
    """Test metrics when prediction and target are identical."""
    pred = np.array([0, 1, 0, 1], dtype=np.uint8)
    target = np.array([0, 1, 0, 1], dtype=np.uint8)
    region = [1]

    assert dice(pred, target, region) == 1.0
    assert iou(pred, target, region) == 1.0
    assert sensitivity(pred, target, region) == 1.0
    assert recall(pred, target, region) == 1.0
    assert precision(pred, target, region) == 1.0
    assert specificity(pred, target, region) == 1.0

def test_metrics_no_overlap():
    """Test metrics when there is no overlap."""
    pred = np.array([0, 1, 0, 0], dtype=np.uint8)
    target = np.array([0, 0, 1, 0], dtype=np.uint8)
    region = [1]

    assert dice(pred, target, region) == 0.0
    assert iou(pred, target, region) == 0.0
    assert sensitivity(pred, target, region) == 0.0
    assert recall(pred, target, region) == 0.0
    assert precision(pred, target, region) == 0.0
    # For target region [1], target has 0 ones (indices 0, 1, 2, 3 all have ~target).
    # Since target has no label 1 at all, sensitivity is NaN or 0 if empty_val set:
    target_has_region = np.array([1, 0, 0, 0], dtype=np.uint8)
    pred_no_overlap = np.array([0, 1, 0, 0], dtype=np.uint8)
    assert dice(pred_no_overlap, target_has_region, region) == 0.0
    assert sensitivity(pred_no_overlap, target_has_region, region) == 0.0
    assert precision(pred_no_overlap, target_has_region, region) == 0.0
    assert specificity(pred_no_overlap, target_has_region, region) == 2 / 3

def test_metrics_empty_masks():
    """Test metrics when masks are empty."""
    pred = np.zeros(10, dtype=np.uint8)
    target = np.zeros(10, dtype=np.uint8)
    region = [1]

    # Default behaviour preserves original NaN returns
    assert np.isnan(dice(pred, target, region))
    assert np.isnan(iou(pred, target, region))
    assert np.isnan(sensitivity(pred, target, region))
    assert np.isnan(precision(pred, target, region))
    assert specificity(pred, target, region) == 1.0  # true negative rate when target has no foreground

    # With empty_val=1.0 handling
    assert dice(pred, target, region, empty_val=1.0) == 1.0
    assert iou(pred, target, region, empty_val=1.0) == 1.0
    assert sensitivity(pred, target, region, empty_val=1.0) == 1.0
    assert recall(pred, target, region, empty_val=1.0) == 1.0
    assert precision(pred, target, region, empty_val=1.0) == 1.0

def test_hausdorff95_3d():
    """Test HD95 in 3D volume."""
    pred = np.zeros((10, 10, 10), dtype=np.uint8)
    target = np.zeros((10, 10, 10), dtype=np.uint8)
    pred[2:5, 2:5, 2:5] = 1
    target[2:5, 2:5, 2:5] = 1

    hd = hausdorff95_mm(pred, target, (1.0, 1.0, 1.0), [1])
    assert hd == 0.0

    # Test empty handling
    empty = np.zeros((10, 10, 10), dtype=np.uint8)
    assert np.isnan(hausdorff95_mm(empty, empty, (1.0, 1.0, 1.0), [1]))
    assert hausdorff95_mm(empty, empty, (1.0, 1.0, 1.0), [1], empty_val=0.0) == 0.0
    assert hausdorff95_mm(pred, empty, (1.0, 1.0, 1.0), [1]) == float("inf")

def test_evaluate_case_and_aggregate():
    """Test case-level evaluation and multi-case aggregation."""
    pred = np.zeros((20, 20, 20), dtype=np.uint8)
    target = np.zeros((20, 20, 20), dtype=np.uint8)

    # Place NCR/NET (1), EDEMA (2), ET (4)
    pred[5:10, 5:10, 5:10] = 1
    target[5:10, 5:10, 5:10] = 1
    pred[10:15, 10:15, 10:15] = 2
    target[10:15, 10:15, 10:15] = 2
    pred[15:18, 15:18, 15:18] = 4
    target[15:18, 15:18, 15:18] = 4

    case_metrics = evaluate_case(pred, target, (1.0, 1.0, 1.0))
    for region in ("WT", "TC", "ET"):
        assert region in case_metrics
        assert case_metrics[region].dice == 1.0
        assert case_metrics[region].recall == 1.0
        assert case_metrics[region].specificity == 1.0

    # Also evaluate individual classes
    class_metrics = evaluate_case(pred, target, (1.0, 1.0, 1.0), regions=BRATS_CLASSES)
    for cls_name in ("NCR_NET", "EDEMA", "ET"):
        assert cls_name in class_metrics
        assert class_metrics[cls_name].dice == 1.0

    # Test aggregation across 2 identical cases
    agg = aggregate_metrics([case_metrics, case_metrics])
    assert "WT" in agg
    assert agg["WT"]["dice"]["mean"] == 1.0
    assert agg["WT"]["dice"]["std"] == 0.0
    assert agg["WT"]["dice"]["median"] == 1.0
    assert agg["WT"]["dice"]["iqr"] == 0.0
    assert agg["WT"]["specificity"]["mean"] == 1.0
    assert "overall_macro" in agg
    assert agg["overall_macro"]["dice"]["mean"] == 1.0
