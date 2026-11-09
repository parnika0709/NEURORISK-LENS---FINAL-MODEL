import numpy as np

from scripts.evaluate_3d import region_metrics


def test_region_metrics_match_brats_region_definitions():
    target = np.zeros((8, 8, 8), dtype=np.uint8)
    prediction = np.zeros_like(target)
    target[1:3, 1:3, 1:3] = 1
    target[3:5, 3:5, 3:5] = 2
    target[5:7, 5:7, 5:7] = 3
    prediction[:] = target

    metrics = region_metrics(prediction, target)

    assert metrics["WT"]["dice"] == 1.0
    assert metrics["TC"]["dice"] == 1.0
    assert metrics["ET"]["dice"] == 1.0
    assert all(value == 1.0 for region in metrics.values() for value in region.values())


def test_empty_region_dice_precision_and_recall_are_defined():
    zeros = np.zeros((4, 4, 4), dtype=np.uint8)

    metrics = region_metrics(zeros, zeros)

    assert all(value == 1.0 for region in metrics.values() for value in region.values())
