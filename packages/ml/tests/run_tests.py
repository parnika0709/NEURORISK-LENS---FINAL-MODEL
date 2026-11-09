import unittest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from braintumor.data.preprocessing import (
    brats_label_to_4class,
    fourclass_to_brats,
)
from braintumor.models.attention_unet_2d import AttentionUnet2D
from braintumor.eval.metrics import dice, iou, sensitivity, precision
import numpy as np
import torch

class TestMLFoundation(unittest.TestCase):

    def test_brats_label_mapping(self):
        label = np.array([0, 1, 2, 4], dtype=np.uint8)
        mapped = brats_label_to_4class(label)
        np.testing.assert_array_equal(mapped, [0, 1, 1, 3])
        back = fourclass_to_brats(mapped)
        np.testing.assert_array_equal(back, [0, 1, 1, 4])

    def test_model_forward_pass(self):
        batch_size = 2
        in_channels = 3
        h, w = 128, 128
        out_channels = 4
        model = AttentionUnet2D(in_channels=in_channels, out_channels=out_channels)
        model.eval()
        x = torch.randn(batch_size, in_channels, h, w)
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, (batch_size, out_channels, h, w))

    def test_metrics_perfect_overlap(self):
        pred = np.array([0, 1, 0, 1], dtype=np.uint8)
        target = np.array([0, 1, 0, 1], dtype=np.uint8)
        region = [1]
        self.assertEqual(dice(pred, target, region), 1.0)
        self.assertEqual(iou(pred, target, region), 1.0)
        self.assertEqual(sensitivity(pred, target, region), 1.0)
        self.assertEqual(precision(pred, target, region), 1.0)

if __name__ == "__main__":
    unittest.main()
