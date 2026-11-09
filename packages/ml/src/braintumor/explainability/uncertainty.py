"""Monte Carlo Dropout uncertainty for segmentation models.

Standard technique:
  1. Keep dropout layers active at inference (``model.train()``).
  2. Run N stochastic forward passes.
  3. Aggregate:
       - mean probability across passes → prediction
       - per-voxel std → uncertainty

Returns a per-volume uncertainty map in [0, ~1] range. Higher = more
disagreement across MC passes = less confident prediction.
"""

from __future__ import annotations

import numpy as np
import torch

from braintumor.registry.predictor import Predictor
from braintumor.schemas.contracts import VolumeInput


@torch.no_grad()
def mc_dropout_predict(
    predictor: Predictor,
    volume: VolumeInput,
    n_passes: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_argmax_mask, mean_prob, uncertainty_std).

    Each forward pass re-enables dropout on the underlying nn.Module.
    """
    if not hasattr(predictor, "model"):
        raise ValueError("Predictor must expose a `.model` attribute (nn.Module).")

    predictor.model.train()  # enable dropout

    x = predictor._to_tensor(volume)  # type: ignore[attr-defined]
    chunks: list[torch.Tensor] = []

    for _ in range(n_passes):
        logits = predictor._forward(x)  # type: ignore[attr-defined]
        probs = torch.softmax(logits, dim=1)  # (4, D, H, W)
        chunks.append(probs)

    predictor.model.eval()

    stack = torch.stack(chunks, dim=0)  # (N, 4, D, H, W)
    mean_prob = stack.mean(dim=0)  # (4, D, H, W)
    std_prob = stack.std(dim=0)  # (4, D, H, W)

    # Reduce class dimension: uncertainty = mean std across classes.
    uncertainty = std_prob.mean(dim=0)  # (D, H, W)
    mean_mask = torch.argmax(mean_prob, dim=0).to(torch.int8)

    return (
        mean_mask.cpu().numpy(),
        mean_prob.cpu().numpy().astype(np.float16),
        uncertainty.cpu().numpy().astype(np.float16),
    )
