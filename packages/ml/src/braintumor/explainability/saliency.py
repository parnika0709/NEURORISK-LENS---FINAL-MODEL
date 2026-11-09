"""Encoder saliency for U-Net segmentation models.

Uses Grad-CAM on the deepest encoder block to highlight which spatial
regions most influenced the segmentation. This is a *spatial* saliency map,
not a per-class attribution — we visualize *where the model looked*, not
*why it chose a class*.

For segmentation models, "class-discriminative" Grad-CAM (as used for
classification) is not strictly meaningful; we frame this as encoder
attention, which is the standard interpretation in medical imaging.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def encoder_gradcam(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int = 1,
) -> np.ndarray:
    """Compute a Grad-CAM saliency map on the deepest encoder block.

    Parameters
    ----------
    model : nn.Module
        The Attention U-Net (or similar with an encoder).
    x : torch.Tensor
        Input tensor (B, C, H, W) — single-slice batch expected.
    target_class : int
        Which output channel to compute saliency for. Default 1 = WT.

    Returns
    -------
    numpy.ndarray
        Saliency map of shape (H, W), normalized to [0, 1].
    """
    if not hasattr(model, "model"):
        raise ValueError("Expected model with `.model` attribute (MONAI wrapper).")

    inner = model.model
    # MONAI AttentionUnet exposes `.model` (down) and `.upcat` blocks, but
    # the deepest encoder activation is accessible via `bottom_block`.
    if not hasattr(inner, "bottom_block"):
        raise ValueError("Model does not expose `bottom_block`; cannot compute Grad-CAM.")

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_module, _input, output):
        activations.append(output)
        output.requires_grad_(True)
        output.retain_grad()

    def bwd_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_f = inner.bottom_block.register_forward_hook(fwd_hook)
    handle_b = inner.bottom_block.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad()
        # Enable grad even if the model is in eval mode for this slice.
        with torch.enable_grad():
            logits = model(x)
            score = logits[:, target_class].sum()
            score.backward()

        if not activations or not gradients:
            return np.zeros((x.shape[-2], x.shape[-1]), dtype=np.float32)

        act = activations[0]  # (B, C', h, w)
        grad = gradients[0]  # (B, C', h, w)

        weights = grad.mean(dim=(-2, -1), keepdim=True)  # (B, C', 1, 1)
        cam = (weights * act).sum(dim=1, keepdim=True)  # (B, 1, h, w)
        cam = F.relu(cam)

        # Resize CAM to input spatial size.
        cam = F.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )  # (B, 1, H, W)

        saliency = cam.squeeze().detach().cpu().numpy()
        saliency = saliency - saliency.min()
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        return saliency.astype(np.float16)

    finally:
        handle_f.remove()
        handle_b.remove()
