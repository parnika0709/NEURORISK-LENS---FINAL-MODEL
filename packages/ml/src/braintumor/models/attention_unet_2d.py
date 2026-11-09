"""2D Attention U-Net for BraTS-style multi-modal MRI segmentation.

Input channels:
    0 = T1ce
    1 = T2
    2 = FLAIR

Output labels:
    0 = Background
    1 = NCR/NET
    2 = Edema
    4 = Enhancing tumor

Input volumes must already use the deterministic validation preprocessing:
    RAS orientation -> 1 mm spacing -> percentile scaling -> z-score.
"""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import AttentionUnet

from braintumor.data.preprocessing import fourclass_to_brats
from braintumor.schemas.contracts import (
    PredictionMeta,
    PredictionOutput,
    VolumeInput,
    VolumeModality,
)
from braintumor.registry.predictor import Predictor


MVP_MODALITIES: tuple[VolumeModality, ...] = (
    VolumeModality.T1CE,
    VolumeModality.T2,
    VolumeModality.FLAIR,
)


class AttentionUnet2D(nn.Module):
    """MONAI Attention U-Net with configurable depth."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: tuple[int, ...] = (2, 2, 2, 2),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if len(channels) - 1 != len(strides):
            raise ValueError(
                "len(channels) must equal len(strides) + 1"
            )

        self.model = AttentionUnet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            dropout=dropout,
        )

        # Total downsampling factor; H and W must be divisible by this.
        total_stride = 1
        for s in strides:
            total_stride *= s
        self.total_stride = total_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AttentionUnet2DPredictor(Predictor):
    """Inference wrapper implementing the Predictor contract."""

    name = "attn_unet_2d"
    version = "0.1.0"
    description = "2D Attention U-Net (T1ce+T2+FLAIR, 4 BraTS classes)"

    def __init__(
        self,
        weights_path: Path | None = None,
        device: str | None = None,
        eval_dropout_passes: int = 1,
        chunk_size: int = 16,
    ) -> None:
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.eval_dropout_passes = max(1, eval_dropout_passes)
        self.chunk_size = chunk_size

        self.model = AttentionUnet2D(
            in_channels=3,
            out_channels=4,
        ).to(self.device)

        self.model.eval()

        if weights_path is not None:
            self.load_weights(weights_path)

    def load_weights(self, path: Path) -> None:
        """Load a checkpoint and require an exact architecture match.

        Checkpoints saved on Windows may contain pickled WindowsPath objects,
        which cannot be instantiated on macOS/Linux. We temporarily map
        WindowsPath to PosixPath while loading so those checkpoints work.
        """
        original_windows_path = pathlib.WindowsPath
        if sys.platform != "win32":
            pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[misc,assignment]

        try:
            state = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )
        finally:
            pathlib.WindowsPath = original_windows_path  # type: ignore[misc]

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.no_grad()
    def predict(self, volume: VolumeInput) -> PredictionOutput:
        import time

        start = time.perf_counter()

        x = self._to_tensor(volume)

        # Model output: (4, X, Y, Z).
        logits = self._forward(x)

        probs = torch.softmax(logits, dim=0)
        class_mask = torch.argmax(probs, dim=0)

        # Convert training classes {0,1,2,3} to BraTS labels {0,1,2,4}.
        native_mask = fourclass_to_brats(
            class_mask.cpu().numpy()
        )

        elapsed = time.perf_counter() - start

        return PredictionOutput(
            mask=native_mask,
            probability=probs.cpu().numpy().astype(np.float16),
            uncertainty=None,
            saliency=None,
            meta=PredictionMeta(
                model_name=self.name,
                model_version=self.version,
                inference_seconds=float(elapsed),
                notes="Input supplied using deterministic validation preprocessing.",
            ),
        )

    def _to_tensor(self, volume: VolumeInput) -> torch.Tensor:
        """Convert preprocessed (C,D,H,W) MRI data to a model tensor.

        Important: preprocessing and normalization happen before this method,
        using val_transforms(). Do not normalize the data a second time here.
        """
        if volume.array is None:
            raise ValueError("VolumeInput.array is None")

        arr = np.asarray(volume.array, dtype=np.float32)

        if arr.ndim != 4:
            raise ValueError(
                f"Expected (C, D, H, W); got shape {arr.shape}"
            )

        name_to_idx = {
            modality.value: i
            for i, modality in enumerate(volume.modalities)
        }

        missing = [
            modality.value
            for modality in MVP_MODALITIES
            if modality.value not in name_to_idx
        ]

        if missing:
            raise ValueError(
                f"Volume missing required modalities: {missing}. "
                f"Provided: {[m.value for m in volume.modalities]}"
            )

        channel_indices = [
            name_to_idx[modality.value]
            for modality in MVP_MODALITIES
        ]

        arr = arr[channel_indices, ...].copy()

        if not np.isfinite(arr).all():
            raise ValueError(
                "Input volume contains NaN or infinite values"
            )

        tensor = torch.from_numpy(arr)
        return tensor.to(self.device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model slice-by-slice in batches.

        Training (braintumor.data.slicing) fed the network X-Y slices taken
        along Z, i.e. ``image[:, :, :, z]`` of a (C, X, Y, Z) volume, so
        inference must slice the same way. Each slice is zero-padded so its
        height/width are divisible by the network's total downsampling factor
        (16), then the output is cropped back.

        Input:  (C, X, Y, Z)
        Output: (4, X, Y, Z)
        """
        # (C, X, Y, Z) -> (Z, C, X, Y): one 2D image per axial (Z) slice.
        slices = x.permute(3, 0, 1, 2)

        depth = slices.shape[0]
        h, w = slices.shape[-2:]

        mult = self.model.total_stride
        pad_h = (-h) % mult
        pad_w = (-w) % mult

        outputs: list[torch.Tensor] = []

        for start in range(0, depth, self.chunk_size):
            end = min(start + self.chunk_size, depth)
            batch = slices[start:end]

            if pad_h or pad_w:
                batch = F.pad(batch, (0, pad_w, 0, pad_h))

            output = self.model(batch)
            output = output[..., :h, :w]
            outputs.append(output)

        combined = torch.cat(outputs, dim=0)  # (Z, 4, X, Y)
        return combined.permute(1, 2, 3, 0)   # (4, X, Y, Z)


def build_attn_unet_2d() -> AttentionUnet2DPredictor:
    """Factory for registry use."""
    return AttentionUnet2DPredictor()