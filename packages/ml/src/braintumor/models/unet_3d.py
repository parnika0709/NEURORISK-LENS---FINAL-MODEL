"""Experimental Volumetric 3D U-Net for BraTS Multi-Modal MRI Segmentation.

IMPORTANT: This is an independent experimental model developed to address Reviewer Comment 7.
It does NOT replace or modify the existing 2D Attention U-Net baseline.
The 2D Attention U-Net remains the permanent default model of the application.

Input channels:
    0 = FLAIR
    1 = T1
    2 = T1ce
    3 = T2

Output labels:
    0 = Background
    1 = NCR/NET (Necrotic / Non-enhancing tumor core)
    2 = Edema (Peritumoral edema)
    4 = Enhancing tumor (ET)

Input volumes must already use the deterministic validation preprocessing:
    RAS orientation -> 1 mm spacing -> percentile scaling -> z-score.
"""

from __future__ import annotations

import pathlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

from braintumor.data.preprocessing import fourclass_to_brats
from braintumor.registry.predictor import Predictor
from braintumor.schemas.contracts import (
    PredictionMeta,
    PredictionOutput,
    VolumeInput,
    VolumeModality,
)

UNET3D_MODALITIES: tuple[VolumeModality, ...] = (
    VolumeModality.FLAIR,
    VolumeModality.T1,
    VolumeModality.T1CE,
    VolumeModality.T2,
)
_LEGACY_UNET3D_MODALITIES: tuple[VolumeModality, ...] = (
    VolumeModality.T1CE,
    VolumeModality.T2,
    VolumeModality.FLAIR,
)


class Unet3D(nn.Module):
    """Volumetric 3D U-Net with genuine 3D convolutions, residual units, and skip connections."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        strides: tuple[int, ...] = (2, 2, 2),
        num_res_units: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if len(channels) - 1 != len(strides):
            raise ValueError("len(channels) must equal len(strides) + 1")

        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout,
        )

        total_stride = 1
        for s in strides:
            total_stride *= s
        self.total_stride = total_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 3D tensor: (B, C, D, H, W) -> (B, out_channels, D, H, W)."""
        return self.model(x)


class Unet3DPredictor(Predictor):
    """Inference wrapper for the 3D U-Net implementing the Predictor contract."""

    name = "unet_3d"
    version = "0.1.0"
    description = "Volumetric 3D U-Net (FLAIR+T1+T1ce+T2, 4 BraTS classes)"

    def __init__(
        self,
        weights_path: Path | None = None,
        device: str | None = None,
        roi_size: tuple[int, int, int] = (64, 64, 64),
        sw_batch_size: int = 2,
        overlap: float = 0.25,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        in_channels: int = 4,
        out_channels: int = 4,
        strides: tuple[int, ...] = (2, 2, 2),
        num_res_units: int = 2,
        dropout: float = 0.2,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap

        self.channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = tuple(strides)
        self.num_res_units = num_res_units
        self.dropout = dropout
        self.weights_path: Path | None = None

        self.model = Unet3D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=self.channels,
            strides=self.strides,
            num_res_units=self.num_res_units,
            dropout=self.dropout,
        ).to(self.device)

        self.model.eval()

        if weights_path is not None:
            self.load_weights(weights_path)

    def load_weights(self, path: Path) -> None:
        """Load a checkpoint with cross-platform PosixPath compatibility and architecture adaptation.

        Supports both original CPU checkpoints with channels (16, 32, 64, 128) and
        high-capacity GPU checkpoints with channels (32, 64, 128, 256) or custom metadata.
        Validates tensor shapes against metadata before loading to prevent silent failures.
        """
        orig_windows_path = getattr(pathlib, "WindowsPath", None)
        if sys.platform != "win32" and orig_windows_path is not None:
            pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[misc,assignment]

        try:
            state = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )
        finally:
            if orig_windows_path is not None:
                pathlib.WindowsPath = orig_windows_path  # type: ignore[misc]

        if not isinstance(state, dict):
            raise ValueError(f"Expected checkpoint dictionary at {path}, got {type(state)}")

        # Extract state_dict
        if "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state

        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError(f"No valid state_dict found in checkpoint at {path}")

        # Resolve architecture metadata
        arch_config = state.get("arch_config", {})
        if not isinstance(arch_config, dict):
            arch_config = {}

        # 1. Resolve channels
        resolved_channels = None
        if "channels" in arch_config:
            resolved_channels = tuple(arch_config["channels"])
        elif "channels" in state:
            resolved_channels = tuple(state["channels"])

        # 2. Resolve in_channels, out_channels, strides, num_res_units, dropout
        resolved_in_channels = arch_config.get(
            "in_channels", state.get("in_channels", self.in_channels)
        )
        resolved_out_channels = arch_config.get(
            "out_channels", state.get("out_channels", self.out_channels)
        )
        resolved_strides = tuple(
            arch_config.get("strides", state.get("strides", self.strides))
        )
        resolved_num_res_units = arch_config.get(
            "num_res_units", state.get("num_res_units", self.num_res_units)
        )
        resolved_dropout = float(
            arch_config.get("dropout", state.get("dropout", self.dropout))
        )

        # Backward compatibility: inspect first layer weights to detect or validate channels
        first_conv_key = "model.model.0.conv.unit0.conv.weight"
        if first_conv_key in state_dict:
            first_weight = state_dict[first_conv_key]
            first_out_ch = first_weight.shape[0]
            first_in_ch = first_weight.shape[1]

            # If checkpoint had no channel metadata, determine channels from first layer
            if resolved_channels is None:
                if first_out_ch == 16:
                    resolved_channels = (16, 32, 64, 128)
                elif first_out_ch == 32:
                    resolved_channels = (32, 64, 128, 256)
                else:
                    raise ValueError(
                        f"Cannot infer channels for checkpoint at {path}: "
                        f"first conv layer has out_channels={first_out_ch}"
                    )

            # Comprehensive multi-level validation: check key milestone tensor shapes
            key_shapes = {
                "model.model.0.conv.unit0.conv.weight": (resolved_channels[0], resolved_in_channels, 3, 3, 3),
                "model.model.1.submodule.0.conv.unit0.conv.weight": (resolved_channels[1], resolved_channels[0], 3, 3, 3),
                "model.model.1.submodule.1.submodule.0.conv.unit0.conv.weight": (resolved_channels[2], resolved_channels[1], 3, 3, 3),
                "model.model.1.submodule.1.submodule.1.submodule.conv.unit0.conv.weight": (resolved_channels[3], resolved_channels[2], 3, 3, 3),
                "model.model.2.1.conv.unit0.conv.weight": (resolved_out_channels, resolved_out_channels, 3, 3, 3),
            }
            for k, expected_shape in key_shapes.items():
                if k in state_dict:
                    actual_shape = tuple(state_dict[k].shape)
                    if actual_shape != expected_shape:
                        raise ValueError(
                            f"Checkpoint architecture mismatch: metadata specifies channels={resolved_channels}, "
                            f"in_channels={resolved_in_channels}, out_channels={resolved_out_channels}. "
                            f"Expected layer tensor '{k}' with shape {expected_shape}, but found {actual_shape}."
                        )
        else:
            if resolved_channels is None:
                resolved_channels = (16, 32, 64, 128)

        # Rebuild model if architecture differs from current model
        if (
            self.channels != resolved_channels
            or self.in_channels != resolved_in_channels
            or self.out_channels != resolved_out_channels
            or self.strides != resolved_strides
            or self.num_res_units != resolved_num_res_units
            or self.dropout != resolved_dropout
        ):
            self.channels = resolved_channels
            self.in_channels = resolved_in_channels
            self.out_channels = resolved_out_channels
            self.strides = resolved_strides
            self.num_res_units = resolved_num_res_units
            self.dropout = resolved_dropout
            self.model = Unet3D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                channels=self.channels,
                strides=self.strides,
                num_res_units=self.num_res_units,
                dropout=self.dropout,
            ).to(self.device)

        # Load state_dict with strict=True
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as err:
            raise RuntimeError(
                f"Failed to load checkpoint into Unet3D architecture with channels {resolved_channels}: {err}"
            ) from err

        self.model.eval()
        self.weights_path = Path(path)

    @torch.no_grad()
    def predict(
        self,
        volume: VolumeInput,
        apply_brain_mask: bool = True,
    ) -> PredictionOutput:
        """Run 3D volumetric sliding-window inference on input volume.

        Parameters
        ----------
        volume : VolumeInput
            Input MRI volume with (C, D, H, W) array and spatial metadata.
        apply_brain_mask : bool, optional
            Whether to enforce an anatomical brain foreground constraint to eliminate
            empty scanner air false positives outside the skull. Default is True.
        """
        start = time.perf_counter()

        x = self._to_tensor(volume)  # (C, D, H, W)
        x_batched = x.unsqueeze(0)   # (1, C, D, H, W)

        # Sliding-window volumetric inference handles arbitrary volume dimensions
        # and manages memory footprint on CPU/GPU
        logits = sliding_window_inference(
            inputs=x_batched,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.model,
            overlap=self.overlap,
            mode="gaussian",
        )  # (1, 4, D, H, W)

        logits = logits.squeeze(0)  # (4, D, H, W)
        probs = torch.softmax(logits, dim=0)
        class_mask = torch.argmax(probs, dim=0)

        # Convert training classes {0,1,2,3} to native BraTS {0,1,2,4}
        native_mask = fourclass_to_brats(class_mask.cpu().numpy())

        # Brain tissue foreground constraint (suppresses scanner air background false positives)
        if apply_brain_mask and volume.array is not None:
            raw_arr = np.asarray(volume.array)
            has_signal = (np.abs(raw_arr) > 1e-4).any(axis=0)
            brain_mask = ndimage.binary_fill_holes(has_signal)
            native_mask = np.where(brain_mask, native_mask, 0).astype(np.uint8)

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
                notes="Experimental volumetric 3D U-Net evaluated with sliding-window inference and brain-mask constraint.",
            ),
        )

    def _to_tensor(self, volume: VolumeInput) -> torch.Tensor:
        """Convert preprocessed MRI volume to a 3D model tensor."""
        if volume.array is None:
            raise ValueError("VolumeInput.array is None")

        arr = np.asarray(volume.array, dtype=np.float32)

        if arr.ndim != 4:
            raise ValueError(f"Expected (C, D, H, W); got shape {arr.shape}")

        name_to_idx = {
            modality.value: i for i, modality in enumerate(volume.modalities)
        }

        required_modalities = (
            UNET3D_MODALITIES
            if self.in_channels == len(UNET3D_MODALITIES)
            else _LEGACY_UNET3D_MODALITIES
            if self.in_channels == len(_LEGACY_UNET3D_MODALITIES)
            else tuple(volume.modalities)
        )
        if len(required_modalities) != self.in_channels:
            raise ValueError(
                f"Checkpoint expects {self.in_channels} input channels, but the "
                f"volume provides {len(volume.modalities)} modalities."
            )

        missing = [m.value for m in required_modalities if m.value not in name_to_idx]

        if missing:
            raise ValueError(
                f"Volume missing required modalities: {missing}. "
                f"Provided: {[m.value for m in volume.modalities]}"
            )

        channel_indices = [name_to_idx[m.value] for m in required_modalities]
        arr = arr[channel_indices, ...].copy()

        if not np.isfinite(arr).all():
            raise ValueError("Input volume contains NaN or infinite values")

        tensor = torch.from_numpy(arr)
        return tensor.to(self.device)


def build_unet_3d(weights_path: Path | None = None) -> Unet3DPredictor:
    """Factory for registry use."""
    return Unet3DPredictor(weights_path=weights_path)
