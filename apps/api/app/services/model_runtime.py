"""Cached inference runtime for the repository's trained BraTS checkpoints."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import MetaTensor

from app.config import settings
from braintumor.data.brats_2d_four_channel import build_2d_volume_transform
from braintumor.data.preprocessing import fourclass_to_brats
from braintumor.data.transforms_3d import build_3d_volume_transform
from braintumor.models.attention_unet_2d_four_channel import (
    AttentionUNet2DFourChannel,
)
from braintumor.models.unet_3d import Unet3DPredictor
from braintumor.schemas.contracts import VolumeInput, VolumeModality

MODEL_MODALITIES: dict[str, tuple[VolumeModality, ...]] = {
    "attn_unet_2d": (
        VolumeModality.FLAIR,
        VolumeModality.T1,
        VolumeModality.T1CE,
        VolumeModality.T2,
    ),
    "unet_3d": (
        VolumeModality.FLAIR,
        VolumeModality.T1,
        VolumeModality.T1CE,
        VolumeModality.T2,
    ),
}


@dataclass(frozen=True)
class PreparedVolume:
    image: np.ndarray
    affine: np.ndarray
    source_image: nib.spatialimages.SpatialImage


@dataclass(frozen=True)
class InferenceResult:
    mask: np.ndarray
    affine: np.ndarray
    source_image: nib.spatialimages.SpatialImage
    inference_seconds: float
    device: str
    checkpoint_epoch: int
    model_name: str
    parameter_count: int
    gpu_memory: dict[str, int | str | None]


def _checkpoint_load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {path} is not a metadata dictionary.")
    return checkpoint


def _to_cpu_numpy(value: Any) -> np.ndarray:
    if isinstance(value, MetaTensor):
        value = value.as_tensor()
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def estimate_resampled_shape(
    source_shape: tuple[int, int, int],
    source_affine: np.ndarray,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[int, int, int]:
    axis_aligned_extent = (
        np.abs(np.asarray(source_affine, dtype=np.float64)[:3, :3])
        @ (np.asarray(source_shape, dtype=np.float64) - 1.0)
    )
    shape = np.ceil(axis_aligned_extent / np.asarray(target_spacing)).astype(int) + 1
    return tuple(int(value) for value in shape)


class ModelRuntime:
    """Load each trained model once and serialize inference for limited VRAM."""

    def __init__(
        self,
        checkpoint_2d: Path | None = None,
        checkpoint_3d: Path | None = None,
    ) -> None:
        self.checkpoints = {
            "attn_unet_2d": checkpoint_2d or settings.model_weights_path_2d_four_channel,
            "unet_3d": checkpoint_3d or settings.model_weights_path_3d,
        }
        if any(path is None for path in self.checkpoints.values()):
            raise ValueError("Both trained-model checkpoint paths must be configured.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: dict[str, torch.nn.Module | Unet3DPredictor] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._inference_lock = threading.RLock()

    def device_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": None,
            "vram_total_bytes": None,
            "vram_free_bytes": None,
        }
        if self.device.type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(self.device)
            try:
                free, total = torch.cuda.mem_get_info(self.device)
                info["vram_total_bytes"] = int(total)
                info["vram_free_bytes"] = int(free)
            except RuntimeError:
                pass
        return info

    def health_info(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "torch_version": torch.__version__,
            **self.device_info(),
        }

    def model_status(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "architecture": (
                    "2D Attention U-Net (custom four-channel, attention-gated)"
                    if name == "attn_unet_2d"
                    else "3D residual U-Net with sliding-window inference"
                ),
                "input_modalities": [
                    modality.value for modality in MODEL_MODALITIES[name]
                ],
                "output_labels": {
                    "0": "background",
                    "1": "NCR/NET",
                    "2": "edema",
                    "4": "enhancing_tumor",
                },
                "checkpoint_path": str(self.checkpoints[name]),
                "checkpoint_available": self.checkpoints[name].is_file(),
                "loaded": name in self._models,
                "device": str(self.device) if name in self._models else None,
                **self._model_metadata(name),
            }
            for name in MODEL_MODALITIES
        ]

    def _model_metadata(self, name: str) -> dict[str, Any]:
        return dict(self._metadata.get(name, {}))

    def get_parameter_count(self, name: str) -> int:
        if name not in self._metadata:
            self._load_model(name)
        return int(self._metadata.get(name, {}).get("parameter_count", 0))

    def _load_model(self, name: str) -> torch.nn.Module | Unet3DPredictor:
        if name not in MODEL_MODALITIES:
            raise ValueError(f"Unsupported model '{name}'.")
        if name in self._models:
            return self._models[name]

        checkpoint_path = self.checkpoints[name]
        checkpoint = _checkpoint_load(checkpoint_path)
        if name == "attn_unet_2d":
            training_config = checkpoint.get("config")
            if not isinstance(training_config, dict):
                raise ValueError("2D checkpoint must contain a training config.")
            model_config = training_config.get("model")
            state_dict = checkpoint.get("model_state_dict")
            if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
                raise ValueError(
                    "2D checkpoint must contain config.model and model_state_dict."
                )
            expected_modalities = ["FLAIR", "T1", "T1ce", "T2"]
            data_config = training_config.get("data")
            if not isinstance(data_config, dict) or data_config.get("modalities") != expected_modalities:
                raise ValueError("2D checkpoint modality ordering is not FLAIR, T1, T1ce, T2.")
            if model_config.get("in_channels") != 4 or model_config.get("out_channels") != 4:
                raise ValueError("2D checkpoint does not describe four input and four output channels.")
            channels_config = model_config.get("channels")
            image_size_config = model_config.get("image_size")
            if (
                not isinstance(channels_config, (list, tuple))
                or len(channels_config) < 3
                or not isinstance(image_size_config, (list, tuple))
                or len(image_size_config) != 2
            ):
                raise ValueError("2D checkpoint has invalid channel or image-size metadata.")
            channels = tuple(int(value) for value in channels_config)
            image_size = tuple(int(value) for value in image_size_config)
            if any(value <= 0 for value in (*channels, *image_size)):
                raise ValueError("2D checkpoint channel and image-size values must be positive.")
            model = AttentionUNet2DFourChannel(
                in_channels=4,
                out_channels=4,
                channels=channels,
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device).eval()
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            self._metadata[name] = {
                "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
                "parameter_count": parameter_count,
                "input_size": list(image_size),
                "channels": list(channels),
                "checkpoint_best_mean_dice": checkpoint.get("best_mean_dice"),
            }
            self._models[name] = model
            return model

        if checkpoint.get("model_name") != "unet_3d":
            raise ValueError("3D checkpoint model_name must be 'unet_3d'.")
        arch = checkpoint.get("arch_config")
        if not isinstance(arch, dict) or arch.get("spatial_dims") != 3:
            raise ValueError("3D checkpoint has missing or incompatible architecture metadata.")
        expected_modalities = ["flair", "t1", "t1ce", "t2"]
        if checkpoint.get("input_modalities") != expected_modalities:
            raise ValueError("3D checkpoint modality order is not flair, t1, t1ce, t2.")
        preprocessing = checkpoint.get("preprocessing")
        if not isinstance(preprocessing, dict):
            raise ValueError("3D checkpoint has no valid preprocessing metadata.")
        if preprocessing.get("orientation") != "RAS":
            raise ValueError("3D checkpoint preprocessing orientation is not RAS.")
        if tuple(preprocessing.get("spacing_mm", ())) != (1.0, 1.0, 1.0):
            raise ValueError("3D checkpoint preprocessing spacing is not 1 mm isotropic.")
        if preprocessing.get("label_map") != {0: 0, 1: 1, 2: 2, 4: 3}:
            raise ValueError("3D checkpoint BraTS label map is incompatible.")
        channels = arch.get("channels")
        strides = arch.get("strides")
        patch_size = checkpoint.get("patch_size")
        if (
            not isinstance(channels, (list, tuple))
            or not isinstance(strides, (list, tuple))
            or arch.get("in_channels") != 4
            or arch.get("out_channels") != 4
            or len(strides) != len(channels) - 1
            or not isinstance(patch_size, (list, tuple))
            or len(patch_size) != 3
            or any(int(value) <= 0 for value in channels)
            or any(int(value) <= 0 for value in patch_size)
        ):
            raise ValueError("3D checkpoint has invalid channel, stride, or patch-size metadata.")
        predictor = Unet3DPredictor(
            weights_path=checkpoint_path,
            device=str(self.device),
            roi_size=tuple(int(value) for value in patch_size),
            sw_batch_size=1,
            overlap=0.25,
            channels=tuple(int(value) for value in channels),
            in_channels=int(arch["in_channels"]),
            out_channels=int(arch["out_channels"]),
            strides=tuple(int(value) for value in arch["strides"]),
            num_res_units=int(arch["num_res_units"]),
            dropout=float(arch["dropout"]),
        )
        param_count = int(checkpoint.get("parameter_count", 0))
        if not param_count and hasattr(predictor, "model"):
            param_count = sum(parameter.numel() for parameter in predictor.model.parameters())
        self._metadata[name] = {
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
            "parameter_count": param_count,
            "input_size": list(patch_size),
            "channels": list(channels),
            "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
        }
        self._models[name] = predictor
        return predictor

    def _move_to_cpu(self) -> None:
        for model in self._models.values():
            if isinstance(model, Unet3DPredictor):
                model.model.to("cpu")
                model.device = "cpu"
            else:
                model.to("cpu")
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(
        self,
        name: str,
        modality_paths: dict[VolumeModality, Path],
    ) -> InferenceResult:
        with self._inference_lock:
            model = self._load_model(name)
            prepared = prepare_volume(name, modality_paths)
            gpu_memory: dict[str, int | str | None] = {
                "device": str(self.device),
                "allocated_before_bytes": None,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "free_after_bytes": None,
                "total_bytes": None,
            }
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
                gpu_memory["allocated_before_bytes"] = int(
                    torch.cuda.memory_allocated(self.device)
                )
            start = time.perf_counter()
            try:
                mask = self._predict_once(name, model, prepared)
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda":
                    raise
                self._move_to_cpu()
                model = self._models[name]
                mask = self._predict_once(name, model, prepared)
            except RuntimeError as error:
                if (
                    self.device.type != "cuda"
                    or "out of memory" not in str(error).lower()
                ):
                    raise
                self._move_to_cpu()
                model = self._models[name]
                mask = self._predict_once(name, model, prepared)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
                gpu_memory.update(
                    {
                        "peak_allocated_bytes": int(
                            torch.cuda.max_memory_allocated(self.device)
                        ),
                        "peak_reserved_bytes": int(
                            torch.cuda.max_memory_reserved(self.device)
                        ),
                        "free_after_bytes": int(free_bytes),
                        "total_bytes": int(total_bytes),
                    }
                )
            elapsed = time.perf_counter() - start

            restored = resample_mask_to_source(
                mask, prepared.affine, prepared.source_image
            )
            metadata = self._metadata[name]
            return InferenceResult(
                mask=restored,
                affine=np.asarray(prepared.source_image.affine, dtype=np.float64),
                source_image=prepared.source_image,
                inference_seconds=float(elapsed),
                device=str(next(model.model.parameters()).device)
                if isinstance(model, Unet3DPredictor)
                else str(next(model.parameters()).device),
                checkpoint_epoch=int(metadata["checkpoint_epoch"]),
                model_name=name,
                parameter_count=int(metadata["parameter_count"]),
                gpu_memory=gpu_memory,
            )

    def _predict_once(
        self,
        name: str,
        model: torch.nn.Module | Unet3DPredictor,
        prepared: PreparedVolume,
    ) -> np.ndarray:
        with torch.inference_mode():
            if name == "unet_3d":
                assert isinstance(model, Unet3DPredictor)
                modalities = MODEL_MODALITIES[name]
                volume = VolumeInput(
                    modalities=list(modalities),
                    array=prepared.image,
                    spacing=tuple(
                        float(value)
                        for value in nib.affines.voxel_sizes(prepared.affine)
                    ),
                    affine=prepared.affine,
                )
                output = model.predict(volume, apply_brain_mask=True)
                return np.asarray(output.mask, dtype=np.uint8)
            assert isinstance(model, AttentionUNet2DFourChannel)
            return predict_2d_volume(
                model,
                prepared.image,
                image_size=tuple(self._metadata[name]["input_size"]),
                device=next(model.parameters()).device,
            )


def prepare_volume(
    model_name: str,
    modality_paths: dict[VolumeModality, Path],
) -> PreparedVolume:
    modalities = MODEL_MODALITIES[model_name]
    ordered_paths = [modality_paths[modality] for modality in modalities]
    images = [nib.load(str(path)) for path in ordered_paths]
    reference = images[0]
    shape = tuple(int(value) for value in reference.shape)
    if len(shape) != 3:
        raise ValueError(f"MRI volumes must be 3D; got shape {shape}.")
    if int(np.prod(shape)) > settings.max_volume_voxels:
        raise ValueError(
            f"MRI volume has {int(np.prod(shape))} voxels; configured limit is "
            f"{settings.max_volume_voxels}."
        )
    reference_affine = np.asarray(reference.affine, dtype=np.float64)
    if not np.isfinite(reference_affine).all() or abs(np.linalg.det(reference_affine[:3, :3])) < 1e-8:
        raise ValueError("MRI affine is invalid or singular.")
    resampled_shape = estimate_resampled_shape(shape, reference_affine)
    resampled_voxels = int(np.prod(resampled_shape))
    if resampled_voxels > settings.max_volume_voxels:
        raise ValueError(
            f"Resampled RAS volume has {resampled_voxels} voxels; configured limit is "
            f"{settings.max_volume_voxels}."
        )
    for modality, image in zip(modalities, images):
        if tuple(image.shape) != shape:
            raise ValueError(
                f"{modality.value} dimensions {image.shape} do not match {shape}."
            )
        if not np.allclose(image.affine, reference_affine, rtol=0.0, atol=1e-4):
            raise ValueError(f"{modality.value} affine is not aligned with FLAIR.")
        spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError(f"{modality.value} voxel spacing is invalid.")
        data = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError(f"{modality.value} contains NaN or infinite voxels.")
        if not np.any(data):
            raise ValueError(f"{modality.value} volume is empty.")

    file_list = [str(modality_paths[modality]) for modality in modalities]
    if model_name == "attn_unet_2d":
        transformed = build_2d_volume_transform(include_label=False)(
            {"image": file_list}
        )
    else:
        transformed = build_3d_volume_transform(include_label=False)(
            {"image": file_list}
        )
    image_data = transformed["image"]
    array = _to_cpu_numpy(image_data).astype(np.float32, copy=False)
    if array.ndim != 4 or array.shape[0] != 4:
        raise ValueError(
            f"Preprocessing returned unexpected model input shape {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Preprocessed MRI contains non-finite values.")
    if isinstance(image_data, MetaTensor):
        affine = np.asarray(image_data.affine.detach().cpu(), dtype=np.float64)
    else:
        affine = reference_affine
    return PreparedVolume(
        image=np.ascontiguousarray(array),
        affine=affine,
        source_image=reference,
    )


def _center_crop_or_pad_batch(
    images: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    height, width = images.shape[-2:]
    target_height, target_width = size
    top = max(0, (height - target_height) // 2)
    left = max(0, (width - target_width) // 2)
    cropped = images[..., top : top + target_height, left : left + target_width]
    pad_height = target_height - cropped.shape[-2]
    pad_width = target_width - cropped.shape[-1]
    if pad_height or pad_width:
        cropped = F.pad(
            cropped,
            (
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
            ),
        )
    return cropped


def predict_2d_volume(
    model: AttentionUNet2DFourChannel,
    image: np.ndarray,
    image_size: tuple[int, int],
    device: torch.device,
    slice_batch_size: int = 4,
) -> np.ndarray:
    channels, height, width, depth = image.shape
    if channels != 4:
        raise ValueError(f"2D model expects four modalities, got {channels}.")
    slices = torch.from_numpy(np.transpose(image, (3, 0, 1, 2)))
    output = np.zeros((height, width, depth), dtype=np.uint8)
    for start in range(0, depth, slice_batch_size):
        batch_end = min(start + slice_batch_size, depth)
        batch = _center_crop_or_pad_batch(
            slices[start:batch_end], image_size
        ).to(device)
        logits = model(batch)
        classes = logits.argmax(dim=1).cpu().numpy()
        for offset, training_mask in enumerate(classes):
            source_z = start + offset
            crop_height = min(height, image_size[0])
            crop_width = min(width, image_size[1])
            source_top = max(0, (height - image_size[0]) // 2)
            source_left = max(0, (width - image_size[1]) // 2)
            padded_top = max(0, (image_size[0] - height) // 2)
            padded_left = max(0, (image_size[1] - width) // 2)
            crop = training_mask[
                padded_top : padded_top + crop_height,
                padded_left : padded_left + crop_width,
            ]
            native = fourclass_to_brats(crop)
            output[
                source_top : source_top + crop_height,
                source_left : source_left + crop_width,
                source_z,
            ] = native
    return output


def resample_mask_to_source(
    mask: np.ndarray,
    model_affine: np.ndarray,
    source_image: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    from nibabel.processing import resample_from_to

    model_image = nib.Nifti1Image(
        np.asarray(mask, dtype=np.uint8), model_affine
    )
    resampled = resample_from_to(
        model_image,
        (source_image.shape[:3], source_image.affine),
        order=0,
        mode="constant",
        cval=0,
    )
    output = np.rint(np.asarray(resampled.dataobj)).astype(np.uint8)
    if not np.isin(output, (0, 1, 2, 4)).all():
        raise ValueError("Restored prediction contains invalid BraTS labels.")
    return output
