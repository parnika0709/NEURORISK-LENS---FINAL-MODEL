"""Patient-level data loading for the isolated four-channel 2D experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)
from torch.utils.data import Dataset

from braintumor.data.brats_loader import BratsCase, discover_cases
from braintumor.schemas.contracts import VolumeModality

MODALITIES_2D_FOUR_CHANNEL: tuple[VolumeModality, ...] = (
    VolumeModality.FLAIR,
    VolumeModality.T1,
    VolumeModality.T1CE,
    VolumeModality.T2,
)
NATIVE_LABELS = (0, 1, 2, 4)
TRAINING_LABELS = (0, 1, 2, 3)


def brats_native_to_training(labels: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Explicitly convert native BraTS labels {0,1,2,4} to classes {0,1,2,3}."""
    if torch.is_tensor(labels):
        rounded = labels.long()
        allowed = torch.tensor(NATIVE_LABELS, device=rounded.device)
        if not torch.isin(rounded, allowed).all():
            invalid = torch.unique(rounded[~torch.isin(rounded, allowed)]).tolist()
            raise ValueError(f"Unexpected native BraTS labels: {invalid}")
        mapped = torch.zeros_like(rounded)
        mapped[rounded == 1] = 1
        mapped[rounded == 2] = 2
        mapped[rounded == 4] = 3
        return mapped

    native = np.asarray(labels)
    if not np.isin(native, NATIVE_LABELS).all():
        invalid = np.unique(native[~np.isin(native, NATIVE_LABELS)]).tolist()
        raise ValueError(f"Unexpected native BraTS labels: {invalid}")
    mapped = np.zeros(native.shape, dtype=np.uint8)
    mapped[native == 1] = 1
    mapped[native == 2] = 2
    mapped[native == 4] = 3
    return mapped


def training_to_brats_native(labels: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert model classes {0,1,2,3} to native BraTS labels {0,1,2,4}."""
    if torch.is_tensor(labels):
        classes = labels.long()
        allowed = torch.tensor(TRAINING_LABELS, device=classes.device)
        if not torch.isin(classes, allowed).all():
            invalid = torch.unique(classes[~torch.isin(classes, allowed)]).tolist()
            raise ValueError(f"Unexpected model classes: {invalid}")
        mapped = torch.zeros_like(classes)
        mapped[classes == 1] = 1
        mapped[classes == 2] = 2
        mapped[classes == 3] = 4
        return mapped

    classes = np.asarray(labels)
    if not np.isin(classes, TRAINING_LABELS).all():
        invalid = np.unique(classes[~np.isin(classes, TRAINING_LABELS)]).tolist()
        raise ValueError(f"Unexpected model classes: {invalid}")
    mapped = np.zeros(classes.shape, dtype=np.uint8)
    mapped[classes == 1] = 1
    mapped[classes == 2] = 2
    mapped[classes == 3] = 4
    return mapped


def load_verified_patient_split(
    split_file: Path,
    data_root: Path,
    expected_train: int = 1001,
    expected_val: int = 250,
) -> tuple[list[BratsCase], list[BratsCase], str]:
    """Load and strictly verify the existing 80/20 split against discovered cases."""
    if not split_file.is_file():
        raise FileNotFoundError(f"Patient split not found: {split_file}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"BraTS data root not found: {data_root}")

    split_data = json.loads(split_file.read_text(encoding="utf-8"))
    train_ids = split_data.get("train")
    val_ids = split_data.get("val")
    test_ids = split_data.get("test")
    if not all(isinstance(ids, list) for ids in (train_ids, val_ids, test_ids)):
        raise ValueError("Split file must contain train, val, and test patient lists.")
    if len(train_ids) != expected_train or len(val_ids) != expected_val or test_ids:
        raise ValueError(
            "Split count mismatch: expected "
            f"{expected_train} train, {expected_val} validation, and 0 test patients; "
            f"found {len(train_ids)}, {len(val_ids)}, and {len(test_ids)}."
        )
    train_set, val_set = set(train_ids), set(val_ids)
    if len(train_set) != len(train_ids) or len(val_set) != len(val_ids):
        raise ValueError("Duplicate patient IDs exist within a split.")
    if train_set & val_set:
        raise ValueError("Patient leakage detected between train and validation.")

    cases = discover_cases(data_root)
    cases_by_id = {case.patient_id: case for case in cases}
    if len(cases_by_id) != len(cases):
        raise ValueError("Duplicate discovered patient IDs.")
    split_ids = train_set | val_set
    if split_ids != set(cases_by_id):
        missing = sorted(split_ids - set(cases_by_id))[:5]
        unassigned = sorted(set(cases_by_id) - split_ids)[:5]
        raise ValueError(
            f"Split does not exactly cover discovered patients; "
            f"missing={missing}, unassigned={unassigned}."
        )

    for patient_id, case in cases_by_id.items():
        missing_modalities = [
            modality.value
            for modality in MODALITIES_2D_FOUR_CHANNEL
            if modality not in case.modality_paths
            or not case.modality_paths[modality].is_file()
        ]
        if missing_modalities or case.seg_path is None or not case.seg_path.is_file():
            raise ValueError(
                f"{patient_id} is missing data: modalities={missing_modalities}, "
                f"segmentation={case.seg_path}."
            )

    canonical = json.dumps(
        {"train": train_ids, "val": val_ids, "test": test_ids},
        separators=(",", ":"),
    ).encode("utf-8")
    split_digest = hashlib.sha256(canonical).hexdigest()
    return (
        [cases_by_id[patient_id] for patient_id in train_ids],
        [cases_by_id[patient_id] for patient_id in val_ids],
        split_digest,
    )


def build_2d_volume_transform(include_label: bool = True) -> Compose:
    """Match the 3D pipeline's orientation, spacing, scaling, and normalization."""
    keys = ["image", "label"] if include_label else ["image"]
    spacing_modes = ("bilinear", "nearest") if include_label else "bilinear"
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=(1.0, 1.0, 1.0),
                mode=spacing_modes,
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=1.0,
                upper=99.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ]
    )


def center_crop_or_pad(
    image: torch.Tensor, label: torch.Tensor, size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center crop/pad a registered image and label slice to the configured size."""
    height, width = image.shape[-2:]
    target_height, target_width = size
    top = max(0, (height - target_height) // 2)
    left = max(0, (width - target_width) // 2)
    image = image[..., top : top + target_height, left : left + target_width]
    label = label[..., top : top + target_height, left : left + target_width]
    pad_height = target_height - image.shape[-2]
    pad_width = target_width - image.shape[-1]
    if pad_height or pad_width:
        image = F.pad(
            image,
            (
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
            ),
        )
        label = F.pad(
            label,
            (
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
            ),
        )
    return image, label


class Brats2DFourChannelDataset(Dataset[dict[str, torch.Tensor]]):
    """One item per patient; training samples slices after loading that patient."""

    def __init__(
        self,
        cases: list[BratsCase],
        image_size: tuple[int, int] = (240, 240),
        slices_per_patient: int = 4,
        foreground_fraction: float = 0.75,
        training: bool = True,
        seed: int = 42,
    ) -> None:
        if not cases:
            raise ValueError("Dataset must contain at least one patient.")
        if slices_per_patient < 1:
            raise ValueError("slices_per_patient must be positive.")
        if not 0.0 <= foreground_fraction <= 1.0:
            raise ValueError("foreground_fraction must be between 0 and 1.")
        self.cases = cases
        self.image_size = image_size
        self.slices_per_patient = slices_per_patient
        self.foreground_fraction = foreground_fraction
        self.training = training
        self.seed = seed
        self.transform = build_2d_volume_transform()
        for case in cases:
            missing = [
                modality.value
                for modality in MODALITIES_2D_FOUR_CHANNEL
                if modality not in case.modality_paths
                or not case.modality_paths[modality].is_file()
            ]
            if missing or case.seg_path is None or not case.seg_path.is_file():
                raise ValueError(
                    f"{case.patient_id} is missing modalities {missing} "
                    "or a segmentation."
                )

    def __len__(self) -> int:
        return len(self.cases)

    @staticmethod
    def _case_item(case: BratsCase) -> dict[str, Any]:
        return {
            "image": [
                str(case.modality_paths[modality])
                for modality in MODALITIES_2D_FOUR_CHANNEL
            ],
            "label": str(case.seg_path),
        }

    def load_volume(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized (4,H,W,D) image and contiguous (H,W,D) labels."""
        case = self.cases[index]
        transformed = self.transform(self._case_item(case))
        image = torch.as_tensor(transformed["image"], dtype=torch.float32)
        label = torch.as_tensor(transformed["label"])
        if label.ndim == 4 and label.shape[0] == 1:
            label = label.squeeze(0)
        if image.ndim != 4 or image.shape[0] != 4 or label.ndim != 3:
            raise ValueError(
                f"{case.patient_id}: expected image (4,H,W,D) and label (H,W,D), "
                f"got {tuple(image.shape)} and {tuple(label.shape)}."
            )
        if image.shape[1:] != label.shape:
            raise ValueError(
                f"{case.patient_id}: image/label shapes differ: "
                f"{tuple(image.shape[1:])} versus {tuple(label.shape)}."
            )
        if not torch.isfinite(image).all():
            raise ValueError(f"{case.patient_id}: MRI contains non-finite values.")
        mapped_label = brats_native_to_training(label)
        assert torch.is_tensor(mapped_label)
        return image, mapped_label

    def _center_slice(
        self, image: torch.Tensor, label: torch.Tensor, slice_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_slice = image[..., slice_index]
        label_slice = label[..., slice_index]
        return center_crop_or_pad(image_slice, label_slice, self.image_size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, label = self.load_volume(index)
        depth = label.shape[-1]
        if self.training:
            foreground = torch.nonzero((label > 0).any(dim=(0, 1))).flatten()
            n_foreground = min(
                self.slices_per_patient,
                int(round(self.slices_per_patient * self.foreground_fraction)),
            )
            selected: list[int] = []
            if foreground.numel() and n_foreground:
                order = torch.randperm(foreground.numel())[:n_foreground]
                selected.extend(int(value) for value in foreground[order])
            remaining = self.slices_per_patient - len(selected)
            if remaining:
                order = torch.randperm(depth)
                selected_set = set(selected)
                selected.extend(
                    int(value)
                    for value in order
                    if int(value) not in selected_set
                )
                while len(selected) < self.slices_per_patient:
                    selected.append(int(torch.randint(depth, (1,)).item()))
            slices = torch.tensor(selected[: self.slices_per_patient])
        else:
            slices = torch.arange(depth)

        images: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for slice_index in slices.tolist():
            image_slice, label_slice = self._center_slice(
                image, label, slice_index
            )
            if self.training:
                if torch.rand(()) < 0.5:
                    image_slice = image_slice.flip(-1)
                    label_slice = label_slice.flip(-1)
                if torch.rand(()) < 0.5:
                    image_slice = image_slice.flip(-2)
                    label_slice = label_slice.flip(-2)
                turns = int(torch.randint(4, (1,)).item())
                image_slice = torch.rot90(image_slice, turns, (-2, -1))
                label_slice = torch.rot90(label_slice, turns, (-2, -1))
                if torch.rand(()) < 0.3:
                    scale = 1.0 + (torch.rand(4, 1, 1) - 0.5) * 0.2
                    image_slice = image_slice * scale
                if torch.rand(()) < 0.15:
                    image_slice = image_slice + torch.randn_like(image_slice) * 0.01
            images.append(image_slice.contiguous())
            labels.append(label_slice.long().contiguous())

        return {
            "image": torch.stack(images),
            "label": torch.stack(labels),
        }
