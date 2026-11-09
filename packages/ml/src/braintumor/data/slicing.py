"""2D slice dataset for BraTS MRI segmentation."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from braintumor.data.dataset import BratsDataset
from braintumor.data.preprocessing import brats_label_to_4class


class SlicingDataset(Dataset):
    """Extract X-Y slices along Z from (C,X,Y,Z) volumes."""

    def __init__(
        self,
        base_dataset: BratsDataset,
        slice_idx: int | None = None,
        *,
        training: bool = False,
        samples_per_volume: int = 8,
        slices_per_volume: int = 8,
        foreground_probability: float = 0.7,
        seed: int = 42,
    ) -> None:
        self.base_dataset = base_dataset
        self.slice_idx = slice_idx
        self.training = training
        self.samples_per_volume = samples_per_volume
        self.slices_per_volume = slices_per_volume
        self.foreground_probability = foreground_probability
        self.seed = seed

        if samples_per_volume < 1 or slices_per_volume < 1:
            raise ValueError("Slice sample counts must be at least 1")

        if not 0.0 <= foreground_probability <= 1.0:
            raise ValueError("foreground_probability must be between 0 and 1")

        self._cache: dict[int, dict[str, Any]] = {}

        sample_count = samples_per_volume if training else slices_per_volume

        self.samples = [
            (patient_idx, sample_idx)
            for patient_idx in range(len(base_dataset))
            for sample_idx in range(sample_count)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_volume(
        self, patient_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if patient_idx not in self._cache:
            item = self.base_dataset[patient_idx]

            image_channels = []

            for key, value in item.items():
                if key == "label":
                    continue

                channel = torch.as_tensor(value).float()

                if channel.ndim == 4 and channel.shape[0] == 1:
                    channel = channel.squeeze(0)

                if channel.ndim != 3:
                    raise ValueError(
                        f"Expected {key} shape (X,Y,Z), got {tuple(channel.shape)}"
                    )

                image_channels.append(channel)

            if not image_channels:
                raise ValueError("No MRI modality channels were found")

            image = torch.stack(image_channels, dim=0)

            label = torch.as_tensor(item["label"])

            if label.ndim == 4 and label.shape[0] == 1:
                label = label.squeeze(0)

            if label.ndim != 3:
                raise ValueError(
                    f"Expected label shape (X,Y,Z), got {tuple(label.shape)}"
                )

            if tuple(image.shape[1:]) != tuple(label.shape):
                raise ValueError(
                    f"Image shape {tuple(image.shape[1:])} does not match "
                    f"label shape {tuple(label.shape)}"
                )

            self._cache[patient_idx] = {
                "image": image,
                "label": label,
            }

        cached = self._cache[patient_idx]
        return cached["image"], cached["label"]

    @staticmethod
    def _foreground_indices(label: torch.Tensor) -> torch.Tensor:
        """Find Z-axis slices containing tumour."""
        return torch.where((label > 0).any(dim=(0, 1)))[0]

    def _choose_training_slice(self, label: torch.Tensor) -> int:
        depth = label.shape[2]

        if self.slice_idx is not None:
            if not 0 <= self.slice_idx < depth:
                raise IndexError(
                    f"slice_idx {self.slice_idx} outside depth {depth}"
                )
            return self.slice_idx

        foreground = self._foreground_indices(label)

        use_foreground = (
            foreground.numel() > 0
            and torch.rand(1).item() < self.foreground_probability
        )

        if use_foreground:
            # Count tumour pixels in each foreground slice.
            foreground_counts = (
                label[:, :, foreground] > 0
            ).sum(dim=(0, 1)).float()

            # Give larger tumour regions more sampling weight.
            weights = torch.sqrt(foreground_counts.clamp_min(1))

            selected = torch.multinomial(
                weights,
                num_samples=1,
            ).item()

            return int(foreground[selected].item())

        return int(torch.randint(0, depth, (1,)).item())

    def _choose_validation_slice(
        self, label: torch.Tensor, sample_number: int
    ) -> int:
        """Select deterministic, evenly spaced slices across the full volume."""

        depth = label.shape[2]

        if self.slice_idx is not None:
            if not 0 <= self.slice_idx < depth:
                raise IndexError(
                    f"slice_idx {self.slice_idx} outside depth {depth}"
                )
            return self.slice_idx

        if not 0 <= sample_number < self.slices_per_volume:
            raise IndexError(
                f"sample_number {sample_number} outside validation sample range"
            )

        if self.slices_per_volume == 1:
            return depth // 2

        positions = torch.linspace(
            0,
            depth - 1,
            steps=self.slices_per_volume,
        ).round().long()

        return int(positions[sample_number].item())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        patient_idx, sample_number = self.samples[index]

        image, label = self._load_volume(patient_idx)

        if self.training:
            z = self._choose_training_slice(label)
        else:
            z = self._choose_validation_slice(label, sample_number)

        # Preserve the X-Y plane at the selected Z position.
        image_slice = image[:, :, :, z]
        label_slice = label[:, :, z]

        mapped_label = brats_label_to_4class(
            label_slice.cpu().numpy()
        )

        return {
            "image": image_slice.contiguous(),
            "label": torch.from_numpy(mapped_label).long(),
        }

