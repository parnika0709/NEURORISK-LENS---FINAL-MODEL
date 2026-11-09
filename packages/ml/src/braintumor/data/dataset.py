"""PyTorch Dataset for BraTS volumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monai.data import Dataset
from braintumor.data.brats_loader import BratsCase
from braintumor.data.preprocessing import train_transforms, val_transforms
from braintumor.schemas.contracts import VolumeModality

class BratsDataset(Dataset):
    """Dataset for BraTS volumes.

    Each item in the dataset is a dictionary containing:
      - 'image': paths to the modality files.
      - 'label': path to the segmentation file.
    """

    def __init__(
        self,
        cases: list[BratsCase],
        modalities: list[VolumeModality],
        transform: Any = None,
    ) -> None:
        self.modalities = modalities
        self.transform = transform

        # Create a list of dictionaries as required by MONAI transforms
        self.data = []
        for case in cases:
            # Map modality enum to filenames (keys used in preprocessing.py)
            img_keys = [m.value for m in modalities]

            item = {k: str(case.modality_paths[m]) for k, m in zip(img_keys, modalities)}
            if case.seg_path:
                item["label"] = str(case.seg_path)

            self.data.append(item)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self.data[index]
        if self.transform:
            return self.transform(data)
        return data
