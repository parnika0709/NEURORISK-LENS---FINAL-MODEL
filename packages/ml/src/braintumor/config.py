
"""ML configuration for the Brain Tumor analysis pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MLConfig:
    """Hyperparameters and paths for training and inference."""

    # Dataset Paths
    data_root: Path = Path(
        os.getenv(
            "BT_DATA_ROOT",
            r"C:\Users\mujis\Downloads\BraTS2021_Clean\BraTS2021_Training_Data\BraTS2021_Training_Data",
        )
    )
    split_file: Path = Path(
        os.getenv("BT_SPLIT_FILE", "data/splits/patients_split.json")
    )

    # Predictor Settings
    active_predictor: str = "attn_unet_2d"
    model_weights_path: Path | None = None

    # Model Hyperparameters
    in_channels: int = 3
    out_channels: int = 4
    model_channels: tuple[int, ...] = (32, 64, 128, 256, 512)
    model_strides: tuple[int, ...] = (2, 2, 2, 2)
    dropout: float = 0.2

    # Training Hyperparameters
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    seed: int = 42
    num_workers: int = 0
    device: str = (
        "cuda"
        if os.getenv("USE_CUDA", "true").lower() == "true"
        else "cpu"
    )

    # Preprocessing
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    image_size: tuple[int, int, int] = (128, 128, 128)

    # Paths for checkpoints
    checkpoint_dir: Path = Path("runs/exp")


config = MLConfig()
