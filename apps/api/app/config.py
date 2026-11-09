"""API configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Settings:
    """Runtime settings, sourced from environment variables with defaults."""

    storage_root: Path = Path(
        os.getenv("BT_STORAGE_ROOT", str(PROJECT_ROOT / "storage"))
    )
    upload_dir: Path = Path(
        os.getenv("BT_UPLOAD_DIR", str(PROJECT_ROOT / "storage" / "uploads"))
    )
    results_dir: Path = Path(
        os.getenv("BT_RESULTS_DIR", str(PROJECT_ROOT / "storage" / "results"))
    )
    reports_dir: Path = Path(
        os.getenv("BT_REPORTS_DIR", str(PROJECT_ROOT / "storage" / "reports"))
    )
    db_path: Path = Path(
        os.getenv("BT_DB_PATH", str(PROJECT_ROOT / "storage" / "app.db"))
    )

    active_predictor: str = os.getenv("BT_PREDICTOR", "attn_unet_2d")
    model_weights_path: Path | None = (
        Path(os.getenv("BT_WEIGHTS_PATH"))
        if os.getenv("BT_WEIGHTS_PATH")
        else PROJECT_ROOT / "runs" / "exp" / "best.pt"
    )
    model_weights_path_2d_four_channel: Path = Path(
        os.getenv(
            "BT_WEIGHTS_PATH_2D_FOUR_CHANNEL",
            str(PROJECT_ROOT / "runs" / "2d_attention_unet" / "checkpoints" / "best.pt"),
        )
    )
    model_weights_path_3d: Path | None = (
        Path(os.getenv("BT_WEIGHTS_PATH_3D"))
        if os.getenv("BT_WEIGHTS_PATH_3D")
        else PROJECT_ROOT
        / "outputs"
        / "3d_unet"
        / "full_80_20"
        / "checkpoints"
        / "best.pt"
    )

    max_upload_mb: int = int(os.getenv("BT_MAX_UPLOAD_MB", "500"))
    max_volume_voxels: int = int(os.getenv("BT_MAX_VOLUME_VOXELS", str(256**3)))
    upload_purge_days: int = int(os.getenv("BT_UPLOAD_PURGE_DAYS", "7"))

    cors_origins: list[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "BT_CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        ]
    )

    title: str = "Brain Tumor MRI Analysis API"
    description: str = (
        "Research/educational prototype for brain MRI tumor segmentation, "
        "quantitative analysis, and visualization. NOT a medical device. "
        "NOT validated for clinical or surgical use."
    )


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.results_dir.mkdir(parents=True, exist_ok=True)
settings.reports_dir.mkdir(parents=True, exist_ok=True)
settings.storage_root.mkdir(parents=True, exist_ok=True)
