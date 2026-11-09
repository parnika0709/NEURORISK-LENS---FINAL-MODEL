"""Predictor interface — the model-swappability contract.

Every segmentation model (2D Attention U-Net, nnU-Net, 3D U-Net, MONAI bundle)
implements this interface. The backend only ever calls ``predict()`` and reads
``meta``. New models drop in without API or UI changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from braintumor.schemas.contracts import PredictionOutput, VolumeInput


class Predictor(ABC):
    """Abstract segmentation predictor.

    Implementations must be deterministic given identical inputs and weights.
    They must be safe to invoke from a FastAPI worker thread.
    """

    name: str = "base"
    version: str = "0.0.0"
    description: str = ""

    @abstractmethod
    def predict(self, volume: VolumeInput) -> PredictionOutput:
        """Run inference on a single volume and return a PredictionOutput."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, path: Path) -> None:
        """Load model weights from disk."""
        raise NotImplementedError

    def info(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
        }
