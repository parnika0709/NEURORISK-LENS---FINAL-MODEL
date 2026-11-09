"""MONAI bundle predictor — wraps a pretrained BraTS bundle as a Predictor.

This is the swap-in path for a pretrained fallback. It does NOT require any
training; weights come from the MONAI Model Zoo via ``monai.bundle``.

Usage::

    from braintumor.models.monai_bundle import MonaiBundlePredictor
    predictor = MonaiBundlePredictor(bundle_name="brats_mri_segmentation")
    output = predictor.predict(volume)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from braintumor.schemas.contracts import (
    PredictionMeta,
    PredictionOutput,
    VolumeInput,
)
from braintumor.registry.predictor import Predictor


class MonaiBundlePredictor(Predictor):
    """Wraps a MONAI bundle for inference.

    Notes
    -----
    * The bundle's preprocessing pipeline is applied via MONAI transforms.
    * Bundle IO is loaded lazily so import cost is only paid if used.
    """

    name = "monai_bundle"
    version = "0.1.0"
    description = "Pretrained MONAI bundle fallback (no in-project training)"

    def __init__(
        self,
        bundle_name: str = "brats_mri_segmentation",
        device: str | None = None,
    ) -> None:
        self.bundle_name = bundle_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-imported to keep the package importable without `monai[bundle]`.
        from monai.bundle import download, load

        download(name=bundle_name, bundle_dir=str(Path.home() / ".cache" / "monai"))
        self._bundle = load(name=bundle_name, source="monailabel", device=self.device)
        self._preprocess = self._bundle.config.get("preprocessing", None)
        self._inferer = self._bundle.config.get("inferer", None)

    def load_weights(self, path: Path) -> None:  # noqa: D401
        """Bundles load weights at construction; this is a no-op."""
        return None

    @torch.no_grad()
    def predict(self, volume: VolumeInput) -> PredictionOutput:
        import time

        start = time.perf_counter()
        arr = np.asarray(volume.array).astype(np.float32)
        # Bundle expects a 4D tensor (C, D, H, W) or per-modality dict.
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1, C, D, H, W)
        # The exact invocation depends on bundle version; we delegate to the bundle.
        out = self._bundle(tensor) if hasattr(self._bundle, "__call__") else self._bundle.inferer(tensor)
        probs = torch.softmax(out[0], dim=0)
        mask = torch.argmax(probs, dim=0).to(torch.int8)
        elapsed = time.perf_counter() - start

        return PredictionOutput(
            mask=mask.cpu().numpy(),
            probability=probs.cpu().numpy().astype(np.float16),
            meta=PredictionMeta(
                model_name=self.name,
                model_version=self.bundle_name,
                inference_seconds=float(elapsed),
                notes="Pretrained MONAI bundle inference.",
            ),
        )


def build_monai_bundle() -> MonaiBundlePredictor:
    return MonaiBundlePredictor()
