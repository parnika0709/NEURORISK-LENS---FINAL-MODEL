"""High-level inference pipeline.

Given a Predictor and a VolumeInput, produce the full pipeline output:
  - Segmentation mask
  - Probability map
  - Uncertainty map (MC-Dropout)
  - Saliency map (encoder Grad-CAM, optional)
  - Quantitative features

This is the function the FastAPI backend calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from braintumor.explainability.uncertainty import mc_dropout_predict
from braintumor.features.extract import extract_features
from braintumor.inference.postprocess import postprocess
from braintumor.registry.predictor import Predictor
from braintumor.schemas.contracts import (
    PredictionMeta,
    PredictionOutput,
    TumorFeaturesBundle,
    VolumeInput,
)


@dataclass
class PipelineOutput:
    prediction: PredictionOutput
    features: TumorFeaturesBundle


def run_pipeline(
    predictor: Predictor,
    volume: VolumeInput,
    *,
    postprocess_mask: bool = True,
    compute_uncertainty: bool = True,
    mc_passes: int = 5,
) -> PipelineOutput:
    """Run the full segmentation → postprocess → features pipeline.

    For MVP, we always run postprocess_mask=True and compute_uncertainty=True.
    """
    raw = predictor.predict(volume)
    mask = raw.mask

    if postprocess_mask:
        mask = postprocess(mask)

    uncertainty = None
    probability = raw.probability
    if compute_uncertainty:
        try:
            _, mean_prob, uncertainty_arr = mc_dropout_predict(
                predictor, volume, n_passes=mc_passes
            )
            uncertainty = uncertainty_arr
            # Use MC mean prob if available; else fall back to single-pass.
            if mean_prob is not None:
                probability = mean_prob  # type: ignore[assignment]
        except Exception:
            # MC-Dropout may fail if predictor doesn't expose internals.
            uncertainty = None

    prediction = PredictionOutput(
        mask=mask.astype(np.int8),
        probability=probability,
        uncertainty=uncertainty,
        saliency=None,
        meta=PredictionMeta(
            model_name=raw.meta.model_name,
            model_version=raw.meta.model_version,
            inference_seconds=raw.meta.inference_seconds,
            notes=(raw.meta.notes or "") + " [postprocessed]",
        ),
    )

    features = extract_features(
        mask=mask,
        volume=volume,
    )

    return PipelineOutput(prediction=prediction, features=features)
