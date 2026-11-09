"""Public API for the registry."""

from braintumor.registry.predictor import Predictor
from braintumor.registry.registry import PredictorRegistry, registry

__all__ = ["Predictor", "PredictorRegistry", "registry"]
