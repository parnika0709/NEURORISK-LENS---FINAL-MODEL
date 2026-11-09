"""Predictor registry — name → Predictor instance.
"""

from __future__ import annotations

import threading
from typing import Callable

from braintumor.registry.predictor import Predictor


class PredictorRegistry:
    """Thread-safe singleton registry."""

    _instance: "PredictorRegistry | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "PredictorRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._predictors = {}
        return cls._instance

    def register(self, name: str, factory: Callable[[], Predictor]) -> None:
        if name in self._predictors:
            raise ValueError(f"Predictor '{name}' already registered.")
        self._predictors[name] = factory

    def get(self, name: str) -> Predictor:
        if name not in self._predictors:
            raise KeyError(
                f"Predictor '{name}' not registered. "
                f"Available: {sorted(self._predictors)}"
            )
        return self._predictors[name]()

    def list_names(self) -> list[str]:
        return sorted(self._predictors)


registry = PredictorRegistry()
