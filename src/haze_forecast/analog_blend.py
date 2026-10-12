"""Analog retrieval used as a conservative correction for a known input regime."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MissingRegimeAnalogBlend:
    """Blend a base prediction with historical analogs only for a selected regime."""

    feature_names: list[str]
    neighbors: int = 5
    blend_weight: float = 0.1
    _median: pd.Series = field(init=False, repr=False)
    _scale: pd.Series = field(init=False, repr=False)
    _reference: np.ndarray = field(init=False, repr=False)
    _target: np.ndarray = field(init=False, repr=False)

    def fit(self, features: pd.DataFrame, remaining_hours: pd.Series) -> MissingRegimeAnalogBlend:
        """Store standardized, completed historical rows for future analog lookup."""
        if self.neighbors < 1:
            raise ValueError("neighbors must be positive")
        if not 0.0 <= self.blend_weight <= 1.0:
            raise ValueError("blend_weight must be between zero and one")
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise ValueError(f"analog features missing columns: {sorted(missing)}")
        if len(features) != len(remaining_hours) or features.empty:
            raise ValueError("analog reference features and targets must be non-empty and aligned")

        selected = features.loc[:, self.feature_names].apply(pd.to_numeric, errors="coerce")
        self._median = selected.median().fillna(0.0)
        iqr = selected.quantile(0.75) - selected.quantile(0.25)
        self._scale = iqr.mask(iqr.le(0) | iqr.isna(), 1.0)
        self._reference = ((selected.fillna(self._median) - self._median) / self._scale).to_numpy(float)
        self._target = pd.to_numeric(remaining_hours, errors="raise").to_numpy(float)
        return self

    def predict(
        self, base_prediction: np.ndarray, features: pd.DataFrame, apply_blend: np.ndarray
    ) -> np.ndarray:
        """Return the base prediction outside the selected regime and a guarded blend within it."""
        if not hasattr(self, "_reference"):
            raise ValueError("analog blend must be fitted before prediction")
        base = np.asarray(base_prediction, dtype=float)
        mask = np.asarray(apply_blend, dtype=bool)
        if base.ndim != 1 or mask.shape != base.shape:
            raise ValueError("apply_blend must have one entry per base prediction")
        if len(features) != len(base):
            raise ValueError("analog features must align with base predictions")
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise ValueError(f"analog features missing columns: {sorted(missing)}")

        selected = features.loc[:, self.feature_names].apply(pd.to_numeric, errors="coerce")
        query = ((selected.fillna(self._median) - self._median) / self._scale).to_numpy(float)
        distances = np.mean((query[:, None, :] - self._reference[None, :, :]) ** 2, axis=2)
        count = min(self.neighbors, len(self._target))
        nearest = np.argpartition(distances, count - 1, axis=1)[:, :count]
        neighbor_distances = np.take_along_axis(distances, nearest, axis=1)
        neighbor_targets = self._target[nearest]
        weights = 1.0 / (neighbor_distances + 1e-4)
        analog = (neighbor_targets * weights).sum(axis=1) / weights.sum(axis=1)
        blended = (1.0 - self.blend_weight) * base + self.blend_weight * analog
        return np.where(mask, blended, base).clip(min=0.0)
