"""Discrete-time hazard survival model using only the current prediction snapshot."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from haze_forecast.features import NON_FEATURE_COLUMNS


def _validate_feature_names(frame: pd.DataFrame, feature_names: Sequence[str]) -> list[str]:
    if isinstance(feature_names, str) or not feature_names:
        raise ValueError("hazard feature names must be non-empty")
    names = list(feature_names)
    if len(names) != len(set(names)) or any(not isinstance(name, str) for name in names):
        raise ValueError("hazard feature names must be unique strings")
    forbidden = set(names).intersection(NON_FEATURE_COLUMNS)
    if forbidden:
        raise ValueError(f"hazard feature names include non-feature columns: {sorted(forbidden)}")
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"hazard feature columns are missing: {sorted(missing)}")
    return names


def _numeric_feature_matrix(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    try:
        matrix = frame[feature_names].astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("hazard features must be float-convertible") from error
    if np.isinf(matrix.to_numpy()).any():
        raise ValueError("hazard features must not contain infinite values")
    return matrix


def expand_event_hours(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Expand each current-time snapshot into at-risk hours through its observed recovery."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("hazard expansion requires a pandas DataFrame")  # noqa: TRY004
    if "remaining_hours" not in frame:
        raise ValueError("hazard expansion requires remaining_hours")
    names = _validate_feature_names(frame, feature_names)
    matrix = _numeric_feature_matrix(frame, names)
    try:
        remaining = pd.to_numeric(frame["remaining_hours"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("hazard remaining_hours must be numeric") from error
    if not np.isfinite(remaining).all() or (remaining <= 0).any():
        raise ValueError("hazard remaining_hours must be positive and finite")
    if not np.allclose(remaining, np.rint(remaining), rtol=0.0, atol=1e-9):
        raise ValueError("hazard remaining_hours must use integer hourly labels")

    hour_counts = remaining.astype(int)
    repeated_rows = np.repeat(np.arange(len(matrix)), hour_counts)
    starts = np.repeat(np.cumsum(hour_counts) - hour_counts, hour_counts)
    horizons = np.arange(hour_counts.sum(), dtype=float) - starts + 1.0
    expanded = matrix.iloc[repeated_rows].reset_index(drop=True)
    expanded["horizon_hours"] = horizons
    expanded["event_ends_in_horizon"] = (horizons == np.repeat(hour_counts, hour_counts)).astype(int)
    return expanded


class DiscreteHazardRegressor:
    """Estimate median recovery time from discrete conditional recovery risks."""

    def __init__(self, max_horizon_hours: int | None = None, seed: int = 42) -> None:
        if max_horizon_hours is not None and (
            isinstance(max_horizon_hours, bool) or max_horizon_hours < 1
        ):
            raise ValueError("max_horizon_hours must be a positive integer or None")
        self.max_horizon_hours = max_horizon_hours
        self.seed = seed
        self.feature_names: list[str] = []
        self.model: LGBMClassifier | None = None

    def fit(self, features: pd.DataFrame, remaining_hours: pd.Series) -> DiscreteHazardRegressor:
        if not isinstance(features, pd.DataFrame) or not isinstance(remaining_hours, pd.Series):
            raise ValueError("hazard fit requires a DataFrame and Series")  # noqa: TRY004
        if features.empty or len(features) != len(remaining_hours) or not features.index.equals(remaining_hours.index):
            raise ValueError("hazard features and target must be non-empty and index-aligned")
        self.feature_names = list(features.columns)
        training = features.copy()
        training["remaining_hours"] = remaining_hours
        expanded = expand_event_hours(training, self.feature_names)
        observed_maximum = int(pd.to_numeric(remaining_hours, errors="raise").max())
        if self.max_horizon_hours is None:
            self.max_horizon_hours = observed_maximum
        self.model = LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=40,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=self.seed,
            n_jobs=1,
            verbosity=-1,
        )
        self.model.fit(expanded[[*self.feature_names, "horizon_hours"]], expanded["event_ends_in_horizon"])
        return self

    def _prediction_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or not self.feature_names or self.max_horizon_hours is None:
            raise ValueError("discrete hazard model must be fitted before prediction")
        if not isinstance(features, pd.DataFrame):
            raise ValueError("hazard prediction requires a pandas DataFrame")  # noqa: TRY004
        if not features.columns.equals(pd.Index(self.feature_names)):
            raise ValueError("hazard prediction features must match fitted feature columns in order")
        matrix = _numeric_feature_matrix(features, self.feature_names)
        repeated = matrix.loc[matrix.index.repeat(self.max_horizon_hours)].reset_index(drop=True)
        repeated["horizon_hours"] = np.tile(
            np.arange(1, self.max_horizon_hours + 1, dtype=float), len(matrix)
        )
        return repeated

    def _predict_hazards(self, features: pd.DataFrame) -> np.ndarray:
        matrix = self._prediction_matrix(features)
        probabilities = np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError("hazard probabilities must be finite")
        return np.clip(probabilities, 1e-6, 1 - 1e-6).reshape(len(features), self.max_horizon_hours)

    def predict_with_censoring(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        hazards = self._predict_hazards(features)
        survival = np.cumprod(1 - hazards, axis=1)
        reached_median = 1 - survival >= 0.5
        has_median = reached_median.any(axis=1)
        first_hour = reached_median.argmax(axis=1).astype(float) + 1.0
        predictions = np.where(has_median, first_hour, float(self.max_horizon_hours))
        return predictions, ~has_median

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return a non-negative median recovery-time prediction in hours."""
        prediction, _ = self.predict_with_censoring(features)
        return prediction
