"""Leakage-safe direct pollutant paths and short-horizon recovery derivation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HORIZONS = (1, 3, 6, 12, 24)
POLLUTANTS = ("aqi", "pm2_5")


def add_future_targets(
    frame: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    """Add targets only when the complete interval to each target is hourly."""
    required = {"timestamp", *POLLUTANTS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"multihorizon frame missing columns: {sorted(missing)}")
    if not horizons or any(not isinstance(value, int) or value < 1 for value in horizons):
        raise ValueError("horizons must contain positive integer hours")

    result = frame.sort_values("timestamp").copy()
    timestamps = pd.to_datetime(result["timestamp"], errors="raise")
    for horizon in horizons:
        expected = timestamps + pd.Timedelta(horizon, unit="h")
        continuous = timestamps.shift(-horizon).eq(expected)
        for pollutant in POLLUTANTS:
            result[f"{pollutant}_target_h{horizon:03d}"] = result[pollutant].shift(-horizon).where(
                continuous
            )
    return result


def recovery_hour_from_curve(
    curve: np.ndarray, threshold: float = 100.0, stable_hours: int = 6
) -> float | None:
    """Return the start of a stable AQI run; curve[0] is forecast hour 1."""
    values = np.asarray(curve, dtype=float)
    if values.ndim != 1:
        raise ValueError("curve must be one-dimensional")
    if stable_hours < 1:
        raise ValueError("stable_hours must be positive")
    for start in range(len(values) - stable_hours + 1):
        window = values[start : start + stable_hours]
        if np.isfinite(window).all() and (window <= threshold).all():
            return float(start + 1)
    return None


class DirectCurveRegressor:
    """Fit one direct MAE model for each pollutant and forecast horizon."""

    def __init__(self, horizons: tuple[int, ...] = HORIZONS, seed: int = 42) -> None:
        if not horizons or any(not isinstance(value, int) or value < 1 for value in horizons):
            raise ValueError("horizons must contain positive integer hours")
        self.horizons = horizons
        self.seed = seed
        self.feature_names: list[str] = []
        self.models: dict[str, LGBMRegressor] = {}

    def fit(
        self,
        frame: pd.DataFrame,
        feature_names: list[str],
        *,
        training_cutoff: pd.Timestamp | None = None,
    ) -> DirectCurveRegressor:
        """Fit observed targets, optionally requiring target times before a cutoff."""
        if not feature_names or set(feature_names) - set(frame.columns):
            raise ValueError("curve features must be present and non-empty")
        timestamps = None
        if training_cutoff is not None:
            if "timestamp" not in frame:
                raise ValueError("training_cutoff requires timestamp column")
            timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
            training_cutoff = pd.Timestamp(training_cutoff)
        self.feature_names = list(feature_names)
        self.models = {}
        for pollutant in POLLUTANTS:
            for horizon in self.horizons:
                target = f"{pollutant}_target_h{horizon:03d}"
                if target not in frame:
                    raise ValueError(f"curve target is missing: {target}")
                observed = frame[target].notna()
                if timestamps is not None:
                    observed &= (timestamps + pd.Timedelta(hours=horizon)).lt(training_cutoff)
                rows = frame.loc[observed]
                if rows.empty:
                    raise ValueError(f"curve target has no observed rows: {target}")
                model = LGBMRegressor(
                    objective="regression_l1",
                    n_estimators=300,
                    learning_rate=0.04,
                    num_leaves=24,
                    min_child_samples=30,
                    random_state=self.seed,
                    n_jobs=1,
                    verbosity=-1,
                )
                model.fit(rows[self.feature_names], rows[target])
                # Keep only the trained trees: ten direct targets otherwise retain ten training datasets.
                model.booster_.free_dataset()
                self.models[f"{pollutant}_h{horizon:03d}"] = model
        return self

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return non-negative direct forecasts keyed by pollutant and horizon."""
        if not self.models:
            raise ValueError("curve model must be fitted before prediction")
        if set(self.feature_names) - set(frame.columns):
            raise ValueError("prediction frame is missing fitted curve features")
        return {
            name: np.asarray(model.predict(frame[self.feature_names]), dtype=float).clip(min=0.0)
            for name, model in self.models.items()
        }


def curve_recovery_or_fallback(aqi_curve: np.ndarray, fallback_hours: float) -> float:
    """Use a stable short-horizon recovery, otherwise retain the baseline duration."""
    recovery = recovery_hour_from_curve(aqi_curve)
    return float(fallback_hours) if recovery is None else recovery


def select_hybrid(selection_mae: dict[str, float], confirmation_mae: dict[str, float]) -> str | None:
    """Accept a 2024-selected curve variant only after a 2025 improvement."""
    if "aft" not in selection_mae or "aft" not in confirmation_mae:
        raise ValueError("selection and confirmation metrics require the aft baseline")
    if set(selection_mae) != set(confirmation_mae):
        raise ValueError("selection and confirmation variants must match")
    selected = min(selection_mae, key=selection_mae.__getitem__)
    if selected == "aft" or confirmation_mae[selected] >= confirmation_mae["aft"]:
        return None
    return selected
