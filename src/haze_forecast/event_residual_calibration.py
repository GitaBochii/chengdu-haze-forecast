"""State-grouped residual correction for already-fitted dissipation models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

STATE_COLUMNS = (
    "ongoing_pollution_hours",
    "aqi_change_3",
    "relative_humidity_pct",
)


class _Regressor(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> _Regressor: ...

    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


def event_state_keys(features: pd.DataFrame) -> pd.Series:
    """Create current-state labels without consulting future observations."""
    missing = set(STATE_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(f"state features missing columns: {sorted(missing)}")

    age = np.where(
        pd.to_numeric(features["ongoing_pollution_hours"], errors="coerce").fillna(0).lt(6),
        "early",
        "late",
    )
    change = pd.to_numeric(features["aqi_change_3"], errors="coerce").fillna(0)
    trend = np.select([change.le(-10), change.ge(5)], ["falling", "rising"], default="flat")
    humid = np.where(
        pd.to_numeric(features["relative_humidity_pct"], errors="coerce").fillna(60).ge(60),
        "humid",
        "dry",
    )
    return pd.Series(age + "_" + trend + "_" + humid, index=features.index)


@dataclass
class EventResidualCalibrator:
    """Apply shrunken, state-specific median residuals to a base predictor."""

    minimum_state_rows: int = 30
    shrinkage_rows: int = 60
    _global_residual: float = field(init=False, repr=False)
    _state_residuals: dict[str, tuple[float, int]] = field(init=False, repr=False)

    def fit(
        self, features: pd.DataFrame, base_prediction: np.ndarray, actual: np.ndarray
    ) -> EventResidualCalibrator:
        """Fit residual summaries from historical out-of-fold predictions."""
        base = np.asarray(base_prediction, dtype=float)
        observed = np.asarray(actual, dtype=float)
        if len(features) == 0 or base.shape != (len(features),) or observed.shape != (len(features),):
            raise ValueError("features, base_prediction, and actual must be non-empty and aligned")
        if self.minimum_state_rows < 1 or self.shrinkage_rows < 0:
            raise ValueError("state row thresholds must be non-negative, with minimum at least one")

        residual = observed - base
        self._global_residual = float(np.median(residual))
        grouped = pd.DataFrame({"state": event_state_keys(features), "residual": residual}).groupby(
            "state", sort=False
        )["residual"]
        self._state_residuals = {
            str(state): (float(values.median()), int(values.size)) for state, values in grouped
        }
        return self

    def predict(self, base_prediction: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        """Correct predictions, falling back to the global residual for sparse states."""
        if not hasattr(self, "_state_residuals"):
            raise ValueError("calibrator must be fitted before prediction")
        base = np.asarray(base_prediction, dtype=float)
        if base.shape != (len(features),):
            raise ValueError("base_prediction must align with features")

        corrections: list[float] = []
        for state in event_state_keys(features):
            state_residual, count = self._state_residuals.get(
                str(state), (self._global_residual, 0)
            )
            weight = (
                count / (count + self.shrinkage_rows)
                if count >= self.minimum_state_rows
                else 0.0
            )
            corrections.append(weight * state_residual + (1.0 - weight) * self._global_residual)
        return (base + np.asarray(corrections)).clip(min=0.0)


def out_of_fold_event_predictions(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    model_factory: Callable[[], _Regressor],
    minimum_train_rows: int = 100,
    fold_count: int | None = None,
) -> pd.DataFrame:
    """Predict future event batches from preceding completed events only."""
    required = {"timestamp", "event_key", "remaining_hours"} | set(feature_names)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"out-of-fold input missing columns: {sorted(missing)}")
    if minimum_train_rows < 1:
        raise ValueError("minimum_train_rows must be positive")
    if fold_count is not None and fold_count < 1:
        raise ValueError("fold_count must be positive when provided")

    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="raise")
    ordered = ordered.sort_values("timestamp")
    event_end = ordered.groupby("event_key", sort=False)["timestamp"].max().sort_values()
    output: list[pd.DataFrame] = []
    if fold_count is None:
        validation_batches = [[event_key] for event_key in event_end.index]
    else:
        event_batches = np.array_split(event_end.index.to_numpy(), fold_count + 1)
        validation_batches = [batch.tolist() for batch in event_batches[1:] if len(batch)]

    for event_keys in validation_batches:
        test = ordered[ordered["event_key"].isin(event_keys)].copy()
        test_start = test["timestamp"].min()
        completed_keys = event_end[event_end.lt(test_start)].index
        train = ordered[ordered["event_key"].isin(completed_keys)]
        if len(train) < minimum_train_rows:
            continue
        model = model_factory().fit(train[feature_names], train["remaining_hours"])
        test["base_prediction"] = np.asarray(model.predict(test[feature_names]), dtype=float).clip(
            min=0.0
        )
        test["residual_train_end_timestamp"] = train["timestamp"].max()
        output.append(test)

    if not output:
        return ordered.iloc[0:0].assign(
            base_prediction=pd.Series(dtype=float),
            residual_train_end_timestamp=pd.Series(dtype=ordered["timestamp"].dtype),
        )
    return pd.concat(output, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def select_candidate(
    selection_mae: dict[str, float], confirmation_mae: dict[str, float], *, baseline_name: str
) -> str | None:
    """Accept a 2024-selected candidate only after an independent improvement."""
    if baseline_name not in selection_mae or baseline_name not in confirmation_mae:
        raise ValueError("baseline must be present in selection and confirmation metrics")
    if set(selection_mae) != set(confirmation_mae):
        raise ValueError("selection and confirmation candidates must match")
    selected = min(selection_mae, key=selection_mae.__getitem__)
    if selected == baseline_name:
        return None
    if confirmation_mae[selected] >= confirmation_mae[baseline_name]:
        return None
    return selected
