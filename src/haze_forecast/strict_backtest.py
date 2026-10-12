"""Generate event-isolated, expanding-window predictions for honest history views."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd

from haze_forecast.training import AFTSurvivalRegressor


class _Predictor(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> _Predictor: ...

    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


def _default_model() -> AFTSurvivalRegressor:
    return AFTSurvivalRegressor(seed=42, nthread=1)


def rolling_event_backtest(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    minimum_train_rows: int = 100,
    model_factory: Callable[[], _Predictor] = _default_model,
) -> pd.DataFrame:
    """Predict each completed event using only complete events that ended before it began."""
    required = {"timestamp", "event_key", "remaining_hours", *feature_names}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"strict backtest frame missing columns: {missing}")
    if minimum_train_rows < 1:
        raise ValueError("minimum_train_rows must be positive")

    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="raise")
    ordered = ordered.sort_values("timestamp").reset_index(drop=True)
    events = (
        ordered.groupby("event_key", sort=False)["timestamp"]
        .agg(event_start="min", event_end="max")
        .sort_values(["event_start", "event_end"])
    )
    completed_event_keys: list[str] = []
    predictions: list[pd.DataFrame] = []
    for event_key, event in events.iterrows():
        training = ordered.loc[ordered["event_key"].isin(completed_event_keys)]
        test = ordered.loc[ordered["event_key"].eq(event_key)].copy()
        if len(training) >= minimum_train_rows:
            model = model_factory().fit(training[feature_names], training["remaining_hours"])
            test["strict_prediction_hours"] = np.asarray(
                model.predict(test[feature_names]), dtype=float
            ).clip(min=0.0)
            test["train_rows"] = len(training)
            test["train_end_timestamp"] = training["timestamp"].max()
            predictions.append(test)
        completed_event_keys.append(event_key)
    if not predictions:
        return ordered.iloc[0:0].assign(
            strict_prediction_hours=pd.Series(dtype=float),
            train_rows=pd.Series(dtype="int64"),
            train_end_timestamp=pd.Series(dtype=ordered["timestamp"].dtype),
        )
    return pd.concat(predictions, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
