"""Leakage-safe post-processing for remaining-time regression predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MedianResidualCalibrator:
    """A robust additive calibration fitted only on pre-test predictions."""

    offset_hours: float

    @classmethod
    def fit(cls, actual: pd.Series, predicted: pd.Series) -> MedianResidualCalibrator:
        if len(actual) != len(predicted) or len(actual) == 0:
            raise ValueError("calibration requires equally sized non-empty actual and prediction series")
        residuals = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
        return cls(offset_hours=float(np.median(residuals)))

    def predict(self, predicted: pd.Series) -> pd.Series:
        return pd.Series(predicted.to_numpy(dtype=float) + self.offset_hours, index=predicted.index).clip(
            lower=0.0
        )


def prediction_bias_summary(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Summarise signed bias without conflating it with MAE/RMSE."""
    residual = predicted.to_numpy(dtype=float) - actual.to_numpy(dtype=float)
    return {
        "median_actual_hours": float(np.median(actual.to_numpy(dtype=float))),
        "median_prediction_hours": float(np.median(predicted.to_numpy(dtype=float))),
        "median_signed_error_hours": float(np.median(residual)),
        "mean_signed_error_hours": float(np.mean(residual)),
    }


def remaining_time_segment_metrics(actual: pd.Series, predicted: pd.Series) -> list[dict[str, float | int | str]]:
    """Report errors in operationally useful remaining-time bands."""
    boundaries = (0.0, 24.0, 48.0, 96.0, float("inf"))
    labels = ("0-24h", "24-48h", "48-96h", "96h+")
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    records: list[dict[str, float | int | str]] = []
    for lower, upper, label in zip(boundaries[:-1], boundaries[1:], labels, strict=True):
        mask = (actual_values >= lower) & (actual_values < upper)
        if not mask.any():
            continue
        error = predicted_values[mask] - actual_values[mask]
        metrics = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.mean(np.square(error)) ** 0.5),
        }
        records.append({"segment": label, "rows": int(mask.sum()), **metrics})
    return records
