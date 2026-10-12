"""Probability outputs calibrated from accepted rolling-validation residuals."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

ACCEPTED_REPLACEMENT_DECISION = "candidate_for_replacement_review"


def build_probability_output(
    predictions: pd.DataFrame, decision: Mapping[str, object]
) -> dict[str, object]:
    """Calibrate accepted rolling-validation predictions with empirical residuals."""
    if decision.get("decision") != ACCEPTED_REPLACEMENT_DECISION:
        raise ValueError("probability output requires an accepted MAE replacement decision")

    required_columns = {"fold", "actual_remaining_hours", "predicted_remaining_hours"}
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("rolling validation predictions must be a pandas DataFrame")
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"rolling validation predictions missing columns: {sorted(missing_columns)}")
    if predictions.empty:
        raise ValueError("rolling validation predictions must not be empty")
    if predictions["fold"].isna().any():
        raise ValueError("rolling validation predictions must identify every fold")
    if predictions["fold"].nunique() < 3:
        raise ValueError("rolling validation predictions must include at least three folds")

    result = predictions.copy()
    try:
        actual = pd.to_numeric(result["actual_remaining_hours"], errors="raise").to_numpy(dtype=float)
        predicted = pd.to_numeric(
            result["predicted_remaining_hours"], errors="raise"
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("rolling validation predictions must contain numeric hours") from error
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("rolling validation predictions must contain finite hours")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError("rolling validation predictions must contain non-negative hours")

    residuals = actual - predicted
    residual_quantiles = np.quantile(residuals, [0.1, 0.25, 0.5, 0.75, 0.9])
    low_80, low_50, median_residual, high_50, high_80 = residual_quantiles
    result["median_remaining_hours"] = np.maximum(0.0, predicted + median_residual)
    result["interval_50_low_hours"] = np.maximum(0.0, predicted + low_50)
    result["interval_50_high_hours"] = predicted + high_50
    result["interval_80_low_hours"] = np.maximum(0.0, predicted + low_80)
    result["interval_80_high_hours"] = predicted + high_80
    result["probability_recovery_within_24h"] = _empirical_recovery_probability(
        predicted, residuals, horizon_hours=24.0
    )
    result["probability_recovery_within_48h"] = _empirical_recovery_probability(
        predicted, residuals, horizon_hours=48.0
    )

    covered_50 = (actual >= result["interval_50_low_hours"]) & (
        actual <= result["interval_50_high_hours"]
    )
    covered_80 = (actual >= result["interval_80_low_hours"]) & (
        actual <= result["interval_80_high_hours"]
    )
    return {
        "predictions": result,
        "coverage": {
            "rows": len(result),
            "interval_50": float(np.mean(covered_50)),
            "interval_80": float(np.mean(covered_80)),
        },
    }


def _empirical_recovery_probability(
    predicted: np.ndarray, residuals: np.ndarray, horizon_hours: float
) -> np.ndarray:
    """Estimate P(actual remaining time <= horizon) from rolling residuals."""
    sorted_residuals = np.sort(residuals)
    thresholds = horizon_hours - predicted
    return np.searchsorted(sorted_residuals, thresholds, side="right") / len(sorted_residuals)
