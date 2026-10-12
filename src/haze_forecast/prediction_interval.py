"""Coverage-calibrated prediction intervals for the dissipation model.

The deployed AFT model returns a conditional *median* remaining time, and the
target distribution is strongly right-skewed. A symmetric ``prediction +/- MAE``
band is therefore wrong twice over: it is centred on the wrong statistic and it
is not calibrated to any coverage level. This module calibrates an asymmetric,
multiplicative interval from out-of-sample ``actual / predicted`` ratios so that
the published band actually covers the stated fraction of observations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_COVERAGE = 0.90
MIN_PREDICTION_HOURS = 0.5
CALIBRATION_LEVELS: tuple[float, ...] = (0.80, 0.90)


def residual_ratios(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    *,
    min_prediction: float = MIN_PREDICTION_HOURS,
) -> np.ndarray:
    """Return ``actual / predicted`` with the denominator floored.

    The floor keeps the ratio finite for near-zero predictions, which the model
    emits for rows that are about to dissipate.
    """
    truth = np.asarray(actual, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    if truth.shape != estimate.shape:
        raise ValueError("actual and predicted must have the same shape")
    return truth / np.maximum(estimate, min_prediction)


def calibrate_ratio_intervals(
    ratios: pd.Series | np.ndarray,
    levels: tuple[float, ...] = CALIBRATION_LEVELS,
) -> dict[str, dict[str, float]]:
    """Return multiplicative bounds that cover each requested level.

    The bounds are the ``(1 - level) / 2`` and ``(1 + level) / 2`` quantiles of
    the observed ratios, so the interval is two-sided and asymmetric.
    """
    values = np.asarray(ratios, dtype=float)
    if values.size == 0:
        raise ValueError("at least one calibration ratio is required")
    if np.any(~np.isfinite(values)):
        raise ValueError("calibration ratios must be finite")
    calibration: dict[str, dict[str, float]] = {}
    for level in levels:
        if not 0.0 < level < 1.0:
            raise ValueError(f"coverage level must lie strictly between 0 and 1, got {level}")
        tail = (1.0 - level) / 2.0
        calibration[_level_key(level)] = {
            "coverage": float(level),
            "lower_ratio": float(np.quantile(values, tail)),
            "upper_ratio": float(np.quantile(values, 1.0 - tail)),
            "samples": int(values.size),
        }
    return calibration


def prediction_interval(
    prediction: float,
    calibration: dict[str, dict[str, float]],
    coverage: float = DEFAULT_COVERAGE,
) -> tuple[float, float]:
    """Return ``(lower, upper)`` bounds in hours for a single prediction."""
    key = _level_key(coverage)
    if key not in calibration:
        available = ", ".join(sorted(calibration))
        raise KeyError(
            f"no calibration for coverage {coverage:.2f}; available: {available}"
        )
    band = calibration[key]
    point = max(float(prediction), 0.0)
    return (
        max(point * float(band["lower_ratio"]), 0.0),
        max(point * float(band["upper_ratio"]), 0.0),
    )


def coverage_summary(
    actual: pd.Series | np.ndarray,
    lower: pd.Series | np.ndarray,
    upper: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Report empirical coverage and mean width of a set of intervals."""
    truth = np.asarray(actual, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if not (truth.shape == low.shape == high.shape):
        raise ValueError("actual, lower and upper must have the same shape")
    covered = (truth >= low) & (truth <= high)
    return {
        "rows": int(truth.size),
        "coverage": float(covered.mean()),
        "mean_width": float(np.mean(high - low)),
    }


def write_calibration(path: Path, payload: dict[str, object]) -> None:
    """Persist a calibration document with sorted keys for stable diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _level_key(level: float) -> str:
    return f"{level:.2f}"
