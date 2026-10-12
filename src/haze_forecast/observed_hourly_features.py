"""Causal feature generation on a complete, unfilled hourly measurement grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from haze_forecast.features import MEASUREMENT_COLUMNS, build_features
from haze_forecast.labels import add_particle_primary


def build_observed_hourly_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild features from measurements only; missing hours remain unobserved.

    Input is one time series with unique, timezone-aware hourly timestamps.
    Existing labels and derived columns are deliberately discarded. This does
    not infer recovery labels or treat an unobserved hour as a clean hour.
    """
    if not frame.columns.is_unique:
        raise ValueError("duplicate input columns")
    required = {"timestamp", "aqi", "pm2_5", "pm10"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing measurement columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("at least one measurement row is required")
    timestamps = frame["timestamp"]
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("timestamp must be timezone-aware datetime")  # noqa: TRY004
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("timestamps must be nonmissing and unique")
    if not (
        timestamps.dt.minute.eq(0)
        & timestamps.dt.second.eq(0)
        & timestamps.dt.microsecond.eq(0)
        & timestamps.dt.nanosecond.eq(0)
    ).all():
        raise ValueError("timestamps must fall on an exact hour")
    columns = [name for name in MEASUREMENT_COLUMNS if name in frame.columns]
    measurements = frame[["timestamp", *columns]].copy()
    for name in columns:
        try:
            measurements[name] = pd.to_numeric(measurements[name], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"measurement {name} must be numeric or missing") from exc
        if np.isinf(measurements[name].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"measurement {name} must not be infinite")
    measurements = measurements.sort_values("timestamp").set_index("timestamp")
    grid = pd.date_range(measurements.index[0], measurements.index[-1], freq="h")
    observed = grid.isin(measurements.index)
    measurements = measurements.reindex(grid.rename("timestamp")).reset_index()
    result = build_features(add_particle_primary(measurements))
    result["input_hour_observed"] = observed
    return result
