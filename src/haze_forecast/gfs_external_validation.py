"""Shared guards and metrics for externally issued GFS forecast experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

GFS_OBSERVATION_FEATURES = (
    "aqi",
    "boundary_layer_height_m",
    "co",
    "no2",
    "o3",
    "pm10",
    "pm2_5",
    "pm10_iaqi",
    "pm2_5_iaqi",
    "relative_humidity_pct",
    "so2",
    "temperature_c",
    "wind_u_ms",
    "wind_v_ms",
)
GFS_F006_FEATURES = (
    "gfs_temperature_c_f006",
    "gfs_relative_humidity_pct_f006",
    "gfs_wind_u_ms_f006",
    "gfs_wind_v_ms_f006",
    "gfs_boundary_layer_height_m_f006",
)


def validate_gfs_forecast_timing(frame: pd.DataFrame) -> None:
    """Require every feature row to use a previously issued, future-valid GFS field."""
    required = {"timestamp", "gfs_forecast_run_utc", "gfs_forecast_valid_utc"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"GFS frame is missing timing columns: {sorted(missing)}")
    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    run = pd.to_datetime(frame["gfs_forecast_run_utc"], utc=True, errors="raise")
    valid = pd.to_datetime(frame["gfs_forecast_valid_utc"], utc=True, errors="raise")
    if not run.le(timestamp).all():
        raise ValueError("GFS forecast run was issued after a prediction timestamp")
    if not timestamp.lt(valid).all():
        raise ValueError("GFS forecast valid time is not after a prediction timestamp")


def event_equal_mae(
    actual: np.ndarray, prediction: np.ndarray, event_ids: pd.Series | np.ndarray
) -> float:
    """Return MAE after giving every complete pollution event equal weight."""
    frame = pd.DataFrame(
        {"event_id": event_ids, "absolute_error": np.abs(np.asarray(prediction) - np.asarray(actual))}
    )
    return float(frame.groupby("event_id", sort=False)["absolute_error"].mean().mean())
