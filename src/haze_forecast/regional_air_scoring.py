"""Causal station support and observed-label checks for city forecast scoring."""

from __future__ import annotations

import numpy as np


def active_node_mask(
    raw_targets: np.ndarray, origins: np.ndarray, lookback: int = 24,
) -> np.ndarray:
    """Select each pollutant's nodes observed in the past/current lookback window."""
    raw = np.asarray(raw_targets)
    origins = np.asarray(origins)
    if raw.ndim != 3 or origins.ndim != 1 or lookback < 1:
        raise ValueError("expected [time,node,channel], one-dimensional origins and positive lookback")
    if not np.issubdtype(origins.dtype, np.integer):
        raise ValueError("origins must be integer indices")
    if ((origins < 0) | (origins >= len(raw))).any():
        raise ValueError("origin outside observed time range")
    active = np.zeros((len(origins), *raw.shape[1:]), dtype=bool)
    for row, origin in enumerate(origins):
        active[row] = np.isfinite(raw[max(0, origin - lookback + 1):origin + 1]).any(axis=0)
    return active


def aggregate_active_predictions(pred: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Apply one origin-known support mask to every lead, without future target masks."""
    prediction = np.asarray(pred, dtype=float)
    active = np.asarray(active, dtype=bool)
    if prediction.ndim != 4 or active.shape != (
        prediction.shape[0], prediction.shape[2], prediction.shape[3],
    ):
        raise ValueError("expected predictions [batch,lead,node,channel] and active [batch,node,channel]")
    excluded = np.broadcast_to(~active[:, None], prediction.shape)
    return np.ma.median(np.ma.array(prediction, mask=excluded), axis=2).filled(np.nan)


def verified_recovery_mask(
    actual_hours: np.ndarray, future_aqi: np.ndarray, stable_hours: int = 6,
) -> np.ndarray:
    """Require observed AQI through recovery confirmation or the entire capped horizon."""
    actual = np.asarray(actual_hours, dtype=float)
    future = np.asarray(future_aqi, dtype=float)
    if actual.ndim != 1 or future.ndim != 2 or len(actual) != len(future):
        raise ValueError("expected actual hours [batch] and future AQI [batch,lead]")
    if stable_hours < 1 or future.shape[1] < stable_hours:
        raise ValueError("forecast horizon must support the positive stable-hour window")
    verified = np.zeros(len(actual), dtype=bool)
    last_confirmable = future.shape[1] - stable_hours + 1
    for row, hours in enumerate(actual):
        if not np.isfinite(hours) or hours < 1 or hours != int(hours):
            continue
        required = int(hours) + stable_hours - 1 if hours <= last_confirmable else future.shape[1]
        verified[row] = np.isfinite(future[row, :required]).all()
    return verified
