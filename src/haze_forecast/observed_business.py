"""Causal business eligibility and conservative grouping on an observed hour grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from haze_forecast.features import MEASUREMENT_COLUMNS, add_missingness_indicators
from haze_forecast.observed_hourly_features import build_observed_hourly_features
from haze_forecast.observed_recovery_labels import observed_recovery_bounds


def build_business_states(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify each hour without retrospectively assigning event membership."""
    required = {"timestamp", "aqi", "pm2_5_iaqi", "pm10_iaqi"}
    if not required.issubset(frame):
        raise ValueError("missing business-state inputs")
    timestamps = frame.timestamp
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype) or timestamps.isna().any():
        raise ValueError("timezone-aware timestamps required")
    if not timestamps.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("complete ordered hourly grid required")
    values = frame[["aqi", "pm2_5_iaqi", "pm10_iaqi"]].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("infinite measurements are invalid")
    aqi, pm25, pm10 = values.T
    polluted = (aqi > 100) & ((pm25 > 100) | (pm10 > 100))
    not_polluted = (aqi <= 100) | ((aqi > 100) & (pm25 <= 100) & (pm10 <= 100))
    states = np.where(
        polluted,
        "confirmed_pollution",
        np.where(not_polluted, "confirmed_not_pollution", "unknown"),
    )
    rows = []
    active = next_episode = pollution_run = clean_run = 0
    block = 1
    pending_block = False
    for aq, is_polluted in zip(aqi, polluted, strict=True):
        if pending_block:
            block += 1
            pending_block = False
        pollution_run = pollution_run + 1 if is_polluted else 0
        clean_run = clean_run + 1 if np.isfinite(aq) and aq <= 100 else 0
        if not np.isfinite(aq):
            active = 0
        elif clean_run == 6:
            active = 0
            pending_block = True
        elif not active and pollution_run == 3:
            next_episode += 1
            active = next_episode
        rows.append((active, block, bool(active and is_polluted)))
    result = frame.copy()
    result["pollution_state"] = states
    result["episode_id"] = pd.Series([r[0] for r in rows], index=frame.index, dtype="int64")
    result["continuity_block"] = pd.Series([r[1] for r in rows], index=frame.index, dtype="int64")
    result["eligible"] = pd.Series([r[2] for r in rows], index=frame.index, dtype=bool)
    return result


def build_observed_business(
    sources: dict[str, pd.DataFrame], feature_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge measurement sources, then retain every causally eligible origin."""
    if not sources or any(frame.empty for frame in sources.values()):
        raise ValueError("nonempty sources required")
    parts = []
    for source, frame in sources.items():
        columns = ["timestamp", *[c for c in MEASUREMENT_COLUMNS if c in frame]]
        part = frame[columns].copy()
        part["source"] = str(source)
        parts.append(part)
    measurements = pd.concat(parts, ignore_index=True)
    full = build_observed_hourly_features(measurements)
    full["source"] = full.timestamp.map(measurements.set_index("timestamp").source)
    full = build_business_states(full)
    pbl = [
        name
        for name in feature_names
        if name.startswith("boundary_layer_height_m") and not name.endswith("_is_missing")
    ]
    full = add_missingness_indicators(full, pbl)
    if "boundary_layer_height_m_change_3_is_missing" in feature_names:
        full["boundary_layer_height_m_change_3_is_missing"] = (
            full.boundary_layer_height_m.isna() | full.boundary_layer_height_m_lag_3.isna()
        ).astype(float)
    missing = set(feature_names) - set(full.columns)
    if missing:
        raise ValueError(f"missing requested features: {sorted(missing)}")
    origins = full.loc[full.eligible].copy().reset_index(drop=True)
    if len(origins):
        bounds = observed_recovery_bounds(full[["timestamp", "aqi"]], origins.timestamp)
        origins = origins.merge(bounds, on="timestamp", validate="one_to_one", sort=False)
    else:
        for name in ("lower_bound_hours", "upper_bound_hours", "exact_hours"):
            origins[name] = pd.Series(dtype=float)
        origins["observed_prefix_hours"] = pd.Series(dtype="int64")
        origins["informative"] = pd.Series(dtype=bool)
        origins["label_available_at"] = pd.Series(dtype=full.timestamp.dtype)
        origins["reason"] = pd.Series(dtype=str)
    return full, origins
