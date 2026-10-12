"""只依赖当前和过去观测的特征工程。"""

from __future__ import annotations

import numpy as np
import pandas as pd

MEASUREMENT_COLUMNS = (
    "aqi",
    "pm2_5",
    "pm10",
    "so2",
    "no2",
    "co",
    "o3",
    "temperature_c",
    "relative_humidity_pct",
    "wind_u_ms",
    "wind_v_ms",
    "boundary_layer_height_m",
)
TARGET_COLUMNS = {"event_id", "remaining_hours"}
NON_FEATURE_COLUMNS = {"timestamp", "station_id", *TARGET_COLUMNS}
SNAPSHOT_DERIVED_COLUMNS = {
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "wind_speed_ms",
}
TRAJECTORY_FEATURE_NAMES = (
    "pollution_age_hours",
    "event_peak_aqi",
    "event_peak_pm2_5",
    "event_peak_pm10",
    "aqi_from_event_peak",
    "pm2_5_from_event_peak",
    "pm10_from_event_peak",
    "aqi_cumulative_excess",
    "pm2_5_cumulative_excess",
    "pm10_cumulative_excess",
    "event_peak_age_hours",
)
_TRAJECTORY_MEASUREMENTS = (
    ("aqi", 100.0),
    ("pm2_5", 75.0),
    ("pm10", 150.0),
)


def _ongoing_pollution_hours(frame: pd.DataFrame) -> pd.Series:
    polluted = frame["aqi"].gt(100) & frame["particle_primary"].fillna(False)
    values: list[float] = []
    duration = 0.0
    previous_timestamp: pd.Timestamp | None = None
    for timestamp, is_polluted in zip(frame["timestamp"], polluted, strict=True):
        if is_polluted and previous_timestamp is not None and timestamp - previous_timestamp == pd.Timedelta(hours=1):
            duration += 1.0
        elif is_polluted:
            duration = 1.0
        else:
            duration = 0.0
        values.append(duration)
        previous_timestamp = timestamp
    return pd.Series(values, index=frame.index, dtype="float64")


def add_pollution_trajectory_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize each current particle-pollution run without reading future hours."""
    required = {"timestamp", "aqi", "pm2_5", "pm10", "particle_primary"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"trajectory features missing columns: {sorted(missing)}")
    result = frame.sort_values("timestamp").reset_index(drop=True).copy()
    active = result["aqi"].gt(100) & result["particle_primary"].fillna(False)
    measurements = {
        name: pd.to_numeric(result[name], errors="coerce") for name, _ in _TRAJECTORY_MEASUREMENTS
    }
    rows: list[dict[str, float]] = []
    previous_timestamp: pd.Timestamp | None = None
    age = 0.0
    peaks = {name: float("nan") for name, _ in _TRAJECTORY_MEASUREMENTS}
    cumulative = {name: 0.0 for name, _ in _TRAJECTORY_MEASUREMENTS}
    last_aqi_peak: pd.Timestamp | None = None
    for index, timestamp in enumerate(result["timestamp"]):
        continuous = previous_timestamp is not None and timestamp - previous_timestamp == pd.Timedelta(hours=1)
        if not active.iloc[index] or not continuous:
            age = 0.0
            peaks = {name: float("nan") for name, _ in _TRAJECTORY_MEASUREMENTS}
            cumulative = {name: 0.0 for name, _ in _TRAJECTORY_MEASUREMENTS}
            last_aqi_peak = None
        if active.iloc[index]:
            age += 1.0
            for name, threshold in _TRAJECTORY_MEASUREMENTS:
                value = float(measurements[name].iloc[index])
                if np.isfinite(value):
                    peaks[name] = value if not np.isfinite(peaks[name]) else max(peaks[name], value)
                    cumulative[name] += max(0.0, value - threshold)
            current_aqi = float(measurements["aqi"].iloc[index])
            if np.isfinite(current_aqi) and current_aqi >= peaks["aqi"]:
                last_aqi_peak = timestamp
            rows.append(
                {
                    "pollution_age_hours": age,
                    "event_peak_aqi": peaks["aqi"],
                    "event_peak_pm2_5": peaks["pm2_5"],
                    "event_peak_pm10": peaks["pm10"],
                    "aqi_from_event_peak": current_aqi - peaks["aqi"],
                    "pm2_5_from_event_peak": measurements["pm2_5"].iloc[index] - peaks["pm2_5"],
                    "pm10_from_event_peak": measurements["pm10"].iloc[index] - peaks["pm10"],
                    "aqi_cumulative_excess": cumulative["aqi"],
                    "pm2_5_cumulative_excess": cumulative["pm2_5"],
                    "pm10_cumulative_excess": cumulative["pm10"],
                    "event_peak_age_hours": float((timestamp - last_aqi_peak).total_seconds() / 3600)
                    if last_aqi_peak is not None
                    else float("nan"),
                }
            )
        else:
            rows.append(
                {
                    "pollution_age_hours": 0.0,
                    "event_peak_aqi": float("nan"),
                    "event_peak_pm2_5": float("nan"),
                    "event_peak_pm10": float("nan"),
                    "aqi_from_event_peak": float("nan"),
                    "pm2_5_from_event_peak": float("nan"),
                    "pm10_from_event_peak": float("nan"),
                    "aqi_cumulative_excess": 0.0,
                    "pm2_5_cumulative_excess": 0.0,
                    "pm10_cumulative_excess": 0.0,
                    "event_peak_age_hours": 0.0,
                }
            )
        previous_timestamp = timestamp
    trajectory = pd.DataFrame(rows, index=result.index)
    return pd.concat([result, trajectory], axis=1)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create cyclic, wind, lag and rolling features without reading future rows."""
    if "timestamp" not in frame.columns:
        raise ValueError("timestamp is required for feature generation")

    result = frame.sort_values("timestamp").reset_index(drop=True).copy()
    hour = result["timestamp"].dt.hour
    month = result["timestamp"].dt.month
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["month_sin"] = np.sin(2 * np.pi * month / 12)
    result["month_cos"] = np.cos(2 * np.pi * month / 12)

    if {"wind_u_ms", "wind_v_ms"}.issubset(result.columns):
        result["wind_speed_ms"] = np.hypot(result["wind_u_ms"], result["wind_v_ms"])
    if {"aqi", "particle_primary"}.issubset(result.columns):
        result["ongoing_pollution_hours"] = _ongoing_pollution_hours(result)
    if {"aqi", "pm2_5", "pm10", "particle_primary"}.issubset(result.columns):
        result = add_pollution_trajectory_features(result)

    for name in (column for column in MEASUREMENT_COLUMNS if column in result.columns):
        historical = pd.to_numeric(result[name], errors="coerce").shift(1)
        result[f"{name}_lag_1"] = historical
        result[f"{name}_lag_3"] = result[name].shift(3)
        result[f"{name}_lag_24"] = result[name].shift(24)
        result[f"{name}_roll_3"] = historical.rolling(window=3, min_periods=1).mean()
        result[f"{name}_roll_24"] = historical.rolling(window=24, min_periods=6).mean()
        result[f"{name}_change_3"] = result[name] - result[name].shift(3)
    return result


def add_missingness_indicators(frame: pd.DataFrame, source_columns: list[str]) -> pd.DataFrame:
    """Expose absent measurements as numeric model inputs without filling them."""
    result = frame.copy()
    missing = set(source_columns) - set(result.columns)
    if missing:
        raise ValueError(f"missing source columns for indicators: {sorted(missing)}")
    for name in source_columns:
        result[f"{name}_is_missing"] = result[name].isna().astype(float)
    return result


def model_feature_names(frame: pd.DataFrame, include_experimental: bool = False) -> list[str]:
    """Return numeric model inputs while excluding timestamps, identifiers, targets and rejected trials."""
    experimental = {
        "ongoing_pollution_hours",
        *TRAJECTORY_FEATURE_NAMES,
        *(name for name in frame.columns if name.endswith("_change_3")),
    }
    return [
        name
        for name in frame.columns
        if name not in NON_FEATURE_COLUMNS
        and (include_experimental or name not in experimental)
        and pd.api.types.is_numeric_dtype(frame[name])
    ]


def snapshot_feature_names(frame: pd.DataFrame) -> list[str]:
    """Return only inputs that the manual form can observe at one timestamp."""
    allowed = {*MEASUREMENT_COLUMNS, *SNAPSHOT_DERIVED_COLUMNS}
    return [name for name in model_feature_names(frame) if name in allowed]


def training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only labeled hours where the user-facing system would make a prediction."""
    required = {"event_id", "remaining_hours", "aqi", "particle_primary"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns for training rows: {sorted(missing)}")

    enriched = build_features(frame)
    mask = (
        enriched["event_id"].gt(0)
        & enriched["remaining_hours"].notna()
        & enriched["aqi"].gt(100)
        & enriched["particle_primary"].fillna(False)
    )
    return enriched.loc[mask].reset_index(drop=True)
