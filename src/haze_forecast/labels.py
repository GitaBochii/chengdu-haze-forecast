"""污染过程识别和剩余消散时间标签。"""

from __future__ import annotations

import numpy as np
import pandas as pd

PM25_BREAKPOINTS = (0.0, 35.0, 75.0, 115.0, 150.0, 250.0, 350.0, 500.0)
PM10_BREAKPOINTS = (0.0, 50.0, 150.0, 250.0, 350.0, 420.0, 500.0, 600.0)
AQI_BREAKPOINTS = (0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0)
AQI_RECOVERY_THRESHOLD = 100.0
EVENT_START_STABLE_HOURS = 3
RECOVERY_STABLE_HOURS = 6


def _iaqi(concentration: pd.Series, concentration_breakpoints: tuple[float, ...]) -> pd.Series:
    result = pd.Series(np.nan, index=concentration.index, dtype="float64")
    values = pd.to_numeric(concentration, errors="coerce")
    for low, high, aqi_low, aqi_high in zip(
        concentration_breakpoints[:-1],
        concentration_breakpoints[1:],
        AQI_BREAKPOINTS[:-1],
        AQI_BREAKPOINTS[1:],
        strict=True,
    ):
        mask = values.ge(low) & values.le(high)
        result.loc[mask] = (aqi_high - aqi_low) / (high - low) * (
            values.loc[mask] - low
        ) + aqi_low
    result.loc[values.gt(concentration_breakpoints[-1])] = AQI_BREAKPOINTS[-1]
    return result.round()


def add_particle_primary(frame: pd.DataFrame) -> pd.DataFrame:
    """Identify hours where particulate pollution independently exceeds the AQI threshold."""
    required = {"timestamp", "aqi", "pm2_5", "pm10"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns for particle-primary inference: {sorted(missing)}")

    result = frame.sort_values("timestamp").copy()
    pm25_24h = result["pm2_5"].rolling(window=24, min_periods=24).mean()
    pm10_24h = result["pm10"].rolling(window=24, min_periods=24).mean()
    result["pm2_5_iaqi"] = _iaqi(pm25_24h, PM25_BREAKPOINTS)
    result["pm10_iaqi"] = _iaqi(pm10_24h, PM10_BREAKPOINTS)
    particle_iaqi = result[["pm2_5_iaqi", "pm10_iaqi"]].max(axis=1)
    result["particle_primary"] = result["aqi"].gt(AQI_RECOVERY_THRESHOLD) & particle_iaqi.gt(
        AQI_RECOVERY_THRESHOLD
    )
    return result


def _is_consecutive_hourly(timestamps: pd.Series, start: int, length: int) -> bool:
    window = timestamps.iloc[start : start + length]
    return len(window) == length and window.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()


def _is_stable_sequence(
    condition: pd.Series, timestamps: pd.Series, start: int, length: int
) -> bool:
    return _is_consecutive_hourly(timestamps, start, length) and bool(
        condition.iloc[start : start + length].all()
    )


def label_pollution_events(
    frame: pd.DataFrame, require_aqi_recovery: bool = True
) -> pd.DataFrame:
    """Label particle-pollution events, optionally requiring overall AQI recovery to end them."""
    required = {"timestamp", "aqi", "particle_primary"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns for event labels: {sorted(missing)}")

    result = frame.sort_values("timestamp").reset_index(drop=True).copy()
    polluted = result["aqi"].gt(AQI_RECOVERY_THRESHOLD) & result["particle_primary"].fillna(False)
    non_particle_primary = ~result["particle_primary"].fillna(False)
    if require_aqi_recovery:
        clean = result["aqi"].le(AQI_RECOVERY_THRESHOLD) & non_particle_primary
    else:
        iaqi_columns = [name for name in ("pm2_5_iaqi", "pm10_iaqi") if name in result]
        particle_observed = result[iaqi_columns].notna().any(axis=1) if iaqi_columns else True
        clean = non_particle_primary & particle_observed
    event_id = np.zeros(len(result), dtype="int64")
    remaining_hours = np.full(len(result), np.nan)

    next_event_id = 0
    start = 0
    while start < len(result):
        if not _is_stable_sequence(polluted, result["timestamp"], start, EVENT_START_STABLE_HOURS):
            start += 1
            continue
        clean_start = start + EVENT_START_STABLE_HOURS
        while clean_start < len(result):
            if _is_stable_sequence(clean, result["timestamp"], clean_start, RECOVERY_STABLE_HOURS):
                next_event_id += 1
                event_id[start:clean_start] = next_event_id
                remaining_hours[start:clean_start] = np.arange(clean_start - start, 0, -1)
                start = clean_start + RECOVERY_STABLE_HOURS
                break
            clean_start += 1
        else:
            break

    result["event_id"] = event_id
    result["remaining_hours"] = remaining_hours
    return result


def label_city_data(frame: pd.DataFrame, require_aqi_recovery: bool = True) -> pd.DataFrame:
    """Infer particle-primary status before applying the configured event state machine."""
    return label_pollution_events(
        add_particle_primary(frame), require_aqi_recovery=require_aqi_recovery
    )


def label_summary(frame: pd.DataFrame) -> dict[str, int]:
    """Summarize only completed pollution events available for supervised learning."""
    completed = frame[frame["event_id"].gt(0)]
    return {
        "completed_events": int(completed["event_id"].nunique()),
        "labeled_hours": int(completed["remaining_hours"].notna().sum()),
        "particle_primary_hours": int(frame["particle_primary"].sum()),
    }
