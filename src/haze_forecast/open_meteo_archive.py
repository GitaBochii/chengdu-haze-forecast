"""Parse audited Open-Meteo archive boundary-layer-height responses."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

WEATHER_COLUMNS = (
    "temperature_c",
    "relative_humidity_pct",
    "wind_u_ms",
    "wind_v_ms",
    "boundary_layer_height_m",
)


def parse_boundary_layer_height(payload: Mapping[str, object]) -> pd.DataFrame:
    """Convert an Asia/Shanghai Open-Meteo hourly response into a joinable table."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping):
        raise TypeError("Open-Meteo payload must contain an hourly mapping")
    timestamps = hourly.get("time")
    values = hourly.get("boundary_layer_height")
    if not isinstance(timestamps, list) or not isinstance(values, list):
        raise TypeError("Open-Meteo hourly payload must contain time and boundary_layer_height lists")
    if not timestamps or len(timestamps) != len(values):
        raise ValueError("Open-Meteo time and boundary_layer_height lists must be non-empty and aligned")
    parsed_timestamps = pd.to_datetime(timestamps, errors="raise")
    if parsed_timestamps.tz is None:
        parsed_timestamps = parsed_timestamps.tz_localize("Asia/Shanghai")
    else:
        parsed_timestamps = parsed_timestamps.tz_convert("Asia/Shanghai")
    return pd.DataFrame(
        {
            "timestamp": parsed_timestamps,
            "boundary_layer_height_m": pd.to_numeric(values, errors="coerce"),
        }
    )


def parse_hourly_weather(payload: Mapping[str, object]) -> pd.DataFrame:
    """Parse Open-Meteo archive weather into the model's local feature schema."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping):
        raise TypeError("Open-Meteo payload must contain an hourly mapping")
    required = {
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "boundary_layer_height",
    }
    missing = required - set(hourly)
    if missing:
        raise ValueError(f"Open-Meteo hourly payload missing columns: {sorted(missing)}")
    lengths = {len(hourly[name]) for name in required if isinstance(hourly[name], list)}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Open-Meteo hourly values must be non-empty and aligned lists")
    if any(not isinstance(hourly[name], list) for name in required):
        raise TypeError("Open-Meteo hourly values must be lists")
    timestamps = pd.to_datetime(hourly["time"], errors="raise")
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("Asia/Shanghai")
    else:
        timestamps = timestamps.tz_convert("Asia/Shanghai")
    speed = pd.to_numeric(pd.Series(hourly["wind_speed_10m"]), errors="coerce")
    direction = np.radians(pd.to_numeric(pd.Series(hourly["wind_direction_10m"]), errors="coerce"))
    wind_u = -speed * np.sin(direction)
    wind_v = -speed * np.cos(direction)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_c": pd.to_numeric(pd.Series(hourly["temperature_2m"]), errors="coerce"),
            "relative_humidity_pct": pd.to_numeric(
                pd.Series(hourly["relative_humidity_2m"]), errors="coerce"
            ),
            "wind_u_ms": wind_u.mask(wind_u.abs().lt(1e-12), 0.0),
            "wind_v_ms": wind_v.mask(wind_v.abs().lt(1e-12), 0.0),
            "boundary_layer_height_m": pd.to_numeric(
                pd.Series(hourly["boundary_layer_height"]), errors="coerce"
            ),
        }
    )


def fill_weather_gaps(observed: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """Fill only unavailable local weather fields with archived reanalysis values."""
    required = {"timestamp", *WEATHER_COLUMNS}
    observed_missing = required - set(observed.columns)
    archive_missing = required - set(archive.columns)
    if observed_missing:
        raise ValueError(f"observed weather missing columns: {sorted(observed_missing)}")
    if archive_missing:
        raise ValueError(f"archive weather missing columns: {sorted(archive_missing)}")
    if archive["timestamp"].duplicated().any():
        raise ValueError("archive weather must have one row per timestamp")
    archive_values = archive[["timestamp", *WEATHER_COLUMNS]].rename(
        columns={column: f"{column}_archive" for column in WEATHER_COLUMNS}
    )
    result = observed.merge(archive_values, on="timestamp", how="left", validate="m:1")
    for column in WEATHER_COLUMNS:
        observed_values = pd.to_numeric(result[column], errors="coerce")
        archive_values = pd.to_numeric(result.pop(f"{column}_archive"), errors="coerce")
        result[column] = observed_values.where(observed_values.notna(), archive_values)
    return result


def add_boundary_layer_height(
    pollution_hourly: pd.DataFrame, boundary_layer_height: pd.DataFrame
) -> pd.DataFrame:
    """Left-join reanalysis PBL height while preserving every pollution hour."""
    required = {"timestamp", "boundary_layer_height_m"}
    missing = required - set(boundary_layer_height.columns)
    if missing:
        raise ValueError(f"boundary-layer-height table missing columns: {sorted(missing)}")
    if "timestamp" not in pollution_hourly:
        raise ValueError("pollution table missing timestamp")
    result = pollution_hourly.drop(columns="boundary_layer_height_m", errors="ignore").merge(
        boundary_layer_height[["timestamp", "boundary_layer_height_m"]],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    return result.sort_values("timestamp").reset_index(drop=True)
