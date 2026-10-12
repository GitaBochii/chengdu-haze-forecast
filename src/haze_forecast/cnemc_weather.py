"""Join 2014 CNEMC pollution with audited NOAA station weather observations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _relative_humidity_percent(temperature_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    saturation = 17.625 * temperature_c / (243.04 + temperature_c)
    actual = 17.625 * dewpoint_c / (243.04 + dewpoint_c)
    return (100 * np.exp(actual - saturation)).clip(lower=0, upper=100)


def add_noaa_weather(pollution_hourly: pd.DataFrame, noaa_hourly: pd.DataFrame) -> pd.DataFrame:
    """Left-join NOAA weather and derive local-model-compatible weather columns."""
    if "timestamp" not in pollution_hourly or "timestamp" not in noaa_hourly:
        raise ValueError("pollution and NOAA hourly frames must contain timestamp")
    weather_columns = [
        "timestamp",
        "noaa_temperature_c",
        "noaa_dewpoint_c",
        "noaa_wind_direction_deg",
        "noaa_wind_speed_ms",
    ]
    available_weather_columns = [
        column for column in weather_columns if column in noaa_hourly.columns
    ]
    result = pollution_hourly.merge(
        noaa_hourly[available_weather_columns].drop_duplicates("timestamp"),
        on="timestamp",
        how="left",
        validate="m:1",
    )
    temperature = pd.to_numeric(result.get("noaa_temperature_c"), errors="coerce")
    dewpoint = pd.to_numeric(result.get("noaa_dewpoint_c"), errors="coerce")
    direction = np.radians(pd.to_numeric(result.get("noaa_wind_direction_deg"), errors="coerce"))
    speed = pd.to_numeric(result.get("noaa_wind_speed_ms"), errors="coerce")
    result["temperature_c"] = temperature
    result["relative_humidity_pct"] = _relative_humidity_percent(temperature, dewpoint)
    result["wind_u_ms"] = -speed * np.sin(direction)
    result["wind_v_ms"] = -speed * np.cos(direction)
    if "boundary_layer_height_m" not in result:
        result["boundary_layer_height_m"] = np.nan
    return result
