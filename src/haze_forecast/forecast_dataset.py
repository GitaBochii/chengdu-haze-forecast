"""Build auditable no-leak GFS feature tables for labelled haze-event rows."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from haze_forecast.forecast_extraction import (
    GFSPointForecast,
    extract_gfs_surface_forecast_at_point,
)
from haze_forecast.forecast_sources import build_operational_gfs_request


def enrich_event_rows_with_gfs_forecast(
    event_rows: pd.DataFrame,
    longitude: float,
    latitude: float,
    wgrib2_executable: Path,
    lead_hours: int = 6,
    max_workers: int = 1,
    message_workers: int = 1,
    message_timeout_s: int = 45,
    *,
    extractor: Callable[..., GFSPointForecast] = extract_gfs_surface_forecast_at_point,
) -> pd.DataFrame:
    """Attach one forecast trajectory point selected from each row's latest GFS cycle."""
    if "timestamp" not in event_rows:
        raise ValueError("event_rows must contain timestamp")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if message_workers <= 0:
        raise ValueError("message_workers must be positive")
    if message_timeout_s <= 0:
        raise ValueError("message_timeout_s must be positive")
    timestamps = pd.to_datetime(event_rows["timestamp"])
    if timestamps.dt.tz is None:
        raise ValueError("event_rows timestamps must include timezone information")
    result = event_rows.copy()
    issue_times_utc = timestamps.dt.tz_convert("UTC")
    run_times = issue_times_utc.map(
        lambda timestamp: build_operational_gfs_request(
            timestamp.to_pydatetime(), latitude, longitude, max_lead_hours=lead_hours
        ).run_time
    )
    unique_run_times = sorted(set(run_times))

    def extract_one(run_time: pd.Timestamp) -> GFSPointForecast:
        return extractor(
            run_time,
            lead_hours,
            longitude,
            latitude,
            wgrib2_executable,
            message_workers=message_workers,
            message_timeout_s=message_timeout_s,
        )

    forecasts: dict[pd.Timestamp, GFSPointForecast] = {}
    if max_workers == 1:
        for run_time in unique_run_times:
            forecasts[pd.Timestamp(run_time)] = extract_one(run_time)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_runs = {executor.submit(extract_one, run_time): run_time for run_time in unique_run_times}
            for future in as_completed(future_runs):
                run_time = future_runs[future]
                forecasts[pd.Timestamp(run_time)] = future.result()

    result["gfs_forecast_run_utc"] = run_times
    result["gfs_forecast_valid_utc"] = run_times.map(
        lambda run_time: pd.Timestamp(forecasts[pd.Timestamp(run_time)].valid_at)
    )
    result["gfs_grid_longitude"] = run_times.map(
        lambda run_time: forecasts[pd.Timestamp(run_time)].grid_longitude
    )
    result["gfs_grid_latitude"] = run_times.map(
        lambda run_time: forecasts[pd.Timestamp(run_time)].grid_latitude
    )
    suffix = f"f{lead_hours:03d}"
    feature_values = {
        "temperature_c": f"gfs_temperature_c_{suffix}",
        "relative_humidity_pct": f"gfs_relative_humidity_pct_{suffix}",
        "wind_u_ms": f"gfs_wind_u_ms_{suffix}",
        "wind_v_ms": f"gfs_wind_v_ms_{suffix}",
        "boundary_layer_height_m": f"gfs_boundary_layer_height_m_{suffix}",
    }
    for source_name, feature_name in feature_values.items():
        result[feature_name] = run_times.map(
            lambda run_time, source_name=source_name: getattr(
                forecasts[pd.Timestamp(run_time)], source_name
            )
        )
    if not result["gfs_forecast_run_utc"].le(issue_times_utc).all():
        raise RuntimeError("a GFS feature row uses a forecast cycle issued after its event timestamp")
    return result
