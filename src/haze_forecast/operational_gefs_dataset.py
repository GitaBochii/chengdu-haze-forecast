"""Feature extraction from time-safe operational GEFS ensemble products."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from haze_forecast.grib import GRIBPointValue, query_grib_point
from haze_forecast.operational_gefs import (
    GEFS_LEADS,
    GEFSFieldMessage,
    download_operational_gefs_surface_messages,
    select_operational_gefs_run,
)

OPERATIONAL_GEFS_BASE_FEATURES = (
    "temperature_c",
    "relative_humidity_pct",
    "wind_u_ms",
    "wind_v_ms",
)
OPERATIONAL_GEFS_FEATURES = tuple(
    f"gefs_{stat}_{name}_f{lead:03d}"
    for lead in GEFS_LEADS
    for stat in ("mean", "spread")
    for name in OPERATIONAL_GEFS_BASE_FEATURES
) + tuple(f"gefs_control_boundary_layer_height_m_f{lead:03d}" for lead in GEFS_LEADS)
_FIELD_OUTPUT_NAMES = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "relative_humidity_pct",
    "u_wind_10m": "wind_u_ms",
    "v_wind_10m": "wind_v_ms",
    "pblh": "boundary_layer_height_m",
}


def _converted_value(field_name: str, stat: str, value: float) -> float:
    if field_name == "temperature_2m" and stat == "mean":
        return value - 273.15
    return value


def _point_values(
    messages: dict[str, GEFSFieldMessage],
    *,
    stat: str,
    longitude: float,
    latitude: float,
    wgrib2_executable: Path,
    point_query: Callable[..., GRIBPointValue],
) -> tuple[dict[str, float], float, float]:
    values: dict[str, float] = {}
    points: list[GRIBPointValue] = []
    with TemporaryDirectory(prefix="operational-gefs-") as directory:
        temp_dir = Path(directory)
        for field_name, message in messages.items():
            message_path = temp_dir / f"{stat}_{field_name}.grib2"
            message_path.write_bytes(message.content)
            point = point_query(message_path, longitude, latitude, wgrib2_executable)
            points.append(point)
            values[_FIELD_OUTPUT_NAMES[field_name]] = _converted_value(field_name, stat, point.value)
    locations = {(point.longitude, point.latitude) for point in points}
    if len(locations) != 1:
        raise ValueError("operational GEFS fields use inconsistent nearest grid points")
    return values, *locations.pop()


def _extract_run(
    run_time: datetime,
    *,
    longitude: float,
    latitude: float,
    wgrib2_executable: Path,
    downloader: Callable[..., dict[str, GEFSFieldMessage]],
    point_query: Callable[..., GRIBPointValue],
) -> dict[str, object]:
    values: dict[str, object] = {"gefs_run_utc": run_time}
    provenance: list[dict[str, object]] = []
    grid_locations: set[tuple[float, float]] = set()
    for lead_hours in GEFS_LEADS:
        values[f"gefs_forecast_valid_utc_f{lead_hours:03d}"] = run_time + timedelta(hours=lead_hours)
        for product, stat in (("geavg", "mean"), ("gespr", "spread"), ("gec00", "control")):
            messages = downloader(run_time, product=product, lead_hours=lead_hours)
            product_values, grid_longitude, grid_latitude = _point_values(
                messages,
                stat=stat,
                longitude=longitude,
                latitude=latitude,
                wgrib2_executable=wgrib2_executable,
                point_query=point_query,
            )
            grid_locations.add((grid_longitude, grid_latitude))
            for name, value in product_values.items():
                values[f"gefs_{stat}_{name}_f{lead_hours:03d}"] = value
            provenance.extend(
                {
                    "product": message.product,
                    "field": message.feature_name,
                    "lead_hours": message.lead_hours,
                    "source_url": message.source_url,
                    "index_url": message.index_url,
                    "byte_range": list(message.byte_range),
                    "sha256": message.sha256,
                }
                for message in messages.values()
            )
    if len(grid_locations) != 1:
        raise ValueError("operational GEFS products use inconsistent nearest grid points")
    values["gefs_grid_longitude"], values["gefs_grid_latitude"] = grid_locations.pop()
    values["gefs_source_manifest_json"] = json.dumps(
        {"run_utc": run_time.isoformat(), "messages": provenance},
        sort_keys=True,
        separators=(",", ":"),
    )
    return values


def enrich_event_rows_with_operational_gefs(
    event_rows: pd.DataFrame,
    *,
    longitude: float,
    latitude: float,
    wgrib2_executable: Path,
    downloader: Callable[..., dict[str, GEFSFieldMessage]] = download_operational_gefs_surface_messages,
    point_query: Callable[..., GRIBPointValue] = query_grib_point,
) -> pd.DataFrame:
    """Attach GEFS mean/spread values using only forecasts issued before each row."""
    if "timestamp" not in event_rows:
        raise ValueError("event_rows must contain timestamp")
    result = event_rows.copy()
    timestamps = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    selections = [select_operational_gefs_run(value.to_pydatetime()) for value in timestamps]
    cache: dict[datetime, dict[str, object]] = {}
    errors: dict[datetime, str] = {}
    for run_time in sorted({run_time for run_time, _ in selections}):
        try:
            cache[run_time] = _extract_run(
                run_time,
                longitude=longitude,
                latitude=latitude,
                wgrib2_executable=wgrib2_executable,
                downloader=downloader,
                point_query=point_query,
            )
        except (OSError, RuntimeError, ValueError) as error:
            errors[run_time] = str(error)
    for feature_name in OPERATIONAL_GEFS_FEATURES:
        result[feature_name] = [cache.get(run_time, {}).get(feature_name, np.nan) for run_time, _ in selections]
    result["gefs_run_utc"] = [run_time for run_time, _ in selections]
    for lead_hours in GEFS_LEADS:
        result[f"gefs_forecast_valid_utc_f{lead_hours:03d}"] = [
            valid_times[lead_hours] for _, valid_times in selections
        ]
    for name in ("gefs_grid_longitude", "gefs_grid_latitude", "gefs_source_manifest_json"):
        result[name] = [cache.get(run_time, {}).get(name, None) for run_time, _ in selections]
    result["gefs_extraction_error"] = [errors.get(run_time, "") for run_time, _ in selections]
    return result
