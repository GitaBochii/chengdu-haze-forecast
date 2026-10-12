"""Turn no-leak NOAA GFS message slices into Chengdu surface-weather features."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from haze_forecast.forecast_sources import GFSFieldMessage, download_gfs_surface_messages
from haze_forecast.grib import GRIBPointValue, query_grib_point


@dataclass(frozen=True)
class GFSPointForecast:
    """One issued GFS forecast, sampled at the closest grid point to Chengdu."""

    run_time: datetime
    valid_at: datetime
    lead_hours: int
    grid_longitude: float
    grid_latitude: float
    temperature_c: float
    relative_humidity_pct: float
    wind_u_ms: float
    wind_v_ms: float
    boundary_layer_height_m: float


def extract_gfs_surface_forecast_at_point(
    run_time: datetime,
    lead_hours: int,
    longitude: float,
    latitude: float,
    wgrib2_executable: Path,
    *,
    message_downloader: Callable[..., dict[str, GFSFieldMessage]] = download_gfs_surface_messages,
    point_query: Callable[..., GRIBPointValue] = query_grib_point,
    message_workers: int = 1,
    message_timeout_s: int = 45,
) -> GFSPointForecast:
    """Download five Range slices and decode their nearest Chengdu grid-point values."""
    if run_time.tzinfo is None or run_time.utcoffset() != timedelta(0):
        raise ValueError("run_time must be timezone-aware UTC")
    messages = message_downloader(
        run_time,
        lead_hours,
        max_workers=message_workers,
        timeout_s=message_timeout_s,
    )
    with TemporaryDirectory(prefix="gfs-point-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        points: dict[str, GRIBPointValue] = {}
        for feature_name, message in messages.items():
            message_path = temporary_path / f"{feature_name}.grb2"
            message_path.write_bytes(message.content)
            points[feature_name] = point_query(
                message_path,
                longitude,
                latitude,
                wgrib2_executable,
            )

    required_names = {
        "temperature_2m",
        "relative_humidity_2m",
        "u_wind_10m",
        "v_wind_10m",
        "pblh",
    }
    missing_names = required_names - set(points)
    if missing_names:
        raise ValueError(f"missing decoded GFS fields: {sorted(missing_names)}")
    grid_locations = {(point.longitude, point.latitude) for point in points.values()}
    if len(grid_locations) != 1:
        raise ValueError("decoded GFS fields use inconsistent nearest grid points")
    grid_longitude, grid_latitude = grid_locations.pop()
    return GFSPointForecast(
        run_time=run_time.astimezone(UTC),
        valid_at=run_time.astimezone(UTC) + timedelta(hours=lead_hours),
        lead_hours=lead_hours,
        grid_longitude=grid_longitude,
        grid_latitude=grid_latitude,
        temperature_c=points["temperature_2m"].value - 273.15,
        relative_humidity_pct=points["relative_humidity_2m"].value,
        wind_u_ms=points["u_wind_10m"].value,
        wind_v_ms=points["v_wind_10m"].value,
        boundary_layer_height_m=points["pblh"].value,
    )
