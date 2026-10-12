"""Build bounded GEFSv12 surface-ensemble features for labelled event rows."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from haze_forecast.gefs_features import select_reforecast_run_and_lead, summarise_ensemble
from haze_forecast.gefs_reforecast import (
    GEFS_REFORECAST_MEMBERS,
    GEFS_SURFACE_VARIABLES,
    GEFSReforecastIndexCache,
    GEFSReforecastMessage,
    download_gefs_reforecast_message,
)
from haze_forecast.grib import GRIBPointValue, query_grib_point

_SUMMARY_NAMES = ("median", "std", "p10", "p90")
_RH_FEATURE_PREFIX = "gefs_relative_humidity"
_RH_UNIT = "pct"


def _summary_feature_columns(variable: str) -> tuple[str, ...]:
    definition = GEFS_SURFACE_VARIABLES[variable]
    return tuple(
        f"{definition.feature_prefix}_{summary_name}_{definition.unit}"
        for summary_name in _SUMMARY_NAMES
    )


GEFS_SURFACE_FEATURE_COLUMNS = tuple(
    column
    for variable in GEFS_SURFACE_VARIABLES
    for column in _summary_feature_columns(variable)
) + tuple(f"{_RH_FEATURE_PREFIX}_{summary_name}_{_RH_UNIT}" for summary_name in _SUMMARY_NAMES)


def _relative_humidity_percent(
    *, temperature_k: float, specific_humidity_kg_kg: float, surface_pressure_pa: float
) -> float:
    """Calculate RH from GEFS temperature, specific humidity, and surface pressure."""
    if not all(math.isfinite(value) for value in (temperature_k, specific_humidity_kg_kg, surface_pressure_pa)):
        raise ValueError("GEFSv12 humidity inputs must be finite")
    if temperature_k <= 0 or not 0 <= specific_humidity_kg_kg < 1 or surface_pressure_pa <= 0:
        raise ValueError("GEFSv12 humidity inputs are physically invalid")
    epsilon = 0.622
    vapour_pressure_pa = (
        specific_humidity_kg_kg * surface_pressure_pa
        / (epsilon + (1.0 - epsilon) * specific_humidity_kg_kg)
    )
    temperature_c = temperature_k - 273.15
    saturation_vapour_pressure_pa = 611.2 * math.exp(
        17.67 * temperature_c / (temperature_c + 243.5)
    )
    return 100.0 * vapour_pressure_pa / saturation_vapour_pressure_pa


def _validate_message_identity(
    message: GEFSReforecastMessage,
    *,
    run_time: datetime,
    member: str,
    variable: str,
    lead_hours: int,
) -> None:
    """Reject injected downloaders whose evidence does not match the request."""
    message_run_time = message.run_time
    if message_run_time.tzinfo is None or message_run_time.utcoffset() is None:
        raise ValueError("GEFSv12 downloader returned a naive run_time identity")
    if (
        message_run_time.astimezone(UTC) != run_time.astimezone(UTC)
        or message.member != member
        or message.variable != variable
        or message.lead_hours != lead_hours
    ):
        raise ValueError("GEFSv12 downloader identity does not match the requested message")


def _message_provenance(message: GEFSReforecastMessage) -> dict[str, object]:
    return {
        "variable": message.variable,
        "member": message.member,
        "source_url": message.source_url,
        "index_url": message.index_url,
        "byte_range": list(message.byte_range),
        "sha256": message.sha256,
        "run_utc": message.run_time.astimezone(UTC).isoformat(),
        "lead_hours": message.lead_hours,
    }


def _selection_is_time_safe(
    *, issue_time: pd.Timestamp, run_time: datetime, lead_hours: int, horizon_hours: int
) -> datetime:
    valid_time = run_time + timedelta(hours=lead_hours)
    if not (run_time <= issue_time.to_pydatetime() < valid_time):
        raise RuntimeError("a GEFSv12 feature must satisfy run <= timestamp < valid")
    if valid_time < issue_time.to_pydatetime() + timedelta(hours=horizon_hours):
        raise RuntimeError("a GEFSv12 feature uses weather valid before its requested horizon")
    return valid_time


def enrich_event_rows_with_gefs_surface(
    event_rows: pd.DataFrame,
    *,
    longitude: float,
    latitude: float,
    wgrib2_executable: Path,
    horizon_hours: int = 6,
    timeout_s: int = 60,
    variables: tuple[str, ...] = tuple(GEFS_SURFACE_VARIABLES),
    max_member_workers: int = 2,
    downloader: Callable[..., GEFSReforecastMessage] = download_gefs_reforecast_message,
    point_query: Callable[..., GRIBPointValue] = query_grib_point,
) -> pd.DataFrame:
    """Attach time-safe, five-member GEFS surface distributions without retaining GRIB files."""
    if "timestamp" not in event_rows:
        raise ValueError("event_rows must contain timestamp")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if (
        isinstance(max_member_workers, bool)
        or not isinstance(max_member_workers, int)
        or not 1 <= max_member_workers <= len(GEFS_REFORECAST_MEMBERS)
    ):
        raise ValueError("max_member_workers must be an integer from 1 through 5")
    if isinstance(horizon_hours, bool) or not isinstance(horizon_hours, int) or horizon_hours < 6:
        raise ValueError("GEFSv12 surface features require a horizon of at least 6 hours")
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("GEFSv12 variables must be a non-empty unique sequence")
    unknown_variables = set(variables).difference(GEFS_SURFACE_VARIABLES)
    if unknown_variables:
        raise ValueError(f"unsupported GEFSv12 surface variables: {sorted(unknown_variables)}")
    timestamps = pd.to_datetime(event_rows["timestamp"])
    if timestamps.dt.tz is None:
        raise ValueError("event_rows timestamps must include timezone information")
    result = event_rows.copy()
    issue_times = timestamps.dt.tz_convert("UTC")
    selections = issue_times.map(
        lambda timestamp: select_reforecast_run_and_lead(
            timestamp.to_pydatetime(), horizon_hours=horizon_hours
        )
    )
    result["gefs_run_utc"] = selections.map(lambda selection: selection[0])
    result["gefs_lead_hours"] = selections.map(lambda selection: selection[1])

    extracted: dict[tuple[pd.Timestamp, int], dict[str, object]] = {}
    index_cache = GEFSReforecastIndexCache()
    unique_selections = sorted(set(selections), key=lambda selection: (selection[0], selection[1]))
    for run_time, lead_hours in unique_selections:
        points: dict[str, dict[str, GRIBPointValue]] = {variable: {} for variable in variables}
        provenance: list[dict[str, object]] = []
        with TemporaryDirectory(prefix="gefs-v12-surface-") as directory:
            temporary_directory = Path(directory)
            for variable in variables:
                definition = GEFS_SURFACE_VARIABLES[variable]
                def extract_member(
                    member: str,
                    task_run_time: datetime,
                    task_lead_hours: int,
                    task_variable: str,
                    task_definition: object,
                    task_directory: Path,
                ) -> tuple[str, GRIBPointValue, dict[str, object]]:
                    download_kwargs: dict[str, object] = {
                        "member": member,
                        "variable": task_variable,
                        "lead_hours": task_lead_hours,
                        "level_hint": task_definition.level_hint,
                        "timeout_s": timeout_s,
                    }
                    if downloader is download_gefs_reforecast_message:
                        download_kwargs["index_cache"] = index_cache
                    message = downloader(
                        task_run_time,
                        **download_kwargs,
                    )
                    _validate_message_identity(
                        message,
                        run_time=task_run_time,
                        member=member,
                        variable=task_variable,
                        lead_hours=task_lead_hours,
                    )
                    message_path = task_directory / f"{task_variable}_{member}.grib2"
                    message_path.write_bytes(message.content)
                    point = point_query(
                        message_path, longitude, latitude, wgrib2_executable, timeout_s=timeout_s
                    )
                    return member, point, _message_provenance(message)

                member_results: dict[str, tuple[GRIBPointValue, dict[str, object]]] = {}
                with ThreadPoolExecutor(max_workers=max_member_workers) as executor:
                    futures = {
                        executor.submit(
                            extract_member,
                            member,
                            run_time,
                            lead_hours,
                            variable,
                            definition,
                            temporary_directory,
                        ): member
                        for member in GEFS_REFORECAST_MEMBERS
                    }
                    for future in as_completed(futures):
                        member = futures[future]
                        try:
                            extracted_member, point, member_provenance = future.result()
                        except Exception as error:
                            for pending_future in futures:
                                pending_future.cancel()
                            error.add_note(
                                "GEFSv12 surface extraction failed for "
                                f"variable={variable}, member={member}"
                            )
                            raise
                        member_results[extracted_member] = (point, member_provenance)
                for member in GEFS_REFORECAST_MEMBERS:
                    point, member_provenance = member_results[member]
                    points[variable][member] = point
                    provenance.append(member_provenance)
        locations = {
            (point.longitude, point.latitude)
            for variable_points in points.values()
            for point in variable_points.values()
        }
        if len(locations) != 1:
            raise ValueError("GEFSv12 ensemble members use inconsistent nearest grid points")
        grid_longitude, grid_latitude = locations.pop()
        features: dict[str, object] = {
            "grid_longitude": grid_longitude,
            "grid_latitude": grid_latitude,
            "source_manifest_json": json.dumps(
                {
                    "run_utc": run_time.astimezone(UTC).isoformat(),
                    "lead_hours": lead_hours,
                    "messages": provenance,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
        for variable in variables:
            summary = summarise_ensemble(
                [points[variable][member].value for member in GEFS_REFORECAST_MEMBERS]
            )
            for summary_name, value in summary.items():
                definition = GEFS_SURFACE_VARIABLES[variable]
                features[f"{definition.feature_prefix}_{summary_name}_{definition.unit}"] = value
        if {"tmp_2m", "spfh_2m", "pres_sfc"}.issubset(variables):
            relative_humidity_values = [
                _relative_humidity_percent(
                    temperature_k=points["tmp_2m"][member].value,
                    specific_humidity_kg_kg=points["spfh_2m"][member].value,
                    surface_pressure_pa=points["pres_sfc"][member].value,
                )
                for member in GEFS_REFORECAST_MEMBERS
            ]
            for summary_name, value in summarise_ensemble(relative_humidity_values).items():
                features[f"{_RH_FEATURE_PREFIX}_{summary_name}_{_RH_UNIT}"] = value
        extracted[(pd.Timestamp(run_time), lead_hours)] = features

    keys = list(zip(result["gefs_run_utc"], result["gefs_lead_hours"], strict=True))
    result["gefs_forecast_valid_utc"] = [
        _selection_is_time_safe(
            issue_time=issue_time,
            run_time=run_time,
            lead_hours=int(lead_hours),
            horizon_hours=horizon_hours,
        )
        for issue_time, (run_time, lead_hours) in zip(issue_times, keys, strict=True)
    ]
    values = [
        extracted[(pd.Timestamp(run_time), int(lead_hours))] for run_time, lead_hours in keys
    ]
    result["gefs_grid_longitude"] = [value["grid_longitude"] for value in values]
    result["gefs_grid_latitude"] = [value["grid_latitude"] for value in values]
    result["gefs_source_manifest_json"] = [value["source_manifest_json"] for value in values]
    for variable in variables:
        for column in _summary_feature_columns(variable):
            result[column] = [value[column] for value in values]
    if {"tmp_2m", "spfh_2m", "pres_sfc"}.issubset(variables):
        for summary_name in _SUMMARY_NAMES:
            column = f"{_RH_FEATURE_PREFIX}_{summary_name}_{_RH_UNIT}"
            result[column] = [value[column] for value in values]
    return result


def enrich_event_rows_with_gefs_pbl(
    event_rows: pd.DataFrame,
    *,
    longitude: float,
    latitude: float,
    wgrib2_executable: Path,
    horizon_hours: int = 6,
    timeout_s: int = 60,
    max_member_workers: int = 2,
    downloader: Callable[..., GEFSReforecastMessage] = download_gefs_reforecast_message,
    point_query: Callable[..., GRIBPointValue] = query_grib_point,
) -> pd.DataFrame:
    """Backward-compatible PBL-only wrapper around surface ensemble extraction."""
    return enrich_event_rows_with_gefs_surface(
        event_rows,
        longitude=longitude,
        latitude=latitude,
        wgrib2_executable=wgrib2_executable,
        horizon_hours=horizon_hours,
        timeout_s=timeout_s,
        variables=("pbl_hgt",),
        max_member_workers=max_member_workers,
        downloader=downloader,
        point_query=point_query,
    )
