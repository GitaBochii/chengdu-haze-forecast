"""Prepare NOAA Integrated Surface Database hourly observations for Chengdu."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

CHENGDU_LONGITUDE = 104.0665
CHENGDU_LATITUDE = 30.5723
EARTH_RADIUS_KM = 6371.0088
REQUIRED_COLUMNS = ("DATE",)
OPTIONAL_COLUMNS = ("TMP", "DEW", "SLP", "MAA", "VIS", "AA1", "GF1", "WND")
TRUSTED_QUALITY_CODES = frozenset({"0", "1", "4", "5", "9"})
NOAA_ISD_BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access"


def haversine_km(longitude: pd.Series, latitude: pd.Series) -> pd.Series:
    """Return great-circle distances from Chengdu city center in kilometers."""
    longitude_numeric = pd.to_numeric(longitude, errors="coerce")
    latitude_numeric = pd.to_numeric(latitude, errors="coerce")
    valid_coordinates = longitude_numeric.between(-180, 180) & latitude_numeric.between(-90, 90)
    longitude_rad = np.radians(longitude_numeric)
    latitude_rad = np.radians(latitude_numeric)
    center_longitude_rad = np.radians(CHENGDU_LONGITUDE)
    center_latitude_rad = np.radians(CHENGDU_LATITUDE)
    delta_longitude = longitude_rad - center_longitude_rad
    delta_latitude = latitude_rad - center_latitude_rad
    haversine = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(center_latitude_rad) * np.cos(latitude_rad) * np.sin(delta_longitude / 2) ** 2
    )
    distance_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine))
    return distance_km.where(valid_coordinates)


def nearby_stations(stations: pd.DataFrame, maximum_distance_km: float = 60.0) -> pd.DataFrame:
    """Keep stations at most ``maximum_distance_km`` from Chengdu city center."""
    missing_columns = {"longitude", "latitude"} - set(stations.columns)
    if missing_columns:
        raise ValueError(f"missing station columns: {sorted(missing_columns)}")
    nearby = stations.copy()
    nearby["distance_km"] = haversine_km(nearby["longitude"], nearby["latitude"])
    return nearby.loc[nearby["distance_km"].le(maximum_distance_km)].reset_index(drop=True)


def _numeric_component(values: pd.Series, component: int) -> pd.Series:
    """Parse one comma-separated ISD field component as a numeric series."""
    component_values = values.astype("string").str.split(",").str.get(component)
    return pd.to_numeric(component_values, errors="coerce")


def _quality_component(values: pd.Series, component: int) -> pd.Series:
    """Parse one comma-separated ISD quality component."""
    return values.astype("string").str.split(",").str.get(component).str.strip()


def _validated(
    values: pd.Series,
    *,
    quality_codes: pd.Series,
    missing_values: tuple[int, ...],
    minimum: float,
    maximum: float,
) -> pd.Series:
    """Replace ISD sentinels and values outside a conservative physical range."""
    invalid_value = values.isin(missing_values) | values.lt(minimum) | values.gt(maximum)
    return values.mask(invalid_value | ~quality_codes.isin(TRUSTED_QUALITY_CODES))


def prepare_noaa_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse ISD fields and select the best observation available for each local hour."""
    missing_columns = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"missing NOAA ISD columns: {sorted(missing_columns)}")

    fields = {
        column: raw[column]
        if column in raw
        else pd.Series(pd.NA, index=raw.index, dtype="string")
        for column in OPTIONAL_COLUMNS
    }
    local_observation_time = pd.to_datetime(raw["DATE"], utc=True, errors="coerce")
    local_observation_time = local_observation_time.dt.tz_convert("Asia/Shanghai")
    timestamp = local_observation_time.dt.ceil("h")
    temperature_c = _validated(
        _numeric_component(fields["TMP"], 0),
        quality_codes=_quality_component(fields["TMP"], 1),
        missing_values=(9999,),
        minimum=-932,
        maximum=618,
    ) / 10
    dewpoint_c = _validated(
        _numeric_component(fields["DEW"], 0),
        quality_codes=_quality_component(fields["DEW"], 1),
        missing_values=(9999,),
        minimum=-1000,
        maximum=700,
    ) / 10
    sea_level_pressure_hpa = _validated(
        _numeric_component(fields["SLP"], 0),
        quality_codes=_quality_component(fields["SLP"], 1),
        missing_values=(99999,),
        minimum=8000,
        maximum=11000,
    ) / 10
    pressure_change_hpa = _validated(
        _numeric_component(fields["MAA"], 1),
        quality_codes=_quality_component(fields["MAA"], 2),
        missing_values=(9999,),
        minimum=0,
        maximum=500,
    ) / 10
    visibility_m = _validated(
        _numeric_component(fields["VIS"], 0),
        quality_codes=_quality_component(fields["VIS"], 1),
        missing_values=(999999,),
        minimum=0,
        maximum=160000,
    )
    precipitation_mm = _validated(
        _numeric_component(fields["AA1"], 1),
        quality_codes=_quality_component(fields["AA1"], 3),
        missing_values=(9999,),
        minimum=0,
        maximum=5000,
    ) / 10
    cloud_cover_oktas = _validated(
        _numeric_component(fields["GF1"], 0),
        quality_codes=_quality_component(fields["GF1"], 1),
        missing_values=(99,),
        minimum=0,
        maximum=9,
    )
    wind_direction_deg = _validated(
        _numeric_component(fields["WND"], 0),
        quality_codes=_quality_component(fields["WND"], 1),
        missing_values=(999,),
        minimum=0,
        maximum=360,
    )
    wind_speed_ms = _validated(
        _numeric_component(fields["WND"], 3),
        quality_codes=_quality_component(fields["WND"], 4),
        missing_values=(9999,),
        minimum=0,
        maximum=1000,
    ) / 10

    prepared = pd.DataFrame(
        {
            "timestamp": timestamp,
            "_local_observation_time": local_observation_time,
            "noaa_temperature_c": temperature_c,
            "noaa_dewpoint_c": dewpoint_c,
            "noaa_sea_level_pressure_hpa": sea_level_pressure_hpa,
            "noaa_pressure_change_hpa": pressure_change_hpa,
            "noaa_visibility_m": visibility_m,
            "noaa_precipitation_mm": precipitation_mm,
            "noaa_cloud_cover_oktas": cloud_cover_oktas,
            "noaa_wind_direction_deg": wind_direction_deg,
            "noaa_wind_speed_ms": wind_speed_ms,
        }
    )
    prepared = prepared.loc[prepared["timestamp"].notna()]
    weather_columns = [column for column in prepared if column.startswith("noaa_")]
    prepared["_valid_weather_field_count"] = prepared[weather_columns].notna().sum(axis=1)
    prepared = prepared.sort_values(
        ["timestamp", "_valid_weather_field_count", "_local_observation_time"],
        ascending=[True, False, False],
        kind="stable",
    )
    prepared = prepared.drop_duplicates("timestamp", keep="first")
    return prepared.drop(columns=["_local_observation_time", "_valid_weather_field_count"]).reset_index(
        drop=True
    )


def annual_csv_url(usaf: str, wban: str, year: int) -> str:
    """Return the NCEI Global Hourly CSV URL for one station-year."""
    return f"{NOAA_ISD_BASE_URL}/{int(year)}/{usaf!s}{wban!s}.csv"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_annual_csv(url: str, destination: Path, timeout_s: int = 60) -> str:
    """Cache one raw NCEI CSV and return the SHA-256 of its exact bytes."""
    if destination.exists():
        return _file_sha256(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    try:
        temporary.write_bytes(response.content)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return _file_sha256(destination)


def _expected_hours(year: int) -> int:
    return len(
        pd.date_range(
            f"{year}-01-01",
            f"{year + 1}-01-01",
            freq="h",
            inclusive="left",
            tz="Asia/Shanghai",
        )
    )


def _normalise_timestamps(hourly: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in hourly.columns:
        raise ValueError("prepared hourly data has no timestamp column")
    normalised = hourly.copy()
    timestamp = pd.to_datetime(normalised["timestamp"], errors="coerce")
    if timestamp.dt.tz is None:
        timestamp = timestamp.dt.tz_localize("Asia/Shanghai")
    else:
        timestamp = timestamp.dt.tz_convert("Asia/Shanghai")
    normalised["timestamp"] = timestamp.dt.floor("h")
    return normalised.loc[normalised["timestamp"].notna()].drop_duplicates("timestamp", keep="last")


def _source_fields(source: Mapping[str, Any] | None) -> dict[str, str | None]:
    source = source or {}
    return {
        "source_url": source.get("source_url", source.get("url")),
        "sha256": source.get("sha256"),
    }


def audit_station_hourly(
    station: Mapping[str, Any],
    yearly_hourly: Mapping[int, pd.DataFrame],
    *,
    years: Sequence[int],
    minimum_coverage: float = 0.70,
    annual_sources: Mapping[int, Mapping[str, Any]] | None = None,
    annual_failures: Mapping[int, str] | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Audit one station, rejecting it unless every requested year clears coverage."""
    annual_sources = annual_sources or {}
    annual_failures = annual_failures or {}
    requested_years = [int(year) for year in years]
    annual_reports: dict[str, dict[str, Any]] = {}
    failure_reasons: list[str] = []
    accepted: list[pd.DataFrame] = []

    for year in requested_years:
        source = _source_fields(annual_sources.get(year))
        expected_hours = _expected_hours(year)
        failure_reason = annual_failures.get(year)
        hourly = yearly_hourly.get(year)
        if failure_reason is not None and hourly is None:
            annual_reports[str(year)] = {
                "year": year,
                "expected_hours": expected_hours,
                "observed_hours": 0,
                "coverage": 0.0,
                "variable_non_null_rate": {},
                **source,
                "failure_reason": failure_reason,
            }
            failure_reasons.append(f"{year}: {failure_reason}")
            continue
        if hourly is None:
            failure_reason = "no prepared hourly data"
            annual_reports[str(year)] = {
                "year": year,
                "expected_hours": expected_hours,
                "observed_hours": 0,
                "coverage": 0.0,
                "variable_non_null_rate": {},
                **source,
                "failure_reason": failure_reason,
            }
            failure_reasons.append(f"{year}: {failure_reason}")
            continue

        try:
            normalised = _normalise_timestamps(hourly)
        except (TypeError, ValueError) as exc:
            failure_reason = f"invalid prepared hourly data: {exc}"
            annual_reports[str(year)] = {
                "year": year,
                "expected_hours": expected_hours,
                "observed_hours": 0,
                "coverage": 0.0,
                "variable_non_null_rate": {},
                **source,
                "failure_reason": failure_reason,
            }
            failure_reasons.append(f"{year}: {failure_reason}")
            continue

        annual = normalised.loc[normalised["timestamp"].dt.year.eq(year)].copy()
        observed_hours = int(annual["timestamp"].nunique())
        coverage = observed_hours / expected_hours
        variable_columns = [column for column in annual.columns if column != "timestamp"]
        non_null_rate = {
            column: float(annual[column].notna().mean()) if observed_hours else 0.0
            for column in variable_columns
        }
        report: dict[str, Any] = {
            "year": year,
            "expected_hours": expected_hours,
            "observed_hours": observed_hours,
            "coverage": coverage,
            "variable_non_null_rate": non_null_rate,
            **source,
            "failure_reason": None,
        }
        if failure_reason is not None:
            report["failure_reason"] = failure_reason
            failure_reasons.append(f"{year}: {failure_reason}")
        elif coverage < minimum_coverage:
            failure_reason = f"coverage {coverage:.3f} is below minimum {minimum_coverage:.3f}"
            report["failure_reason"] = failure_reason
            failure_reasons.append(f"{year}: {failure_reason}")
        else:
            accepted.append(annual)
        annual_reports[str(year)] = report

    eligible = not failure_reasons and len(accepted) == len(requested_years)
    report = {
        "station": dict(station),
        "eligible": eligible,
        "minimum_coverage": float(minimum_coverage),
        "years": annual_reports,
        "failure_reasons": failure_reasons,
    }
    if not eligible:
        return None, report
    return pd.concat(accepted, ignore_index=True).sort_values("timestamp").reset_index(drop=True), report


def aggregate_station_hourly(station_hourly: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Return the city-level median for every meteorological variable and local hour."""
    if not station_hourly:
        return pd.DataFrame(columns=["timestamp"])
    combined = pd.concat(station_hourly, ignore_index=True)
    numeric_columns = [
        column for column in combined.columns if column != "timestamp" and pd.api.types.is_numeric_dtype(combined[column])
    ]
    if not numeric_columns:
        return combined[["timestamp"]].drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    return (
        combined.groupby("timestamp", as_index=False)[numeric_columns]
        .median()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _station_identifier(station: Mapping[str, Any], key: str, width: int) -> str:
    value = station.get(key)
    if value is None or pd.isna(value):
        raise ValueError(f"station metadata is missing {key}")
    value = str(value).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    if not value.isdigit():
        raise ValueError(f"station metadata has invalid {key}: {value!r}")
    return value.zfill(width)


def audit_noaa_isd(
    stations: pd.DataFrame,
    raw_dir: Path,
    *,
    years: Sequence[int] = tuple(range(2015, 2022)),
    maximum_distance_km: float = 60.0,
    minimum_coverage: float = 0.70,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Download, prepare, audit, and aggregate qualified nearby NOAA ISD stations."""
    nearby = nearby_stations(stations, maximum_distance_km=maximum_distance_km)
    station_reports: list[dict[str, Any]] = []
    qualified: list[pd.DataFrame] = []
    requested_years = [int(year) for year in years]
    if not requested_years:
        raise ValueError("years must not be empty")
    boundary_utc_year_by_target = {year: year - 1 for year in requested_years}
    source_years = sorted(set(requested_years) | set(boundary_utc_year_by_target.values()))

    for _, station_row in nearby.iterrows():
        station = station_row.to_dict()
        try:
            usaf = _station_identifier(station, "usaf", 6)
            wban = _station_identifier(station, "wban", 5)
        except ValueError as exc:
            _, station_report = audit_station_hourly(
                station,
                {},
                years=requested_years,
                minimum_coverage=minimum_coverage,
                annual_failures={year: str(exc) for year in requested_years},
            )
            station_report["variable_non_null_rate"] = {}
            station_report["boundary_sources"] = {
                str(target_year): {
                    "source_utc_year": boundary_utc_year,
                    "source_url": None,
                    "sha256": None,
                    "failure_reason": str(exc),
                }
                for target_year, boundary_utc_year in boundary_utc_year_by_target.items()
            }
            station_reports.append(station_report)
            continue

        prepared_by_utc_year: dict[int, pd.DataFrame] = {}
        source_records: dict[int, dict[str, str | None]] = {}
        annual_failures: dict[int, str] = {}
        for year in source_years:
            url = annual_csv_url(usaf, wban, year)
            destination = raw_dir / str(year) / f"{usaf}{wban}.csv"
            source = {"source_url": url, "sha256": None, "failure_reason": None}
            source_records[year] = source
            try:
                source["sha256"] = download_annual_csv(url, destination)
                prepared_by_utc_year[year] = prepare_noaa_hourly(pd.read_csv(destination, dtype="string"))
            except (OSError, pd.errors.ParserError, requests.RequestException, ValueError) as exc:
                source["failure_reason"] = f"{type(exc).__name__}: {exc}"

        annual_sources = {
            year: {
                "source_url": source_records[year]["source_url"],
                "sha256": source_records[year]["sha256"],
            }
            for year in requested_years
        }
        for year in requested_years:
            source_failure = source_records[year]["failure_reason"]
            if source_failure is not None:
                annual_failures[year] = source_failure

        boundary_sources: dict[str, dict[str, int | str | None]] = {}
        for target_year, boundary_utc_year in boundary_utc_year_by_target.items():
            boundary_source = source_records[boundary_utc_year]
            boundary_sources[str(target_year)] = {
                "source_utc_year": boundary_utc_year,
                "source_url": boundary_source["source_url"],
                "sha256": boundary_source["sha256"],
                "failure_reason": boundary_source["failure_reason"],
            }
            if boundary_source["failure_reason"] is not None:
                boundary_failure = (
                    f"boundary source {boundary_utc_year} failed: "
                    f"{boundary_source['failure_reason']}"
                )
                existing_failure = annual_failures.get(target_year)
                annual_failures[target_year] = (
                    f"{existing_failure}; {boundary_failure}"
                    if existing_failure is not None
                    else boundary_failure
                )

        if prepared_by_utc_year:
            all_prepared = pd.concat(prepared_by_utc_year.values(), ignore_index=True)
            local_timestamp = pd.to_datetime(all_prepared["timestamp"], errors="coerce")
            if local_timestamp.dt.tz is None:
                local_timestamp = local_timestamp.dt.tz_localize("Asia/Shanghai")
            else:
                local_timestamp = local_timestamp.dt.tz_convert("Asia/Shanghai")
            yearly_hourly = {
                year: all_prepared.loc[local_timestamp.dt.year.eq(year)].copy()
                for year in requested_years
            }
        else:
            yearly_hourly = {}

        prepared, station_report = audit_station_hourly(
            station,
            yearly_hourly,
            years=requested_years,
            minimum_coverage=minimum_coverage,
            annual_sources=annual_sources,
            annual_failures=annual_failures,
        )
        station_report["boundary_sources"] = boundary_sources
        station_reports.append(station_report)
        if prepared is not None:
            qualified.append(prepared)

    city_hourly = aggregate_station_hourly(qualified) if qualified else None
    report = {
        "years": requested_years,
        "maximum_distance_km": float(maximum_distance_km),
        "minimum_coverage": float(minimum_coverage),
        "nearby_station_count": len(nearby),
        "eligible_station_count": len(qualified),
        "city_hourly_rows": 0 if city_hourly is None else len(city_hourly),
        "stations": station_reports,
    }
    return city_hourly, report
