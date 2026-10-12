"""Time-safe access helpers for NOAA's operational GEFS archive."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import requests

from haze_forecast.forecast_sources import (
    parse_gfs_index,
    range_for_gfs_index_entry,
)

GEFS_AWS_ROOT = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_CYCLE_HOURS = 6
GEFS_LEADS = (6, 24)
GEFS_PRODUCT_LAYOUTS = {
    "geavg": ("pgrb2ap5", "pgrb2a"),
    "gespr": ("pgrb2ap5", "pgrb2a"),
    "gec00": ("pgrb2bp5", "pgrb2b"),
}
GEFS_PRODUCTS = tuple(GEFS_PRODUCT_LAYOUTS)
GEFS_FIELD_DEFINITIONS = {
    "temperature_2m": ("TMP", "2 m above ground"),
    "relative_humidity_2m": ("RH", "2 m above ground"),
    "u_wind_10m": ("UGRD", "10 m above ground"),
    "v_wind_10m": ("VGRD", "10 m above ground"),
    "pblh": ("HPBL", "surface"),
}
_ENSEMBLE_FIELDS = ("temperature_2m", "relative_humidity_2m", "u_wind_10m", "v_wind_10m")


@dataclass(frozen=True)
class GEFSFieldMessage:
    """One identity-checked HTTP range response from an operational GEFS object."""

    product: str
    feature_name: str
    run_time: datetime
    lead_hours: int
    valid_time: datetime
    source_url: str
    index_url: str
    byte_range: tuple[int, int]
    content: bytes
    sha256: str


def _require_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")
    return value.astimezone(UTC)


def _require_operational_run(run_time: datetime) -> datetime:
    run = _require_utc(run_time, "run_time")
    if run.hour % GEFS_CYCLE_HOURS or any((run.minute, run.second, run.microsecond)):
        raise ValueError("run_time must fall on a six-hour GEFS cycle")
    return run


def _require_product_and_lead(product: str, lead_hours: int) -> None:
    if product not in GEFS_PRODUCTS:
        raise ValueError(f"unsupported GEFS product: {product}")
    if lead_hours not in GEFS_LEADS:
        raise ValueError(f"unsupported GEFS lead: {lead_hours}")


def gefs_data_url(run_time: datetime, *, product: str, lead_hours: int) -> str:
    """Return an official NOAA public-S3 GEFS ensemble GRIB object URL."""
    run = _require_operational_run(run_time)
    _require_product_and_lead(product, lead_hours)
    directory, suffix = GEFS_PRODUCT_LAYOUTS[product]
    filename = f"{product}.t{run:%H}z.{suffix}.0p50.f{lead_hours:03d}"
    return f"{GEFS_AWS_ROOT}/gefs.{run:%Y%m%d}/{run:%H}/atmos/{directory}/{filename}"


def gefs_index_url(run_time: datetime, *, product: str, lead_hours: int) -> str:
    """Return the compact index URL paired with a GEFS GRIB object."""
    return f"{gefs_data_url(run_time, product=product, lead_hours=lead_hours)}.idx"


def validate_operational_gefs_timing(
    issue_time: datetime, run_time: datetime, valid_time: datetime
) -> None:
    """Require a forecast to have been issued and still valid after prediction time."""
    issue = _require_utc(issue_time, "issue_time")
    run = _require_operational_run(run_time)
    valid = _require_utc(valid_time, "valid_time")
    if run > issue:
        raise ValueError("GEFS forecast was issued after the prediction timestamp")
    if valid <= issue:
        raise ValueError("GEFS forecast valid time is not after the prediction timestamp")


def select_operational_gefs_run(issue_time: datetime) -> tuple[datetime, dict[int, datetime]]:
    """Select the latest GEFS cycle already issued at the prediction timestamp."""
    issue = _require_utc(issue_time, "issue_time")
    run = issue.replace(
        hour=(issue.hour // GEFS_CYCLE_HOURS) * GEFS_CYCLE_HOURS,
        minute=0,
        second=0,
        microsecond=0,
    )
    valid_times = {lead: run + timedelta(hours=lead) for lead in GEFS_LEADS}
    for valid in valid_times.values():
        validate_operational_gefs_timing(issue, run, valid)
    return run, valid_times


def download_operational_gefs_surface_messages(
    run_time: datetime,
    *,
    product: str,
    lead_hours: int,
    timeout_s: int = 60,
    max_workers: int = 4,
) -> dict[str, GEFSFieldMessage]:
    """Download five selected GEFS surface messages without accepting full GRIB files."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    run = _require_operational_run(run_time)
    _require_product_and_lead(product, lead_hours)
    index_url = gefs_index_url(run, product=product, lead_hours=lead_hours)
    index_response = requests.get(index_url, timeout=timeout_s)
    index_response.raise_for_status()
    entries = parse_gfs_index(index_response.content.decode("utf-8"))
    requested_names = _ENSEMBLE_FIELDS if product in {"geavg", "gespr"} else ("pblh",)
    selected = {}
    for feature_name in requested_names:
        variable, level = GEFS_FIELD_DEFINITIONS[feature_name]
        matches = [entry for entry in entries if entry.variable == variable and entry.level == level]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one GEFS field for {feature_name}, found {len(matches)}")
        selected[feature_name] = matches[0]
    source_url = gefs_data_url(run, product=product, lead_hours=lead_hours)
    valid_time = run + timedelta(hours=lead_hours)
    def download_one(feature_name: str, entry: object) -> tuple[str, GEFSFieldMessage]:
        byte_range = range_for_gfs_index_entry(entries, entry)
        expected_range = f"bytes {byte_range[0]}-{byte_range[1]}/"
        response = requests.get(
            source_url,
            headers={"Range": f"bytes={byte_range[0]}-{byte_range[1]}"},
            timeout=timeout_s,
        )
        if response.status_code != 206:
            raise RuntimeError(f"operational GEFS expected HTTP 206 for {feature_name}")
        if not response.headers.get("Content-Range", "").startswith(expected_range):
            raise RuntimeError(f"operational GEFS has unexpected Content-Range for {feature_name}")
        message = GEFSFieldMessage(
            product=product,
            feature_name=feature_name,
            run_time=run,
            lead_hours=lead_hours,
            valid_time=valid_time,
            source_url=source_url,
            index_url=index_url,
            byte_range=byte_range,
            content=response.content,
            sha256=hashlib.sha256(response.content).hexdigest(),
        )
        return feature_name, message

    messages: dict[str, GEFSFieldMessage] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(selected))) as executor:
        futures = {
            executor.submit(download_one, feature_name, entry): feature_name
            for feature_name, entry in selected.items()
        }
        for future in as_completed(futures):
            feature_name, message = future.result()
            messages[feature_name] = message
    return messages
