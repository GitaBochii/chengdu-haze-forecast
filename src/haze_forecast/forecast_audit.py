"""Audit historical forecast files before they can be used as model features.

This module intentionally records only evidence.  It never downloads forecast
grids or constructs features: a source without a verifiable publication time
and exact raw-file hash remains ineligible.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from haze_forecast.forecast_sources import gfs_aws_data_url, gfs_dods_url

CHENGDU_LATITUDE = 30.5723
CHENGDU_LONGITUDE = 104.0665
FORECAST_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "u_wind_10m",
    "v_wind_10m",
    "boundary_layer_height",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _utc_or_none(value: datetime | None, field_name: str, reasons: list[str]) -> datetime | None:
    if value is None:
        reasons.append(f"missing {field_name}")
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        reasons.append(f"{field_name} must be timezone-aware")
        return None
    return value.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _normalise_event_hours(
    event_hours: Sequence[datetime] | None, reasons: list[str]
) -> set[datetime] | None:
    if event_hours is None:
        return None
    normalised: set[datetime] = set()
    for event_hour in event_hours:
        parsed = _utc_or_none(event_hour, "event hour", reasons)
        if parsed is not None:
            normalised.add(parsed)
    if not normalised:
        reasons.append("event hours must not be empty")
    return normalised


def _valid_coordinates(latitude: float | None, longitude: float | None) -> bool:
    if isinstance(latitude, bool) or isinstance(longitude, bool):
        return False
    try:
        return -90 <= float(latitude) <= 90 and -180 <= float(longitude) <= 180
    except (TypeError, ValueError):
        return False


def audit_forecast_file(
    prediction_time: datetime,
    *,
    run_time: datetime | None,
    published_time: datetime | None,
    lead_hours: int | None,
    latitude: float | None,
    longitude: float | None,
    variables: Sequence[str] | None,
    download_url: str | None,
    sha256: str | None,
    event_hours: Sequence[datetime] | None = None,
    source_name: str | None = None,
    source_available: bool | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready eligibility record for one historical forecast file.

    A file is eligible only when it demonstrably existed at ``prediction_time``
    and its run, lead, target location, variables, URL, and exact bytes can be
    reviewed later.  ``event_hours`` makes alignment strict for a supplied set
    of Chengdu event timestamps; otherwise a whole UTC hour is required.
    """
    reasons: list[str] = []
    prediction_utc = _utc_or_none(prediction_time, "prediction_time", reasons)
    run_utc = _utc_or_none(run_time, "run_time", reasons)
    published_utc = _utc_or_none(published_time, "published_time", reasons)

    valid_time: datetime | None = None
    if isinstance(lead_hours, bool) or not isinstance(lead_hours, int):
        reasons.append("missing or invalid lead_hours")
    elif not 0 <= lead_hours <= 72:
        reasons.append("lead_hours must be between 0 and 72")
    elif run_utc is not None:
        valid_time = run_utc + timedelta(hours=lead_hours)

    if prediction_utc is not None and published_utc is not None and published_utc > prediction_utc:
        reasons.append("published_time is after prediction_time")
    if prediction_utc is not None and run_utc is not None and run_utc > prediction_utc:
        reasons.append("run_time is after prediction_time")
    if prediction_utc is not None and valid_time is not None and valid_time < prediction_utc:
        reasons.append("valid_time is before prediction_time")

    aligned_hours = _normalise_event_hours(event_hours, reasons)
    if valid_time is not None:
        if aligned_hours is not None and valid_time not in aligned_hours:
            reasons.append("valid_time does not align with a Chengdu event hour")
        elif aligned_hours is None and any(
            (valid_time.minute, valid_time.second, valid_time.microsecond)
        ):
            reasons.append("valid_time does not align with a whole Chengdu event hour")

    if not _valid_coordinates(latitude, longitude):
        reasons.append("invalid forecast latitude or longitude")
    normalised_variables = list(dict.fromkeys(variables or ()))
    if not normalised_variables or any(not isinstance(item, str) or not item.strip() for item in normalised_variables):
        reasons.append("missing forecast variables")
    if not isinstance(download_url, str) or urlparse(download_url).scheme not in {"http", "https"}:
        reasons.append("missing or invalid download_url")
    if not isinstance(sha256, str) or not _SHA256_PATTERN.fullmatch(sha256.lower()):
        reasons.append("missing or invalid sha256")
    if source_available is False:
        reasons.append("source probe is unavailable")

    return {
        "source_name": source_name,
        "prediction_time": _iso(prediction_utc),
        "run_time": _iso(run_utc),
        "published_time": _iso(published_utc),
        "lead_hours": lead_hours,
        "valid_time": _iso(valid_time),
        "latitude": float(latitude) if _valid_coordinates(latitude, longitude) else latitude,
        "longitude": float(longitude) if _valid_coordinates(latitude, longitude) else longitude,
        "variables": normalised_variables,
        "download_url": download_url,
        "sha256": sha256.lower() if isinstance(sha256, str) else sha256,
        "source_available": source_available,
        "evidence": dict(evidence or {}),
        "eligible": not reasons,
        "reason": "; ".join(reasons) if reasons else None,
        "failure_reasons": reasons,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_evidence(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"probe_manifest_path": str(path), "probe_manifest_error": "file not found"}
    try:
        probe = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"probe_manifest_path": str(path), "probe_manifest_error": str(error)}
    return {
        "probe_manifest_path": str(path),
        "probe_manifest_sha256": _file_sha256(path),
        "probe_available": probe.get("available"),
        "probe_url": probe.get("url"),
        "probe_status_code": probe.get("status_code"),
        "probe_error": probe.get("error"),
    }


def audit_local_gfs_sources(
    processed_dir: Path,
    *,
    run_time: datetime = datetime(2021, 3, 23, 18, tzinfo=UTC),
    lead_hours: int = 6,
) -> list[dict[str, Any]]:
    """Audit local GFS access evidence without downloading any forecast data.

    The existing probes demonstrate endpoint reachability only.  They do not
    supply a publisher timestamp or raw GRIB checksum, so the returned records
    correctly remain ineligible rather than creating pseudo-features.
    """
    if run_time.tzinfo is None or run_time.utcoffset() is None:
        raise ValueError("run_time must be timezone-aware")
    run_time = run_time.astimezone(UTC)
    prediction_time = run_time + timedelta(hours=lead_hours)
    shared = {
        "prediction_time": prediction_time,
        "run_time": run_time,
        "published_time": None,
        "lead_hours": lead_hours,
        "latitude": CHENGDU_LATITUDE,
        "longitude": CHENGDU_LONGITUDE,
        "variables": FORECAST_VARIABLES,
        "sha256": None,
        "event_hours": (prediction_time,),
    }
    aws_evidence = _probe_evidence(
        processed_dir / f"gfs_aws_probe_{run_time:%Y%m%d_%H%M}_f{lead_hours:03d}.json"
    )
    ncei_evidence = _probe_evidence(
        processed_dir / f"gfs_archive_probe_{run_time:%Y%m%d_%H%M}_f{lead_hours:03d}.json"
    )
    return [
        audit_forecast_file(
            **shared,
            source_name="NOAA public AWS GFS 0.25 degree",
            source_available=bool(aws_evidence.get("probe_available")),
            download_url=gfs_aws_data_url(run_time, lead_hours),
            evidence=aws_evidence,
        ),
        audit_forecast_file(
            **shared,
            source_name="NOAA NCEI THREDDS GFS Grid 3 archive",
            source_available=bool(ncei_evidence.get("probe_available")),
            download_url=gfs_dods_url(run_time, lead_hours),
            evidence=ncei_evidence,
        ),
    ]


def write_audit_report(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write a standalone JSON report; this audit never publishes features."""
    serialised_records = [dict(record) for record in records]
    eligible_count = sum(bool(record.get("eligible")) for record in serialised_records)
    payload = {
        "eligible": eligible_count > 0,
        "eligible_record_count": eligible_count,
        "record_count": len(serialised_records),
        "records": serialised_records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
