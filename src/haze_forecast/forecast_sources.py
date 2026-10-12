"""No-leak access planning for archived numerical weather forecasts.

The model must only see a forecast cycle that had been issued by the event time.
This module deliberately separates archive access from feature construction so a
failed remote probe cannot silently substitute analysed (future-informed) weather.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import requests

GFS_ARCHIVE_CATALOG_ROOT = "https://www.ncei.noaa.gov/thredds/catalog/model-gfs-004-files"
GFS_ARCHIVE_DODS_ROOT = "https://www.ncei.noaa.gov/thredds/dodsC/gfs-g4-files"
GFS_AWS_ROOT = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GFS_CYCLE_HOURS = 6
GFS_FORECAST_INTERVAL_HOURS = 3
GFS_MAX_LEAD_HOURS = 384
GFS_AWS_ATMOS_LAYOUT_START = datetime(2021, 3, 23, tzinfo=UTC)


@dataclass(frozen=True)
class GFSForecastRequest:
    """A historical GFS request whose selected run was available at issue time."""

    issued_at: datetime
    run_time: datetime
    latitude: float
    longitude: float
    lead_hours: tuple[int, ...]

    @property
    def valid_times(self) -> tuple[datetime, ...]:
        return tuple(self.run_time + timedelta(hours=lead) for lead in self.lead_hours)


@dataclass(frozen=True)
class GFSProbeResult:
    """Result of a small OPeNDAP metadata request, never a global GRIB download."""

    requested_at_utc: str
    url: str
    status_code: int | None
    response_bytes: int
    error: str | None

    @property
    def available(self) -> bool:
        return self.status_code == 200 and self.error is None and self.response_bytes > 0


@dataclass(frozen=True)
class GFSIndexEntry:
    """One GRIB message location advertised by NOAA's compact ``.idx`` file."""

    record_number: int
    byte_offset: int
    variable: str
    level: str
    description: str


@dataclass(frozen=True)
class GFSFieldMessage:
    """A single downloaded GRIB message, obtained with an HTTP Range request."""

    feature_name: str
    source_url: str
    byte_range: tuple[int, int]
    content: bytes


def _require_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")
    return value.astimezone(UTC)


def _cycle_at_or_before(issued_at: datetime) -> datetime:
    return issued_at.replace(
        hour=(issued_at.hour // GFS_CYCLE_HOURS) * GFS_CYCLE_HOURS,
        minute=0,
        second=0,
        microsecond=0,
    )


def _validate_location(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be in [-90, 90]")
    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be in [-180, 180]")


def _validate_lead_hours(max_lead_hours: int) -> None:
    if not GFS_FORECAST_INTERVAL_HOURS <= max_lead_hours <= GFS_MAX_LEAD_HOURS:
        raise ValueError(
            f"max_lead_hours must be in [{GFS_FORECAST_INTERVAL_HOURS}, {GFS_MAX_LEAD_HOURS}]"
        )
    if max_lead_hours % GFS_FORECAST_INTERVAL_HOURS:
        raise ValueError(f"max_lead_hours must be divisible by {GFS_FORECAST_INTERVAL_HOURS}")


def gfs_catalog_url(run_time: datetime) -> str:
    """Return the NCEI catalog URL that lists all cycles for one UTC date."""
    run_time = _require_utc(run_time, "run_time")
    return f"{GFS_ARCHIVE_CATALOG_ROOT}/{run_time:%Y%m}/{run_time:%Y%m%d}/catalog.xml"


def gfs_dods_url(run_time: datetime, lead_hours: int) -> str:
    """Return an OPeNDAP metadata/data endpoint for one archived GFS Grid 3 file."""
    run_time = _require_utc(run_time, "run_time")
    _validate_lead_hours(lead_hours)
    if run_time.hour % GFS_CYCLE_HOURS:
        raise ValueError("run_time must fall on a six-hour GFS cycle")
    filename = f"gfs_3_{run_time:%Y%m%d}_{run_time:%H}00_{lead_hours:03d}.grb2"
    return f"{GFS_ARCHIVE_DODS_ROOT}/{run_time:%Y%m}/{run_time:%Y%m%d}/{filename}"


def _gfs_aws_object_key(run_time: datetime, lead_hours: int) -> str:
    run_time = _require_utc(run_time, "run_time")
    _validate_lead_hours(lead_hours)
    if run_time.hour % GFS_CYCLE_HOURS:
        raise ValueError("run_time must fall on a six-hour GFS cycle")
    filename = f"gfs.t{run_time:%H}z.pgrb2.0p25.f{lead_hours:03d}"
    prefix = f"gfs.{run_time:%Y%m%d}/{run_time:%H}"
    if run_time >= GFS_AWS_ATMOS_LAYOUT_START:
        return f"{prefix}/atmos/{filename}"
    return f"{prefix}/{filename}"


def gfs_aws_data_url(run_time: datetime, lead_hours: int) -> str:
    """Return an official NOAA public-S3 GFS GRIB object URL."""
    return f"{GFS_AWS_ROOT}/{_gfs_aws_object_key(run_time, lead_hours)}"


def gfs_aws_index_url(run_time: datetime, lead_hours: int) -> str:
    """Return the small sidecar index for a public-S3 GFS GRIB object."""
    return f"{gfs_aws_data_url(run_time, lead_hours)}.idx"


def parse_gfs_index(index_text: str) -> tuple[GFSIndexEntry, ...]:
    """Parse NOAA's line-oriented GRIB index without opening a global data file."""
    entries: list[GFSIndexEntry] = []
    for line_number, raw_line in enumerate(index_text.splitlines(), start=1):
        if not raw_line:
            continue
        parts = raw_line.split(":")
        if len(parts) < 5:
            raise ValueError(f"invalid GFS index line {line_number}: {raw_line!r}")
        try:
            record_number = int(parts[0])
            byte_offset = int(parts[1])
        except ValueError as error:
            raise ValueError(f"invalid GFS index line {line_number}: {raw_line!r}") from error
        entries.append(
            GFSIndexEntry(
                record_number=record_number,
                byte_offset=byte_offset,
                variable=parts[3],
                level=parts[4],
                description=":".join(parts[5:]).rstrip(":"),
            )
        )
    if not entries:
        raise ValueError("GFS index has no entries")
    if any(current.byte_offset <= previous.byte_offset for previous, current in pairwise(entries)):
        raise ValueError("GFS index offsets must be strictly increasing")
    return tuple(entries)


def select_gfs_surface_fields(entries: tuple[GFSIndexEntry, ...]) -> dict[str, GFSIndexEntry]:
    """Select unambiguous fields that are useful for Chengdu haze transport."""
    requested = {
        "temperature_2m": ("TMP", "2 m above ground"),
        "relative_humidity_2m": ("RH", "2 m above ground"),
        "u_wind_10m": ("UGRD", "10 m above ground"),
        "v_wind_10m": ("VGRD", "10 m above ground"),
        "pblh": ("HPBL", "surface"),
    }
    selected: dict[str, GFSIndexEntry] = {}
    for feature_name, (variable, level) in requested.items():
        matches = [entry for entry in entries if entry.variable == variable and entry.level == level]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one GFS field for {feature_name}, found {len(matches)}"
            )
        selected[feature_name] = matches[0]
    return selected


def range_for_gfs_index_entry(
    entries: tuple[GFSIndexEntry, ...], entry: GFSIndexEntry
) -> tuple[int, int]:
    """Return an inclusive HTTP Range for a non-final GRIB message."""
    for index, candidate in enumerate(entries[:-1]):
        if candidate == entry:
            return entry.byte_offset, entries[index + 1].byte_offset - 1
    if entry == entries[-1]:
        raise ValueError("final GFS index entry requires the full object size")
    raise ValueError("entry does not belong to the supplied GFS index")


def download_gfs_surface_messages(
    run_time: datetime,
    lead_hours: int,
    field_names: tuple[str, ...] | None = None,
    timeout_s: int = 60,
    max_workers: int = 1,
    max_attempts: int = 3,
    retry_delay_s: float = 1.0,
) -> dict[str, GFSFieldMessage]:
    """Download selected GFS messages without ever reading the global GRIB object.

    A successful response must be ``206 Partial Content`` and match the exact
    requested range.  This defensive check prevents a misconfigured endpoint
    from accidentally downloading a 300-500 MB global forecast file.
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if retry_delay_s < 0:
        raise ValueError("retry_delay_s must not be negative")
    index_response = requests.get(gfs_aws_index_url(run_time, lead_hours), timeout=timeout_s)
    index_response.raise_for_status()
    entries = parse_gfs_index(index_response.content.decode("utf-8"))
    selected = select_gfs_surface_fields(entries)
    requested_names = tuple(selected) if field_names is None else field_names
    unknown_names = set(requested_names) - set(selected)
    if unknown_names:
        raise ValueError(f"unsupported GFS field names: {sorted(unknown_names)}")

    source_url = gfs_aws_data_url(run_time, lead_hours)
    def download_one(feature_name: str) -> GFSFieldMessage:
        byte_range = range_for_gfs_index_entry(entries, selected[feature_name])
        expected_content_range = f"bytes {byte_range[0]}-{byte_range[1]}/"
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    source_url,
                    headers={"Range": f"bytes={byte_range[0]}-{byte_range[1]}"},
                    stream=True,
                    timeout=timeout_s,
                )
                if response.status_code != 206:
                    raise RuntimeError(
                        f"expected HTTP 206 for {feature_name}, got {response.status_code}; "
                        "global download refused"
                    )
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(expected_content_range):
                    raise RuntimeError(f"unexpected Content-Range for {feature_name}: {content_range!r}")
                content = response.content
                break
            except requests.RequestException:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(retry_delay_s * (attempt + 1))
        return GFSFieldMessage(
            feature_name=feature_name,
            source_url=source_url,
            byte_range=byte_range,
            content=content,
        )

    messages: dict[str, GFSFieldMessage] = {}
    if max_workers == 1:
        for feature_name in requested_names:
            messages[feature_name] = download_one(feature_name)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_one, feature_name): feature_name for feature_name in requested_names}
            for future in as_completed(futures):
                feature_name = futures[future]
                messages[feature_name] = future.result()
    return messages


def build_operational_gfs_request(
    issued_at: datetime,
    latitude: float,
    longitude: float,
    max_lead_hours: int = 24,
) -> GFSForecastRequest:
    """Select the latest archived GFS cycle that existed when an event began.

    Grid 3 archives expose three-hour forecast steps, so an event at 14:00 UTC
    uses the 12:00 cycle and leads +3 to +24.  The cycle never moves forward to
    a later forecast even when its valid time would align better with an event.
    """
    issued_at = _require_utc(issued_at, "issued_at")
    _validate_location(latitude, longitude)
    _validate_lead_hours(max_lead_hours)
    run_time = _cycle_at_or_before(issued_at)
    leads = tuple(range(GFS_FORECAST_INTERVAL_HOURS, max_lead_hours + 1, GFS_FORECAST_INTERVAL_HOURS))
    return GFSForecastRequest(
        issued_at=issued_at,
        run_time=run_time,
        latitude=latitude,
        longitude=longitude,
        lead_hours=leads,
    )


def probe_gfs_opendap(run_time: datetime, lead_hours: int, timeout_s: int = 30) -> GFSProbeResult:
    """Read a tiny OPeNDAP schema document to prove archive access is usable."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    url = f"{gfs_dods_url(run_time, lead_hours)}.dds"
    requested_at = datetime.now(UTC).isoformat()
    try:
        response = requests.get(
            url,
            timeout=timeout_s,
            headers={"User-Agent": "ChengduHazeForecast/0.1"},
        )
        return GFSProbeResult(
            requested_at_utc=requested_at,
            url=url,
            status_code=response.status_code,
            response_bytes=len(response.content),
            error=None,
        )
    except requests.RequestException as error:
        return GFSProbeResult(
            requested_at_utc=requested_at,
            url=url,
            status_code=None,
            response_bytes=0,
            error=f"{type(error).__name__}: {error}",
        )


def probe_gfs_aws(run_time: datetime, lead_hours: int, timeout_s: int = 30) -> GFSProbeResult:
    """Probe NOAA's public-S3 compact index before fetching any GRIB message."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    url = gfs_aws_index_url(run_time, lead_hours)
    requested_at = datetime.now(UTC).isoformat()
    try:
        response = requests.get(url, timeout=timeout_s)
        return GFSProbeResult(
            requested_at_utc=requested_at,
            url=url,
            status_code=response.status_code,
            response_bytes=len(response.content),
            error=None,
        )
    except requests.RequestException as error:
        return GFSProbeResult(
            requested_at_utc=requested_at,
            url=url,
            status_code=None,
            response_bytes=0,
            error=f"{type(error).__name__}: {error}",
        )


def write_probe_manifest(path: Path, result: GFSProbeResult) -> None:
    """Persist probe evidence so later feature generation cannot hide source failures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(result), "available": result.available}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
