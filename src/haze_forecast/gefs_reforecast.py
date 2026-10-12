"""Bounded access helpers for NOAA's GEFSv12 retrospective archive.

The archive is a retrospective hindcast product.  These helpers deliberately
keep that provenance explicit and only address individual GRIB messages via
their public ``.idx`` sidecars; they do not create training features.
"""

from __future__ import annotations

import hashlib
import re
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock

import requests

from haze_forecast.forecast_sources import (
    GFSIndexEntry,
    parse_gfs_index,
    range_for_gfs_index_entry,
)

GEFS_REFORECAST_ROOT = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
GEFS_REFORECAST_FIRST_YEAR = 2000
GEFS_REFORECAST_LAST_YEAR = 2019
GEFS_REFORECAST_MEMBERS = ("c00", "p01", "p02", "p03", "p04")
_LEAD_PATTERN = re.compile(r"(?:^|:)\s*(\d+) hour fcst(?:$|:)")
_MAX_REQUEST_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 0.5
_RETRYABLE_REQUEST_ERRORS = (
    requests.exceptions.SSLError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


@dataclass(frozen=True)
class GEFSReforecastMessage:
    """One downloaded retrospective GRIB message with immutable byte evidence."""

    run_time: datetime
    member: str
    variable: str
    lead_hours: int
    source_url: str
    index_url: str
    byte_range: tuple[int, int]
    content: bytes
    sha256: str


@dataclass(frozen=True)
class GEFSSurfaceVariable:
    """A GEFSv12 surface message and its unambiguous GRIB-level selection."""

    level_hint: str | None
    feature_prefix: str
    unit: str


def _get_with_transient_retry(
    url: str, *, timeout_s: int, headers: dict[str, str] | None = None
) -> requests.Response:
    """Fetch one GEFS object with bounded retry only for transient transport failures."""
    request_kwargs: dict[str, object] = {"timeout": timeout_s}
    if headers is not None:
        request_kwargs["headers"] = headers
    for attempt in range(_MAX_REQUEST_ATTEMPTS):
        try:
            response = requests.get(url, **request_kwargs)
        except _RETRYABLE_REQUEST_ERRORS:
            if attempt == _MAX_REQUEST_ATTEMPTS - 1:
                raise
        else:
            status_code = getattr(response, "status_code", None)
            if status_code is None or not 500 <= status_code <= 599:
                response.raise_for_status()
                return response
            if attempt == _MAX_REQUEST_ATTEMPTS - 1:
                response.raise_for_status()
                raise RuntimeError("GEFSv12 request returned a retryable HTTP status without an error")
        time.sleep(_RETRY_BACKOFF_SECONDS * (2**attempt))
    raise AssertionError("GEFSv12 retry loop exited unexpectedly")


@dataclass
class GEFSReforecastIndexCache:
    """Thread-safe parsed-index cache keyed by one immutable GEFS GRIB object."""

    _entries: dict[tuple[datetime, str, str], tuple[GFSIndexEntry, ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _in_flight: dict[tuple[datetime, str, str], Future[tuple[GFSIndexEntry, ...]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def get_entries(
        self,
        run_time: datetime,
        *,
        member: str,
        variable: str,
        timeout_s: int,
    ) -> tuple[GFSIndexEntry, ...]:
        """Fetch and parse an index once, including when callers share threads."""
        key = (run_time, member, variable)
        with self._lock:
            cached_entries = self._entries.get(key)
            if cached_entries is not None:
                return cached_entries
            pending = self._in_flight.get(key)
            if pending is None:
                pending = Future()
                self._in_flight[key] = pending
                should_fetch = True
            else:
                should_fetch = False

        if not should_fetch:
            return pending.result()

        try:
            index_url = gefs_reforecast_index_url(run_time, member=member, variable=variable)
            index_response = _get_with_transient_retry(index_url, timeout_s=timeout_s)
            entries = parse_gfs_index(index_response.content.decode("utf-8"))
        except BaseException as error:
            pending.set_exception(error)
            with self._lock:
                if self._in_flight.get(key) is pending:
                    del self._in_flight[key]
            raise

        with self._lock:
            self._entries[key] = entries
        pending.set_result(entries)
        with self._lock:
            if self._in_flight.get(key) is pending:
                del self._in_flight[key]
        return entries


# Wind GRIB files contain both 10 m and 100 m fields, so their height is part
# of the source contract rather than an optional caller preference.
GEFS_SURFACE_VARIABLES = {
    "tmp_2m": GEFSSurfaceVariable(None, "gefs_tmp_2m", "k"),
    "spfh_2m": GEFSSurfaceVariable(None, "gefs_spfh_2m", "kg_kg"),
    "ugrd_hgt": GEFSSurfaceVariable("10 m above ground", "gefs_ugrd_10m", "m_s"),
    "vgrd_hgt": GEFSSurfaceVariable("10 m above ground", "gefs_vgrd_10m", "m_s"),
    "pres_sfc": GEFSSurfaceVariable(None, "gefs_pres_sfc", "pa"),
    "pbl_hgt": GEFSSurfaceVariable(None, "gefs_pblh", "m"),
}
GEFS_REFORECAST_VARIABLES = tuple(GEFS_SURFACE_VARIABLES)


def _validate_request(run_time: datetime, member: str, variable: str) -> datetime:
    if run_time.tzinfo is None or run_time.utcoffset() is None:
        raise ValueError("run_time must be timezone-aware UTC")
    run_time = run_time.astimezone(UTC)
    if not GEFS_REFORECAST_FIRST_YEAR <= run_time.year <= GEFS_REFORECAST_LAST_YEAR:
        raise ValueError("run_time is outside GEFSv12 retrospective coverage")
    if any((run_time.hour, run_time.minute, run_time.second, run_time.microsecond)):
        raise ValueError("GEFSv12 retrospective runs must start at 00:00 UTC")
    if member not in GEFS_REFORECAST_MEMBERS:
        raise ValueError(f"unsupported GEFSv12 member: {member}")
    if variable not in GEFS_REFORECAST_VARIABLES:
        raise ValueError(f"unsupported GEFSv12 variable: {variable}")
    return run_time


def gefs_reforecast_data_url(run_time: datetime, *, member: str, variable: str) -> str:
    """Return one public GRIB object URL in the fixed GEFSv12 archive layout."""
    run_time = _validate_request(run_time, member, variable)
    filename = f"{variable}_{run_time:%Y%m%d%H}_{member}.grib2"
    return (
        f"{GEFS_REFORECAST_ROOT}/{run_time:%Y}/{run_time:%Y%m%d%H}/{member}/Days:1-10/"
        f"{filename}"
    )


def gefs_reforecast_index_url(run_time: datetime, *, member: str, variable: str) -> str:
    """Return the compact index URL belonging to one GEFSv12 GRIB object."""
    return f"{gefs_reforecast_data_url(run_time, member=member, variable=variable)}.idx"


def select_reforecast_lead(
    entries: tuple[GFSIndexEntry, ...], *, lead_hours: int, level_hint: str | None = None
) -> GFSIndexEntry:
    """Select one GEFS forecast message by lead and optional vertical-level text."""
    if isinstance(lead_hours, bool) or not isinstance(lead_hours, int) or lead_hours <= 0:
        raise ValueError("lead_hours must be a positive integer")
    matches = []
    for entry in entries:
        match = _LEAD_PATTERN.search(entry.description)
        matches_level = level_hint is None or level_hint == entry.level
        if match is not None and int(match.group(1)) == lead_hours and matches_level:
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one GEFSv12 index entry for {lead_hours} hour lead, found {len(matches)}"
        )
    return matches[0]


def range_for_reforecast_lead(
    entries: tuple[GFSIndexEntry, ...], entry: GFSIndexEntry
) -> tuple[int, int]:
    """Return the inclusive byte range for a non-final selected message."""
    return range_for_gfs_index_entry(entries, entry)


def download_gefs_reforecast_message(
    run_time: datetime,
    *,
    member: str,
    variable: str,
    lead_hours: int,
    level_hint: str | None = None,
    timeout_s: int = 60,
    index_cache: GEFSReforecastIndexCache | None = None,
) -> GEFSReforecastMessage:
    """Read one indexed GEFS message without fetching its whole global GRIB file."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    run_time = _validate_request(run_time, member, variable)
    index_url = gefs_reforecast_index_url(run_time, member=member, variable=variable)
    if index_cache is None:
        index_response = _get_with_transient_retry(index_url, timeout_s=timeout_s)
        entries = parse_gfs_index(index_response.content.decode("utf-8"))
    else:
        entries = index_cache.get_entries(
            run_time, member=member, variable=variable, timeout_s=timeout_s
        )
    entry = select_reforecast_lead(entries, lead_hours=lead_hours, level_hint=level_hint)
    byte_range = range_for_reforecast_lead(entries, entry)
    source_url = gefs_reforecast_data_url(run_time, member=member, variable=variable)
    range_response = _get_with_transient_retry(
        source_url,
        headers={"Range": f"bytes={byte_range[0]}-{byte_range[1]}"},
        timeout_s=timeout_s,
    )
    expected_content_range = f"bytes {byte_range[0]}-{byte_range[1]}/"
    if range_response.status_code != 206:
        raise RuntimeError(
            f"expected HTTP 206 for GEFSv12 retrospective message, got {range_response.status_code}"
        )
    if not range_response.headers.get("Content-Range", "").startswith(expected_content_range):
        raise RuntimeError("GEFSv12 retrospective response does not match requested byte range")
    content = range_response.content
    return GEFSReforecastMessage(
        run_time=run_time,
        member=member,
        variable=variable,
        lead_hours=lead_hours,
        source_url=source_url,
        index_url=index_url,
        byte_range=byte_range,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
    )
