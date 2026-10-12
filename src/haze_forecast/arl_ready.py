"""Client helpers for NOAA ARL READY's archived GFS subset service."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from zipfile import ZipFile

import requests

READY_ROOT = "https://www.ready.noaa.gov"
READY_GFS_DATASET = "GFS0P25"
READY_GFS_DIRECTORY = "/pub/archives/gfs0p25/"
DEFAULT_DOMAIN = (22.0, 95.0, 40.0, 112.0)


def ready_gfs_filename(day: date) -> str:
    """Return the READY daily archive name that contains the requested UTC day."""
    return f"{day:%Y%m%d}_gfs0p25"


def parse_ready_job_id(html: str) -> str:
    """Read the transient READY extraction job id from a form response."""
    match = re.search(r'name="proc" value="(\d+)"', html)
    if match is None:
        raise ValueError("READY did not return an extraction job id")
    return match.group(1)


def parse_ready_download_path(html: str, job_id: str) -> str | None:
    """Return the completed ZIP path, or None while the job is still running."""
    match = re.search(r'href="([^"]*extract_' + re.escape(job_id) + r'\.zip)"', html)
    return None if match is None else match.group(1)


def _request_payload(job_id: str, filename: str) -> dict[str, str]:
    return {
        "metdata": READY_GFS_DATASET,
        "mdatacfg": READY_GFS_DATASET,
        "metdir1": READY_GFS_DIRECTORY,
        "metfil1": filename,
        "xtype": "3",
        "proc": job_id,
    }


def download_ready_gfs_day(
    day: date,
    output_dir: Path,
    *,
    domain: tuple[float, float, float, float] = DEFAULT_DOMAIN,
    start_hour: int = 0,
    end_hour: int = 21,
    timeout_s: int = 120,
    poll_interval_s: float = 10.0,
    max_wait_s: float = 600.0,
    session: requests.Session | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> Path:
    """Request a full UTC day subset and return its extracted ARL binary file.

    READY creates a temporary job. This client deliberately submits one job at a
    time, avoiding unbounded load on the public archive service.
    """
    if timeout_s <= 0 or poll_interval_s <= 0 or max_wait_s <= 0:
        raise ValueError("READY timeouts and polling interval must be positive")
    if (
        start_hour not in range(0, 22, 3)
        or end_hour not in range(0, 22, 3)
        or start_hour > end_hour
    ):
        raise ValueError("READY extraction hours must be ordered three-hour UTC cycles")
    lat_lower, lon_lower, lat_upper, lon_upper = domain
    if not lat_lower < lat_upper or not lon_lower < lon_upper:
        raise ValueError("READY extraction domain corners are invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = ready_gfs_filename(day)
    time_suffix = "" if (start_hour, end_hour) == (0, 21) else f"_{start_hour:02d}_{end_hour:02d}"
    expected = output_dir / f"{filename}{time_suffix}.bin"
    if expected.exists():
        return expected
    client = session or requests.Session()
    initial = client.post(
        f"{READY_ROOT}/ready2-bin/extract/extract1a.pl",
        data={
            "xtype": "3",
            "metdata": READY_GFS_DATASET,
            "metdatasm": "gfs0p25",
            "metfile": filename,
        },
        timeout=timeout_s,
    )
    initial.raise_for_status()
    job_id = parse_ready_job_id(initial.text)
    selection = _request_payload(job_id, filename) | {
        "Start year": f"{day:%y}",
        "Start month": f"{day:%m}",
        "Start day": f"{day:%d}",
        "Start hour": f"{start_hour:02d}",
        "End year": f"{day:%y}",
        "End month": f"{day:%m}",
        "End day": f"{day:%d}",
        "End hour": f"{end_hour:02d}",
        "latL": str(lat_lower),
        "lonL": str(lon_lower),
        "latR": str(lat_upper),
        "lonR": str(lon_upper),
    }
    review = client.get(
        f"{READY_ROOT}/ready2-bin/extract/extract2a.pl", params=selection, timeout=timeout_s
    )
    review.raise_for_status()
    submit = _request_payload(job_id, filename) | {
        "syr": f"{day:%y}",
        "smn": f"{day:%m}",
        "sdy": f"{day:%d}",
        "shr": f"{start_hour:02d}",
        "eyr": f"{day:%y}",
        "emn": f"{day:%m}",
        "edy": f"{day:%d}",
        "ehr": f"{end_hour:02d}",
        "latL": str(lat_lower),
        "lonL": str(lon_lower),
        "latR": str(lat_upper),
        "lonR": str(lon_upper),
    }
    response = client.post(
        f"{READY_ROOT}/ready2-bin/extract/extract3a.pl", data=submit, timeout=timeout_s
    )
    response.raise_for_status()
    deadline = time.monotonic() + max_wait_s
    download_path: str | None = None
    while time.monotonic() < deadline:
        status = client.get(
            f"{READY_ROOT}/ready2-bin/extract/results.pl", params={"proc": job_id}, timeout=timeout_s
        )
        status.raise_for_status()
        download_path = parse_ready_download_path(status.text, job_id)
        if download_path is not None:
            break
        sleeper(poll_interval_s)
    if download_path is None:
        raise TimeoutError(f"READY extraction job {job_id} did not finish in time")
    archive = client.get(f"{READY_ROOT}{download_path}", timeout=timeout_s)
    archive.raise_for_status()
    zip_path = output_dir / f"{filename}{time_suffix}.zip"
    zip_path.write_bytes(archive.content)
    with ZipFile(zip_path) as bundle:
        binary_members = [member for member in bundle.namelist() if member.lower().endswith(".bin")]
        if len(binary_members) != 1:
            raise ValueError("READY ZIP must contain exactly one ARL binary subset")
        with bundle.open(binary_members[0]) as source, expected.open("wb") as destination:
            destination.write(source.read())
    return expected


def arl_f003_request_times(timestamps) -> tuple:
    """Keep hourly rows where the latest available six-hour cycle has a +3h forecast."""
    import pandas as pd

    values = pd.to_datetime(timestamps, utc=True)
    runs = values.dt.floor("6h") if isinstance(values, pd.Series) else values.floor("6h")
    valid_times = runs + pd.offsets.Hour(3)
    return values < valid_times, runs, valid_times


def ensure_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("forecast time must include timezone information")
    return value.astimezone(UTC)
