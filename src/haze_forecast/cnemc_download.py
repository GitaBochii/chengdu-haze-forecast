"""Resumable access to publicly mirrored CNEMC daily station observations."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import requests

CNEMC_DAILY_ROOT = "https://quotsoft.net/air/data"
DEFAULT_TIMEOUT_SECONDS = 60


def cnemc_daily_url(day: date) -> str:
    """Return the documented daily CNEMC station CSV URL for one calendar day."""
    return f"{CNEMC_DAILY_ROOT}/china_sites_{day:%Y%m%d}.csv"


def inclusive_dates(start: date, end: date) -> Iterator[date]:
    """Yield every calendar day in a closed interval."""
    if start > end:
        raise ValueError("start date must not be after end date")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def download_cnemc_daily_csv(
    day: date,
    destination: Path,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Path:
    """Download one daily CSV atomically, retaining an existing nonempty file."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "haze-forecast-research/1.0"}
    try:
        with requests.get(
            cnemc_daily_url(day), stream=True, timeout=timeout_seconds, headers=headers
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if destination.stat().st_size == 0:
        raise ValueError(f"CNEMC returned an empty daily file for {day.isoformat()}")
    return destination
