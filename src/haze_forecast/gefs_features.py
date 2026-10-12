"""Feature-time alignment and ensemble summaries for GEFSv12 pilot data."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np

from haze_forecast.gefs_reforecast import (
    GEFS_REFORECAST_LAST_YEAR,
    GEFS_REFORECAST_MEMBERS,
)

GEFS_LEAD_INTERVAL_HOURS = 3
GEFS_MAX_LEAD_HOURS = 240


def select_reforecast_run_and_lead(
    issue_time: datetime, *, horizon_hours: int = 6
) -> tuple[datetime, int]:
    """Choose a daily run and a future valid time from the GEFS archive.

    The retrospective archive has one 00:00 UTC run per day.  The selected
    valid time is the first three-hourly forecast at or after the requested
    future horizon, never a field that is already valid at ``issue_time``.
    """
    if issue_time.tzinfo is None or issue_time.utcoffset() is None:
        raise ValueError("issue_time must be timezone-aware")
    if isinstance(horizon_hours, bool) or not isinstance(horizon_hours, int) or horizon_hours <= 0:
        raise ValueError("horizon_hours must be a positive integer")
    issue_utc = issue_time.astimezone(UTC)
    if not 2000 <= issue_utc.year <= GEFS_REFORECAST_LAST_YEAR:
        raise ValueError("issue_time is outside GEFSv12 retrospective coverage")
    run_time = issue_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    required_seconds = (issue_utc + timedelta(hours=horizon_hours) - run_time).total_seconds()
    lead_hours = int(np.ceil(required_seconds / 3600 / GEFS_LEAD_INTERVAL_HOURS)) * GEFS_LEAD_INTERVAL_HOURS
    if not 0 < lead_hours <= GEFS_MAX_LEAD_HOURS:
        raise ValueError("selected GEFSv12 lead is outside the Days:1-10 archive")
    return run_time, lead_hours


def summarise_ensemble(values: Sequence[float]) -> dict[str, float]:
    """Return deterministic distribution features for the complete five-member ensemble."""
    if len(values) != len(GEFS_REFORECAST_MEMBERS):
        raise ValueError("GEFSv12 ensemble summary requires exactly five members")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("GEFSv12 ensemble values must be finite")
    return {
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=0)),
        "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
    }
