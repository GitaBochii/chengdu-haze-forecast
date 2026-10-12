"""Recovery bounds from a contiguous, actually observed future hourly prefix."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


def _hourly_index(values, name: str) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain timezone-aware timestamps") from exc
    if len(index) == 0 or index.hasnans or index.tz is None:
        raise ValueError(f"{name} must be nonempty, timezone-aware and nonmissing")
    if not index.is_unique:
        raise ValueError(f"{name} timestamps must be unique")
    if any(
        (index.minute != 0)
        | (index.second != 0)
        | (index.microsecond != 0)
        | (index.nanosecond != 0)
    ):
        raise ValueError(f"{name} timestamps must be hour-aligned")
    return index


def observed_recovery_bounds(frame: pd.DataFrame, origins, stable_hours: int = 6) -> pd.DataFrame:
    """Return exact or right-censored discrete-hour labels for supplied origins.

    Missing AQI, including absent hourly rows, terminates observation. Recovery
    starts strictly after the origin and requires ``stable_hours`` consecutive
    values <= 100. Censoring bounds are the first still-possible discrete hour,
    not an invented recovery time. ``observed_prefix_hours`` is the full known
    future prefix, even if recovery was confirmed earlier within that prefix.
    Input order is normalized; output follows the supplied origins' order.
    """
    if isinstance(stable_hours, bool) or not isinstance(stable_hours, Integral) or stable_hours < 1:
        raise ValueError("stable_hours must be a positive integer")
    if not {"timestamp", "aqi"}.issubset(frame.columns):
        raise ValueError("frame requires timestamp and aqi columns")
    timestamps = _hourly_index(frame["timestamp"], "frame")
    origin_index = _hourly_index(origins, "origins").tz_convert(timestamps.tz)
    try:
        aqi = pd.to_numeric(frame["aqi"], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError("aqi must be numeric or missing") from exc
    if np.isinf(aqi).any():
        raise ValueError("aqi must not contain infinity")
    grid = pd.date_range(timestamps.min(), timestamps.max(), freq="h")
    positions = grid.get_indexer(origin_index)
    if (positions < 0).any():
        raise ValueError("origins must lie on the frame's hourly grid")
    values = pd.Series(aqi, index=timestamps).reindex(grid).to_numpy()
    missing = np.flatnonzero(np.isnan(values))
    boundaries = np.append(missing, len(grid))

    # Each possible stable start is indexed once, so many origins need not scan
    # the full history repeatedly. A candidate beyond the first gap is rejected.
    clean = np.isfinite(values) & (values <= 100)
    cumulative = np.concatenate(([0], np.cumsum(clean, dtype=np.int64)))
    if len(values) >= stable_hours:
        starts = np.flatnonzero(
            cumulative[stable_hours:] - cumulative[:-stable_hours] == stable_hours
        )
    else:
        starts = np.array([], dtype=np.int64)
    rows = []
    for origin, position in zip(origin_index, positions, strict=True):
        boundary = int(boundaries[np.searchsorted(boundaries, position + 1)])
        prefix = boundary - position - 1
        candidate_index = np.searchsorted(starts, position + 1)
        candidate = int(starts[candidate_index]) if candidate_index < len(starts) else len(grid)
        exact = candidate + stable_hours <= boundary
        if exact:
            hours = float((grid[candidate].value - origin.value) / 3_600_000_000_000)
            lower = upper = hours
            available = grid[candidate + stable_hours - 1]
            informative = True
            reason = "exact"
        else:
            risk_bins = max(0, prefix - stable_hours + 1)
            lower, upper, hours = float(risk_bins + 1), np.inf, np.nan
            informative = risk_bins > 0
            reason = "gap" if boundary < len(grid) else "end_of_data"
            available = grid[boundary] if reason == "gap" else grid[max(position, boundary - 1)]
        rows.append(
            {
                "timestamp": origin,
                "lower_bound_hours": lower,
                "upper_bound_hours": upper,
                "exact_hours": hours,
                "observed_prefix_hours": prefix,
                "informative": informative,
                "label_available_at": available,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)
