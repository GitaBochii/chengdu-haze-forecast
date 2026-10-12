"""Plan, merge and validate NOAA ARL +3 hour feature batches for held-out-event evaluation.

The helpers here are deliberately pure so the extraction scripts stay thin and the
strict no-leak rules can be unit tested without touching the network.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime

import pandas as pd

from haze_forecast.arl_ready import arl_f003_request_times

ARL_RUN_COLUMN = "arl_forecast_run_utc"
ARL_VALID_COLUMN = "arl_forecast_valid_utc"
ARL_ARCHIVE_DAY_COLUMN = "arl_archive_day"
ARL_SOURCE_COLUMN = "arl_source_batch"
ARL_FEATURE_COLUMNS: tuple[str, ...] = (
    "arl_temperature_c_f003",
    "arl_relative_humidity_pct_f003",
    "arl_wind_u_ms_f003",
    "arl_wind_v_ms_f003",
    "arl_boundary_layer_height_m_f003",
)
ARL_META_COLUMNS: tuple[str, ...] = (
    ARL_RUN_COLUMN,
    ARL_VALID_COLUMN,
    "arl_grid_longitude",
    "arl_grid_latitude",
)
ROW_IDENTITY_COLUMNS: tuple[str, ...] = ("timestamp", "event_id")


def _coerce_day(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _coerce_timestamp(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value, tz=UTC)


def candidate_arl_rows(
    frame: pd.DataFrame,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
) -> pd.DataFrame:
    """Filter labeled haze events and attach the strict no-leak +3 hour cycle columns."""
    required = {"timestamp", "event_id", "remaining_hours", "aqi"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"candidate frame is missing columns: {sorted(missing)}")
    lower = _coerce_timestamp(start)
    upper = _coerce_timestamp(end)
    if lower >= upper:
        raise ValueError("candidate window start must be earlier than its end")

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    mask = (
        result["timestamp"].ge(lower)
        & result["timestamp"].lt(upper)
        & result["event_id"].gt(0)
        & result["remaining_hours"].notna()
        & result["aqi"].gt(100)
    )
    result = result.loc[mask].copy()
    keep, runs, valid_times = arl_f003_request_times(result["timestamp"])
    result = result.loc[keep].copy()
    result[ARL_RUN_COLUMN] = runs[keep].to_numpy()
    result[ARL_VALID_COLUMN] = valid_times[keep].to_numpy()
    result[ARL_ARCHIVE_DAY_COLUMN] = result[ARL_VALID_COLUMN].dt.date
    return result.reset_index(drop=True)


def resolve_archive_days(
    all_days: Iterable[date],
    *,
    days: Sequence[date | str] | None = None,
    day_offset: int = 0,
    max_days: int = 1,
) -> list[date]:
    """Return explicit archive days, or one bounded slice of the sorted candidate days."""
    ordered = sorted({_coerce_day(day) for day in all_days})
    if days is not None:
        requested = {_coerce_day(day) for day in days}
        unknown = sorted(day.isoformat() for day in requested - set(ordered))
        if unknown:
            raise ValueError(f"archive days without candidate rows: {unknown}")
        return [day for day in ordered if day in requested]
    if max_days <= 0:
        raise ValueError("--max-days must be positive")
    if day_offset < 0:
        raise ValueError("--day-offset cannot be negative")
    selected = ordered[day_offset : day_offset + max_days]
    if not selected:
        raise ValueError("the requested --day-offset and --max-days match no candidate day")
    return selected


def merge_arl_feature_batches(batches: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate same-schema ARL batches, drop repeated rows and keep provenance.

    Batches are merged on the original row identity instead of the row index so a
    rerun of an already covered archive day cannot silently double-count an event.
    """
    if not batches:
        raise ValueError("at least one ARL batch is required")
    required_columns = [
        *ROW_IDENTITY_COLUMNS,
        *ARL_META_COLUMNS,
        *ARL_FEATURE_COLUMNS,
    ]
    frames: list[pd.DataFrame] = []
    for name, frame in batches.items():
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"ARL batch {name} is missing columns: {missing}")
        part = frame.copy()
        part[ARL_SOURCE_COLUMN] = str(name)
        frames.append(part)
    schemas = {tuple(frame.columns) for frame in frames}
    if len(schemas) > 1:
        raise ValueError("ARL batches must share one column schema before merging")

    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    identity = list(ROW_IDENTITY_COLUMNS)
    repeated = merged.duplicated(subset=identity, keep=False)
    if repeated.any():
        conflicting = (
            merged.loc[repeated]
            .groupby(identity, dropna=False)[list(ARL_FEATURE_COLUMNS)]
            .nunique()
            .gt(1)
            .any(axis=1)
        )
        if conflicting.any():
            raise ValueError(f"repeated ARL rows disagree on feature values: {int(conflicting.sum())}")
    merged = merged.drop_duplicates(subset=identity, keep="first").reset_index(drop=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    return merged.sort_values("timestamp").reset_index(drop=True)


def validate_arl_feature_frame(frame: pd.DataFrame) -> dict[str, object]:
    """Raise when a merged ARL frame breaks the no-leak or completeness rules."""
    if frame.empty:
        raise ValueError("merged ARL frame is empty")
    required = [*ROW_IDENTITY_COLUMNS, ARL_RUN_COLUMN, ARL_VALID_COLUMN, *ARL_FEATURE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"merged ARL frame is missing columns: {missing}")

    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    run = pd.to_datetime(frame[ARL_RUN_COLUMN], utc=True)
    valid = pd.to_datetime(frame[ARL_VALID_COLUMN], utc=True)
    if not run.le(timestamp).all():
        raise ValueError("ARL features use a forecast cycle issued after the event timestamp")
    if not valid.gt(timestamp).all():
        raise ValueError("ARL features use a valid time that is not in the future of the row")
    if not valid.sub(run).dt.total_seconds().eq(3 * 3600).all():
        raise ValueError("ARL features must all come from the +3 hour forecast field")
    missing_values = int(frame[list(ARL_FEATURE_COLUMNS)].isna().sum().sum())
    if missing_values:
        raise ValueError(f"merged ARL frame has {missing_values} missing feature values")
    events = sorted(int(value) for value in frame["event_id"].unique())
    summary: dict[str, object] = {
        "rows": len(frame),
        "events": len(events),
        "event_ids": events,
        "missing_feature_values": 0,
        "no_future_run_check": True,
    }
    if ARL_SOURCE_COLUMN in frame.columns:
        summary["source_batches"] = sorted(frame[ARL_SOURCE_COLUMN].unique().tolist())
        summary["archive_days"] = sorted(
            {
                pd.to_datetime(value, utc=True).date().isoformat()
                for value in frame[ARL_VALID_COLUMN]
            }
        )
    return summary


def recommend_archive_days(
    candidates: pd.DataFrame,
    covered_days: Iterable[date] = (),
    covered_events: Iterable[int] = (),
    *,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Greedily order archive days by how many not yet covered events they unlock."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if ARL_ARCHIVE_DAY_COLUMN not in candidates.columns:
        raise ValueError(f"candidate frame is missing {ARL_ARCHIVE_DAY_COLUMN}")
    blocked = {_coerce_day(day) for day in covered_days}
    seen = {int(event) for event in covered_events}
    events_by_day = {
        day: {int(value) for value in group["event_id"]}
        for day, group in candidates.groupby(ARL_ARCHIVE_DAY_COLUMN)
    }
    rows_by_day = {
        day: len(group) for day, group in candidates.groupby(ARL_ARCHIVE_DAY_COLUMN)
    }
    recommendations: list[dict[str, object]] = []
    while len(recommendations) < limit:
        options = [
            (len(events_by_day[day] - seen), rows_by_day[day], day)
            for day in sorted(events_by_day)
            if day not in blocked and events_by_day[day] - seen
        ]
        if not options:
            break
        new_events, rows, day = max(options, key=lambda item: (item[0], item[1], -item[2].toordinal()))
        recommendations.append(
            {
                "archive_day": day.isoformat(),
                "new_events": int(new_events),
                "new_event_ids": sorted(events_by_day[day] - seen),
                "rows": rows,
            }
        )
        blocked.add(day)
        seen |= events_by_day[day]
    return recommendations


def arl_feature_names(frame: pd.DataFrame) -> list[str]:
    """Return the ARL +3 hour columns present in a frame, in canonical order."""
    return [column for column in ARL_FEATURE_COLUMNS if column in frame.columns]
