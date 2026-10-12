"""Leakage-safe chronological evaluation helpers."""

from __future__ import annotations

import pandas as pd


def event_isolated_year_split(
    frame: pd.DataFrame, test_year: int, event_column: str = "event_key"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one calendar year without placing a cross-year event in training.

    The test partition contains rows observed during ``test_year``. Any event
    that remains active at the year's start is excluded from the training set,
    because its later recovery label would otherwise leak into the model.
    """
    required = {"timestamp", event_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"year split missing columns: {sorted(missing)}")

    result = frame.copy()
    timestamps = pd.to_datetime(result["timestamp"], errors="raise")
    result["timestamp"] = timestamps
    year_start = pd.Timestamp(f"{test_year}-01-01", tz=timestamps.dt.tz)
    event_end = result.groupby(event_column, sort=False)["timestamp"].max()
    train_events = event_end[event_end.lt(year_start)].index
    train = result[result[event_column].isin(train_events)].copy()
    test = result[timestamps.dt.year.eq(test_year) & ~result[event_column].isin(train_events)].copy()
    if train.empty or test.empty:
        raise ValueError("year split requires non-empty event-isolated train and test partitions")
    return train.sort_values("timestamp").reset_index(drop=True), test.sort_values("timestamp").reset_index(drop=True)
