"""Isolated unfilled historical panel with explicit eight-hour ozone semantics."""

from __future__ import annotations

import pandas as pd

from haze_forecast.regional_air_data import (
    POLLUTANTS as LEGACY_POLLUTANTS,
)
from haze_forecast.regional_air_data import (
    aggregate_regional_city_hourly,
    complete_station_hours,
    coverage_report,
    parse_station_daily,
)

POLLUTANTS = ["o3_8h" if key == "o3" else key for key in LEGACY_POLLUTANTS]


def parse_history_daily(raw: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Use exact CNEMC O3_8h only; the legacy parser remains unchanged."""
    if "type" not in raw:
        raise ValueError("missing date/hour/type columns")
    selected = raw.loc[raw.type != "O3"].copy()
    selected["type"] = selected.type.replace({"O3_8h": "O3"})
    return parse_station_daily(selected, metadata).rename(columns={"o3": "o3_8h"})


def normalize_history_interim(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Map audited workbook-derived o3 (O3-8h) without carrying old weather."""
    required = {"timestamp", "station_id", *LEGACY_POLLUTANTS}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing interim columns: {sorted(required - set(frame.columns))}")
    result = frame[["timestamp", "station_id", *LEGACY_POLLUTANTS]].copy()
    result["timestamp"] = pd.to_datetime(result.timestamp, errors="raise")
    if result.timestamp.dt.tz is None:
        result["timestamp"] = result.timestamp.dt.tz_localize("Asia/Shanghai")
    else:
        result["timestamp"] = result.timestamp.dt.tz_convert("Asia/Shanghai")
    if result.station_id.isna().any() or not result.station_id.isin(metadata.station_id).all():
        raise ValueError("unknown station in historical interim")
    if result.timestamp.isna().any() or not result.timestamp.eq(result.timestamp.dt.floor("h")).all():
        raise ValueError("invalid historical hourly timestamp")
    if result.duplicated(["timestamp", "station_id"]).any():
        raise ValueError("duplicate timestamp/station_id observations")
    result[LEGACY_POLLUTANTS] = result[LEGACY_POLLUTANTS].apply(pd.to_numeric, errors="raise")
    result = result.rename(columns={"o3": "o3_8h"})
    result = result.merge(metadata, on="station_id", validate="many_to_one")
    result["node_id"] = result.station_id
    return result.sort_values(["timestamp", "station_id"]).reset_index(drop=True)


def complete_history_hours(
    frame: pd.DataFrame, metadata: pd.DataFrame, start: str, end: str,
) -> pd.DataFrame:
    """Fill only the time grid, rejecting rather than clipping outside observations."""
    first = pd.Timestamp(start, tz="Asia/Shanghai")
    last = pd.Timestamp(f"{end} 23:00:00", tz="Asia/Shanghai")
    if not frame.timestamp.between(first, last).all():
        raise ValueError("observations outside requested historical period")
    if not frame.station_id.isin(metadata.station_id).all():
        raise ValueError("unknown station in historical panel")
    return complete_station_hours(
        frame.rename(columns={"o3_8h": "o3"}), metadata, start, end,
    ).rename(columns={"o3": "o3_8h"})


def aggregate_history_city_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    return aggregate_regional_city_hourly(frame.rename(columns={"o3_8h": "o3"})).rename(
        columns={"o3": "o3_8h", "o3_station_count": "o3_8h_station_count"},
    )


def history_coverage_report(frame: pd.DataFrame, node_column: str) -> list[dict]:
    report = coverage_report(frame.rename(columns={"o3_8h": "o3"}), node_column)
    for row in report:
        for key in ["observed_hours", "observed_fraction"]:
            row[key]["o3_8h"] = row[key].pop("o3")
    return report
