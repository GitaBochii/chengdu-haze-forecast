"""Unfilled station and city observations for isolated spatial model experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from haze_forecast.cnemc_2014 import CNEMC_TYPE_COLUMNS

POLLUTANTS = list(CNEMC_TYPE_COLUMNS.values())
METADATA_COLUMNS = ["station_id", "city", "longitude", "latitude"]


def load_station_metadata(path: Path) -> pd.DataFrame:
    """Read the existing Chengdu coordinate JSON or a normalized regional CSV."""
    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame([
            {"station_id": key, "city": "chengdu", "longitude": value[0], "latitude": value[1]}
            for key, value in records.items()
        ])
    else:
        frame = pd.read_csv(path, dtype={"station_id": str})
    missing = set(METADATA_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing station metadata columns: {sorted(missing)}")
    frame = frame[METADATA_COLUMNS].copy()
    if frame.empty or frame.isna().any().any() or frame.station_id.duplicated().any():
        raise ValueError("station metadata must have nonempty unique IDs and complete coordinates")
    if not frame.longitude.between(-180, 180).all() or not frame.latitude.between(-90, 90).all():
        raise ValueError("station coordinates out of range")
    return frame.sort_values("station_id").reset_index(drop=True)


def parse_station_daily(raw: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Preserve source hourly observations; never use 24-hour moving-average rows."""
    required = {"date", "hour", "type"}
    if not required.issubset(raw.columns):
        raise ValueError("missing date/hour/type columns")
    selected = raw.loc[raw.type.isin(CNEMC_TYPE_COLUMNS)].copy()
    if selected.duplicated(["date", "hour", "type"]).any():
        raise ValueError("duplicate date/hour/type observations")
    hours = pd.to_numeric(selected.hour, errors="raise")
    if not hours.between(0, 23).all() or not (hours % 1 == 0).all():
        raise ValueError("CNEMC hour must be an integer between 0 and 23")
    selected["timestamp"] = (
        pd.to_datetime(selected.date.astype(str), format="%Y%m%d")
        + pd.to_timedelta(hours, unit="h")
    ).dt.tz_localize("Asia/Shanghai")
    station_ids = metadata.station_id.tolist()
    for station_id in station_ids:
        if station_id not in selected:
            selected[station_id] = float("nan")
    long = selected.melt(
        id_vars=["timestamp", "type"], value_vars=station_ids,
        var_name="station_id", value_name="value",
    )
    long["value"] = pd.to_numeric(long.value, errors="coerce")
    frame = long.pivot(index=["timestamp", "station_id"], columns="type", values="value")
    frame = frame.rename(columns=CNEMC_TYPE_COLUMNS).reindex(columns=POLLUTANTS).reset_index()
    frame.columns.name = None
    frame = frame.merge(metadata, on="station_id", how="left", validate="many_to_one")
    frame["node_id"] = frame.station_id
    return frame.sort_values(["timestamp", "station_id"]).reset_index(drop=True)


def complete_station_hours(
    frame: pd.DataFrame, metadata: pd.DataFrame, start: str, end: str,
) -> pd.DataFrame:
    """Expose every missing hour as NaN, including missing days and trailing hours."""
    if frame.duplicated(["timestamp", "station_id"]).any():
        raise ValueError("duplicate timestamp/station_id observations")
    index = pd.MultiIndex.from_product([
        pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(23, unit="h"),
                      freq="h", tz="Asia/Shanghai"),
        metadata.station_id,
    ], names=["timestamp", "station_id"])
    result = frame.set_index(["timestamp", "station_id"])[POLLUTANTS].reindex(index)
    result = result.reset_index().merge(metadata, on="station_id", validate="many_to_one")
    result["node_id"] = result.station_id
    return result.sort_values(["timestamp", "station_id"]).reset_index(drop=True)


def aggregate_regional_city_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Match the existing CNEMC pipeline's station median (not official city AQI)."""
    groups = frame.groupby(["timestamp", "city"], sort=True)
    values = groups[POLLUTANTS].median()
    counts = groups[POLLUTANTS].count().add_suffix("_station_count")
    coordinates = frame.groupby("city")[["latitude", "longitude"]].mean()
    result = values.join(counts).reset_index().merge(coordinates, on="city")
    result["node_id"] = result.city
    return result.sort_values(["timestamp", "city"]).reset_index(drop=True)


def coverage_report(frame: pd.DataFrame, node_column: str) -> list[dict]:
    """Report coverage by calendar year and node without treating gaps as zero."""
    frame = frame.assign(year=frame.timestamp.dt.year)
    report = []
    for (year, node), group in frame.groupby(["year", node_column]):
        report.append({
            "year": int(year), "node_id": str(node), "hours": len(group),
            "first_timestamp": group.timestamp.min().isoformat(),
            "last_timestamp": group.timestamp.max().isoformat(),
            "observed_hours": {key: int(group[key].notna().sum()) for key in POLLUTANTS},
            "observed_fraction": {key: float(group[key].notna().mean()) for key in POLLUTANTS},
        })
    return report
