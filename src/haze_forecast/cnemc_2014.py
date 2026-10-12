"""Read CNEMC wide station archives into Chengdu city-hourly pollution."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

CNEMC_TYPE_COLUMNS = {
    "AQI": "aqi",
    "PM2.5": "pm2_5",
    "PM10": "pm10",
    "SO2": "so2",
    "NO2": "no2",
    "CO": "co",
    "O3": "o3",
}


def aggregate_cnemc_hourly(raw: pd.DataFrame, *, station_ids: Sequence[str]) -> pd.DataFrame:
    """Aggregate one CNEMC daily wide file using median values across Chengdu stations."""
    required = {"date", "hour", "type"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"missing CNEMC columns: {sorted(missing)}")
    selected_stations = [station_id for station_id in station_ids if station_id in raw.columns]
    if not selected_stations:
        raise ValueError("no requested Chengdu station columns exist in the CNEMC file")
    selected = raw.loc[raw["type"].isin(CNEMC_TYPE_COLUMNS), ["date", "hour", "type", *selected_stations]]
    long = selected.melt(
        id_vars=["date", "hour", "type"], value_vars=selected_stations, value_name="value"
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    grouped = long.groupby(["date", "hour", "type"], as_index=False)["value"].median()
    hourly = grouped.pivot(index=["date", "hour"], columns="type", values="value").reset_index()
    hourly = hourly.rename(columns=CNEMC_TYPE_COLUMNS)
    for column in CNEMC_TYPE_COLUMNS.values():
        if column not in hourly:
            hourly[column] = float("nan")
    hour_values = pd.to_numeric(hourly["hour"], errors="raise")
    if not hour_values.between(0, 23).all():
        raise ValueError("CNEMC hour must be between 0 and 23")
    timestamp_text = hourly["date"].astype(str) + hour_values.astype(int).astype(str).str.zfill(2)
    hourly["timestamp"] = pd.to_datetime(timestamp_text, format="%Y%m%d%H").dt.tz_localize(
        "Asia/Shanghai"
    )
    return hourly[["timestamp", *CNEMC_TYPE_COLUMNS.values()]].sort_values("timestamp").reset_index(
        drop=True
    )


def load_cnemc_zip(archive_path: Path, *, station_ids: Sequence[str]) -> pd.DataFrame:
    """Stream daily CNEMC station CSV files from one ZIP into a sorted hourly table."""
    daily_frames: list[pd.DataFrame] = []
    with ZipFile(archive_path) as archive:
        filenames = sorted(
            name
            for name in archive.namelist()
            if name.rsplit("/", maxsplit=1)[-1].startswith("china_sites_")
            and name.endswith(".csv")
        )
        if not filenames:
            raise ValueError("CNEMC ZIP contains no daily station CSV files")
        for filename in filenames:
            with archive.open(filename) as handle:
                daily_frames.append(aggregate_cnemc_hourly(pd.read_csv(handle), station_ids=station_ids))
    return pd.concat(daily_frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def load_cnemc_daily_directory(directory: Path, *, station_ids: Sequence[str]) -> pd.DataFrame:
    """Load independently downloaded CNEMC daily station CSV files in timestamp order."""
    filenames = sorted(directory.glob("china_sites_????????.csv"))
    if not filenames:
        raise ValueError("CNEMC directory contains no daily station CSV files")
    daily_frames = [
        aggregate_cnemc_hourly(pd.read_csv(filename), station_ids=station_ids)
        for filename in filenames
    ]
    return pd.concat(daily_frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
