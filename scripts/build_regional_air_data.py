"""Build auditable, unfilled pollution sequences without changing deployed data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from haze_forecast.regional_air_data import (
    aggregate_regional_city_hourly,
    complete_station_hours,
    coverage_report,
    load_station_metadata,
    parse_station_daily,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/data/chengdu/stations.json"))
    parser.add_argument("--directories", nargs="+", type=Path, default=[
        Path("data/raw/cnemc_2023_daily"), Path("data/raw/cnemc_2024_2025_daily"),
        Path("data/raw/cnemc_2026_daily"),
    ])
    parser.add_argument("--output", type=Path, default=Path("data/processed/regional_air_models"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/regional_air_models/data_audit"))
    args = parser.parse_args()
    metadata = load_station_metadata(args.metadata)
    filenames = sorted(path for directory in args.directories
                       for path in directory.glob("china_sites_????????.csv"))
    if not filenames:
        raise ValueError("no daily source files")
    names = [path.name for path in filenames]
    if len(names) != len(set(names)):
        raise ValueError("duplicate input daily files")
    needed = {"date", "hour", "type", *metadata.station_id}
    frames = []
    for number, path in enumerate(filenames, 1):
        raw = pd.read_csv(path, usecols=lambda name: name in needed)
        frames.append(parse_station_daily(raw, metadata))
        if number % 100 == 0:
            print(f"parsed {number}/{len(filenames)} days", flush=True)
    dates = sorted(pd.to_datetime([name[12:20] for name in names], format="%Y%m%d"))
    station = complete_station_hours(
        pd.concat(frames, ignore_index=True), metadata,
        str(dates[0].date()), str(dates[-1].date()),
    )
    city = aggregate_regional_city_hourly(station)
    args.output.mkdir(parents=True, exist_ok=True)
    args.audit.mkdir(parents=True, exist_ok=True)
    station.to_parquet(args.output / "station_hourly.parquet", index=False)
    city.to_parquet(args.output / "city_hourly.parquet", index=False)
    metadata.to_csv(args.output / "station_metadata.csv", index=False)
    report = {
        "source": "Local CNEMC daily CSV archives mirrored by https://quotsoft.net/air/",
        "metadata_path": str(args.metadata.resolve()),
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "scope": "chengdu_station_pilot" if metadata.city.nunique() == 1 else "regional_cities",
        "timezone": "Asia/Shanghai", "imputation": "none",
        "city_aggregation": "median across available stations, matching cnemc_2014.py",
        "coordinate_aggregation": "arithmetic mean of selected station coordinates per city",
        "limits": [
            "Static metadata does not establish station relocation history.",
            "City AQI is the existing project's station median, not an official city AQI calculation.",
            "Missing station/hour values remain NaN; downstream training must fit imputation on training data only.",
            "A Chengdu-only graph cannot validate regional intercity pollution transport.",
        ],
        "input_daily_file_count": len(filenames),
        "missing_daily_files": [str(day.date()) for day in pd.date_range(dates[0], dates[-1])
                                if day not in set(dates)],
        "station_rows": len(station), "city_rows": len(city),
        "station_coverage": coverage_report(station, "station_id"),
        "city_coverage": coverage_report(city, "city"),
        "source_files": [{"path": str(path.resolve()), "bytes": path.stat().st_size}
                         for path in filenames],
    }
    (args.audit / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["scope", "input_daily_file_count", "station_rows", "city_rows"]}))


if __name__ == "__main__":
    main()
