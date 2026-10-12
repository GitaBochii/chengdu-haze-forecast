"""Build an auditable 2014 Chengdu pollution-only extension from CNEMC data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from haze_forecast.cnemc_2014 import CNEMC_TYPE_COLUMNS, load_cnemc_zip
from haze_forecast.labels import label_city_data, label_summary


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("data/raw/20140513-20141231_CNEMC.zip")
    )
    parser.add_argument(
        "--stations", type=Path, default=Path("data/raw/data/chengdu/stations.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/interim/cnemc_chengdu_2014_hourly.parquet")
    )
    parser.add_argument(
        "--labeled-output",
        type=Path,
        default=Path("data/processed/cnemc_chengdu_2014_labeled.parquet"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/cnemc_2014_extension/report.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stations = json.loads(args.stations.read_text(encoding="utf-8"))
    if not isinstance(stations, dict) or not stations:
        raise ValueError("stations.json must contain a non-empty station mapping")
    hourly = load_cnemc_zip(args.archive, station_ids=list(stations))
    labeled = label_city_data(hourly)
    expected_timestamps = pd.date_range(
        "2014-05-13 00:00:00+08:00", "2014-12-31 23:00:00+08:00", freq="h"
    )
    observed_timestamps = pd.DatetimeIndex(hourly["timestamp"])
    missing_timestamps = expected_timestamps.difference(observed_timestamps)
    hours_per_day = hourly.groupby(hourly["timestamp"].dt.strftime("%Y-%m-%d")).size()
    with ZipFile(args.archive) as archive:
        daily_file_count = sum(
            name.startswith("2014/china_sites_") and name.endswith(".csv")
            for name in archive.namelist()
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.labeled_output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(args.output, index=False)
    labeled.to_parquet(args.labeled_output, index=False)
    pollutant_columns = list(CNEMC_TYPE_COLUMNS.values())
    payload = {
        "source": "Zenodo 10836299 CNEMC historical site observations",
        "source_archive": str(args.archive),
        "archive_md5": file_md5(args.archive),
        "expected_archive_md5": "8837a297e3dfa751c68aa2b6a4c1d56c",
        "stations": len(stations),
        "daily_file_count": daily_file_count,
        "rows": len(hourly),
        "first_timestamp": hourly["timestamp"].min().isoformat(),
        "last_timestamp": hourly["timestamp"].max().isoformat(),
        "expected_hour_count": len(expected_timestamps),
        "missing_hour_count": len(missing_timestamps),
        "full_missing_dates": [
            date for date, count in hours_per_day.reindex(
                expected_timestamps.strftime("%Y-%m-%d").unique(), fill_value=0
            ).items() if count == 0
        ],
        "hourly_continuity": bool(hourly["timestamp"].diff().iloc[1:].eq(timedelta(hours=1)).all()),
        "pollutant_nonnull_rate": {
            column: float(hourly[column].notna().mean()) for column in pollutant_columns
        },
        "label_summary": label_summary(labeled),
        "weather_fields": "not included; not eligible for default-model training",
        "output": str(args.output),
        "labeled_output": str(args.labeled_output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
