"""Download and aggregate Chengdu's 2024-2025 hourly CNEMC observations.

The downloader is resumable: completed nonempty daily CSVs are never fetched
again.  It writes the city aggregate separately so later weather enrichment can
be audited without rereading every national-station file.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import requests

from haze_forecast.cnemc_2014 import aggregate_cnemc_hourly
from haze_forecast.cnemc_download import download_cnemc_daily_csv, inclusive_dates

ROOT = Path(__file__).resolve().parents[1]


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("日期必须是 YYYY-MM-DD") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=parse_date, default=date(2024, 1, 1))
    parser.add_argument("--end", type=parse_date, default=date(2025, 12, 31))
    parser.add_argument("--max-days", type=int, default=0, help="0 表示抓取整个区间")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument(
        "--raw-dir", type=Path, default=ROOT / "data/raw/cnemc_2024_2025_daily"
    )
    parser.add_argument(
        "--stations", type=Path, default=ROOT / "data/raw/data/chengdu/stations.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/interim/cnemc_chengdu_2024_2025_hourly.parquet"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "artifacts/cnemc_2024_2025_ingest/report.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_days < 0:
        raise ValueError("--max-days 不能为负数")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds 必须为正数")
    station_map = json.loads(args.stations.read_text(encoding="utf-8"))
    if not isinstance(station_map, dict) or not station_map:
        raise ValueError("stations.json 必须是非空对象")
    selected_days = list(inclusive_dates(args.start, args.end))
    if args.max_days:
        selected_days = selected_days[: args.max_days]
    if not selected_days:
        raise ValueError("请求的日期范围为空")

    hourly_frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for index, day in enumerate(selected_days, start=1):
        daily_path = args.raw_dir / f"china_sites_{day:%Y%m%d}.csv"
        try:
            downloaded = download_cnemc_daily_csv(
                day, daily_path, timeout_seconds=args.timeout_seconds
            )
            daily = pd.read_csv(downloaded, usecols=lambda name: name in {"date", "hour", "type", *station_map})
            hourly_frames.append(aggregate_cnemc_hourly(daily, station_ids=list(station_map)))
        except (OSError, pd.errors.ParserError, requests.RequestException, ValueError) as error:
            failures.append({"date": day.isoformat(), "error": f"{type(error).__name__}: {error}"})
        if index % 20 == 0 or index == len(selected_days):
            print(f"已处理 {index}/{len(selected_days)} 天，成功 {len(hourly_frames)} 天，失败 {len(failures)} 天")
    if not hourly_frames:
        raise RuntimeError("没有获得任何可聚合的 CNEMC 日数据")
    hourly = pd.concat(hourly_frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(args.output, index=False)
    manifest = {
        "source": "quotsoft.net daily CNEMC station CSV mirror; underlying source CNEMC",
        "requested_interval": {"start": args.start.isoformat(), "end": args.end.isoformat()},
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "station_ids": list(station_map),
        "requested_days": len(selected_days),
        "successful_days": len(hourly_frames),
        "failed_days": failures,
        "hourly_rows": len(hourly),
        "first_timestamp": hourly["timestamp"].min().isoformat(),
        "last_timestamp": hourly["timestamp"].max().isoformat(),
        "nonnull_rates": {name: float(hourly[name].notna().mean()) for name in hourly.columns if name != "timestamp"},
        "output": str(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"城市小时数据: {args.output}")
    print(f"审计报告: {args.manifest}")


if __name__ == "__main__":
    main()
