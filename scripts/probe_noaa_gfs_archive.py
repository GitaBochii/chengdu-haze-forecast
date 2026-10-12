"""Probe one historical NOAA GFS metadata endpoint without downloading a GRIB file."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from haze_forecast.forecast_sources import probe_gfs_opendap, write_probe_manifest


def parse_utc_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise argparse.ArgumentTypeError("--run-time 必须是带 +00:00 或 Z 的 UTC 时间")
    return parsed.astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-time", type=parse_utc_timestamp, default="2021-03-23T18:00:00+00:00")
    parser.add_argument("--lead-hours", type=int, default=6)
    parser.add_argument("--timeout-s", type=int, default=45)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/gfs_archive_probe_20210323_1800_f006.json"),
    )
    args = parser.parse_args()

    run_time = args.run_time
    if isinstance(run_time, str):
        run_time = parse_utc_timestamp(run_time)
    result = probe_gfs_opendap(run_time, args.lead_hours, timeout_s=args.timeout_s)
    write_probe_manifest(args.output, result)
    print(f"探测结果: {'可访问' if result.available else '不可访问'}")
    print(f"元数据地址: {result.url}")
    print(f"证据清单: {args.output}")
    if result.error:
        print(f"失败原因: {result.error}")


if __name__ == "__main__":
    main()
