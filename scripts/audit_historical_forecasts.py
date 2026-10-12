"""Publish no-leak evidence for locally probed historical forecast sources."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from haze_forecast.forecast_audit import audit_local_gfs_sources, write_audit_report


def parse_utc_timestamp(value: str) -> datetime:
    """Parse an explicit UTC timestamp for a bounded, metadata-only audit."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("--run-time 必须是带时区的 UTC 时间")
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise argparse.ArgumentTypeError("--run-time 必须使用 UTC（+00:00 或 Z）")
    return parsed.astimezone(UTC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/historical_forecast_audit"))
    parser.add_argument("--run-time", type=parse_utc_timestamp, default="2021-03-23T18:00:00+00:00")
    parser.add_argument("--lead-hours", type=int, default=6)
    args = parser.parse_args()
    if not 0 <= args.lead_hours <= 72:
        parser.error("--lead-hours 必须在 0 到 72 之间")
    if isinstance(args.run_time, str):
        args.run_time = parse_utc_timestamp(args.run_time)
    return args


def main() -> None:
    args = parse_args()
    records = audit_local_gfs_sources(
        args.processed_dir,
        run_time=args.run_time,
        lead_hours=args.lead_hours,
    )
    output = args.output if args.output.suffix.lower() == ".json" else args.output / "report.json"
    write_audit_report(output, records)
    print(
        json.dumps(
            {
                "eligible": any(record["eligible"] for record in records),
                "record_count": len(records),
                "report": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
