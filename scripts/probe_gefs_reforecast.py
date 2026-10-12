"""Download a bounded GEFSv12 retrospective-message sample and write evidence."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import requests

from haze_forecast.gefs_reforecast import (
    GEFS_REFORECAST_MEMBERS,
    GEFS_REFORECAST_VARIABLES,
    download_gefs_reforecast_message,
)


def parse_run_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise argparse.ArgumentTypeError("--run-time 必须是 UTC 时间")
    return parsed.astimezone(UTC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-time", type=parse_run_time, default="2015-01-01T00:00:00Z")
    parser.add_argument("--variable", choices=GEFS_REFORECAST_VARIABLES, default="pbl_hgt")
    parser.add_argument("--lead-hours", type=int, default=6)
    parser.add_argument("--members", nargs="+", choices=GEFS_REFORECAST_MEMBERS)
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/gefs_v12_reforecast_probe/report.json")
    )
    args = parser.parse_args()
    if args.lead_hours <= 0:
        parser.error("--lead-hours 必须为正整数")
    if args.timeout_s <= 0:
        parser.error("--timeout-s 必须为正整数")
    if args.members is None:
        args.members = GEFS_REFORECAST_MEMBERS
    return args


def main() -> None:
    args = parse_args()
    records: list[dict[str, object]] = []
    for member in args.members:
        try:
            message = download_gefs_reforecast_message(
                args.run_time,
                member=member,
                variable=args.variable,
                lead_hours=args.lead_hours,
                timeout_s=args.timeout_s,
            )
            records.append(
                {
                    "member": member,
                    "available": True,
                    "source_url": message.source_url,
                    "index_url": message.index_url,
                    "byte_range": list(message.byte_range),
                    "downloaded_bytes": len(message.content),
                    "sha256": message.sha256,
                }
            )
        except (requests.RequestException, RuntimeError, ValueError, UnicodeDecodeError) as error:
            records.append(
                {"member": member, "available": False, "error": f"{type(error).__name__}: {error}"}
            )

    payload = {
        "source": "NOAA GEFSv12 retrospective public S3 archive",
        "retrospective_only": True,
        "run_time_utc": args.run_time.isoformat(),
        "variable": args.variable,
        "lead_hours": args.lead_hours,
        "requested_member_count": len(args.members),
        "available_member_count": sum(record["available"] for record in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if payload["available_member_count"] != payload["requested_member_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
