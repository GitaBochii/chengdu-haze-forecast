"""Extract a bounded, no-leak 2021 GFS feature pilot for labelled haze events."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from haze_forecast.forecast_dataset import enrich_event_rows_with_gfs_forecast


def gfs_run_at_or_before(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp.tz_convert("UTC").floor("6h")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/city_labeled.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/gfs_event_features_f006_pilot.parquet")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/gfs_event_features_f006_pilot.json")
    )
    parser.add_argument("--max-runs", type=int, default=3)
    parser.add_argument("--run-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--message-workers", type=int, default=1)
    parser.add_argument("--message-timeout-s", type=int, default=45)
    parser.add_argument("--lead-hours", type=int, default=6)
    parser.add_argument("--longitude", type=float, default=104.06)
    parser.add_argument("--latitude", type=float, default=30.67)
    parser.add_argument(
        "--wgrib2",
        type=Path,
        default=Path("tools/wgrib2/v3.1.3/wgrib2.exe"),
    )
    args = parser.parse_args()
    if args.max_runs <= 0:
        raise ValueError("--max-runs 必须为正数")
    if args.run_offset < 0:
        raise ValueError("--run-offset 不能为负数")
    if args.workers <= 0:
        raise ValueError("--workers 必须为正数")
    if args.message_workers <= 0:
        raise ValueError("--message-workers 必须为正数")
    if args.message_timeout_s <= 0:
        raise ValueError("--message-timeout-s 必须为正数")

    frame = pd.read_parquet(args.input)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    event_rows = frame.loc[
        frame["timestamp"].ge(pd.Timestamp("2021-03-01", tz=UTC))
        & frame["event_id"].gt(0)
        & frame["remaining_hours"].notna()
        & frame["aqi"].gt(100)
    ].copy()
    event_rows["_gfs_run"] = event_rows["timestamp"].map(gfs_run_at_or_before)
    all_runs = sorted(event_rows["_gfs_run"].unique())
    selected_runs = all_runs[args.run_offset : args.run_offset + args.max_runs]
    if not selected_runs:
        raise ValueError("当前 --run-offset 和 --max-runs 没有匹配的 GFS 周期")
    selected_rows = event_rows.loc[event_rows["_gfs_run"].isin(selected_runs)].drop(columns="_gfs_run")
    enriched = enrich_event_rows_with_gfs_forecast(
        selected_rows,
        longitude=args.longitude,
        latitude=args.latitude,
        wgrib2_executable=args.wgrib2,
        lead_hours=args.lead_hours,
        max_workers=args.workers,
        message_workers=args.message_workers,
        message_timeout_s=args.message_timeout_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False)
    manifest = {
        "source": "NOAA public AWS GFS bucket, index + HTTP Range GRIB messages",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input": str(args.input),
        "output": str(args.output),
        "lead_hours": args.lead_hours,
        "event_rows": len(enriched),
        "gfs_runs": len(selected_runs),
        "run_offset": args.run_offset,
        "workers": args.workers,
        "message_workers": args.message_workers,
        "message_timeout_s": args.message_timeout_s,
        "first_run_utc": str(selected_runs[0]),
        "last_run_utc": str(selected_runs[-1]),
        "no_future_run_check": bool(
            enriched["gfs_forecast_run_utc"].le(enriched["timestamp"]).all()
        ),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 {len(enriched)} 行、{len(selected_runs)} 个 GFS 预报周期")
    print(f"特征文件: {args.output}")
    print(f"证据清单: {args.manifest}")


if __name__ == "__main__":
    main()
