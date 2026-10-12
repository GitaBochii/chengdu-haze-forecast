"""Extract a bounded five-member GEFSv12 PBL feature pilot for haze events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from haze_forecast.gefs_dataset import enrich_event_rows_with_gefs_pbl
from haze_forecast.gefs_features import select_reforecast_run_and_lead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/city_labeled.parquet"))
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--longitude", type=float, default=104.0665)
    parser.add_argument("--latitude", type=float, default=30.5723)
    parser.add_argument("--horizon-hours", type=int, default=6)
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument(
        "--wgrib2", type=Path, default=Path("tools/wgrib2/v3.1.3/wgrib2.exe")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/gefs_v12_pbl_event_pilot.parquet")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/gefs_v12_pbl_event_pilot/report.json")
    )
    args = parser.parse_args()
    if args.max_pairs <= 0:
        parser.error("--max-pairs 必须为正整数")
    if args.horizon_hours <= 0:
        parser.error("--horizon-hours 必须为正整数")
    if args.timeout_s <= 0:
        parser.error("--timeout-s 必须为正整数")
    return args


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    events = frame.loc[
        frame["event_id"].gt(0)
        & frame["remaining_hours"].notna()
        & frame["timestamp"].dt.year.eq(args.year)
    ].copy()
    if events.empty:
        raise ValueError(f"{args.year} 年没有可提取的已标注事件行")
    events["_gefs_key"] = events["timestamp"].map(
        lambda timestamp: select_reforecast_run_and_lead(
            timestamp.to_pydatetime(), horizon_hours=args.horizon_hours
        )
    )
    selected_keys = sorted(events["_gefs_key"].unique(), key=lambda key: (key[0], key[1]))[
        : args.max_pairs
    ]
    selected = events.loc[events["_gefs_key"].isin(selected_keys)].drop(columns="_gefs_key")
    enriched = enrich_event_rows_with_gefs_pbl(
        selected,
        longitude=args.longitude,
        latitude=args.latitude,
        wgrib2_executable=args.wgrib2,
        horizon_hours=args.horizon_hours,
        timeout_s=args.timeout_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False)
    payload = {
        "source": "NOAA GEFSv12 retrospective public S3 archive",
        "retrospective_only": True,
        "input": str(args.input),
        "output": str(args.output),
        "year": args.year,
        "horizon_hours": args.horizon_hours,
        "selected_run_lead_pairs": [
            {"run_time_utc": run_time.isoformat(), "lead_hours": lead_hours}
            for run_time, lead_hours in selected_keys
        ],
        "event_rows": len(enriched),
        "no_past_valid_time": bool(
            enriched["gefs_forecast_valid_utc"].ge(
                enriched["timestamp"] + pd.Timedelta(f"{args.horizon_hours}h")
            ).all()
        ),
        "feature_columns": [
            "gefs_pblh_median_m",
            "gefs_pblh_std_m",
            "gefs_pblh_p10_m",
            "gefs_pblh_p90_m",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
