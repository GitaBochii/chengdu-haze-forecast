"""Extract a bounded no-leak NOAA ARL GFS +3-hour feature batch for haze events."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from haze_forecast.arl import read_arl_surface_forecast
from haze_forecast.arl_dataset import (
    ARL_ARCHIVE_DAY_COLUMN,
    ARL_FEATURE_COLUMNS,
    ARL_RUN_COLUMN,
    ARL_VALID_COLUMN,
    candidate_arl_rows,
    resolve_archive_days,
)
from haze_forecast.arl_ready import download_ready_gfs_day


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/city_labeled.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/arl_f003_event_features_pilot.parquet")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/arl_f003_event_features_pilot.json")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/arl_ready_cache"))
    parser.add_argument("--start", default="2019-06-01")
    parser.add_argument("--end", default="2021-03-01")
    parser.add_argument("--max-days", type=int, default=2)
    parser.add_argument("--day-offset", type=int, default=0)
    parser.add_argument(
        "--days",
        nargs="*",
        default=None,
        help="explicit archive days (YYYY-MM-DD); overrides --day-offset when given",
    )
    parser.add_argument("--longitude", type=float, default=104.06)
    parser.add_argument("--latitude", type=float, default=30.67)
    parser.add_argument("--poll-interval-s", type=float, default=10.0)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    candidates = candidate_arl_rows(frame, args.start, args.end)
    selected_days = resolve_archive_days(
        candidates[ARL_ARCHIVE_DAY_COLUMN].unique(),
        days=args.days,
        day_offset=args.day_offset,
        max_days=args.max_days,
    )
    selected = candidates.loc[candidates[ARL_ARCHIVE_DAY_COLUMN].isin(selected_days)].copy()

    decoded = {}
    for day in selected_days:
        day_times = sorted(
            pd.to_datetime(
                selected.loc[selected[ARL_ARCHIVE_DAY_COLUMN].eq(day), ARL_VALID_COLUMN].unique()
            )
        )
        valid_hours = [pd.Timestamp(value).hour for value in day_times]
        subset = download_ready_gfs_day(
            day,
            args.cache_dir,
            start_hour=min(valid_hours) - 3,
            end_hour=max(valid_hours),
            poll_interval_s=args.poll_interval_s,
        )
        for valid_at in day_times:
            decoded[pd.Timestamp(valid_at)] = read_arl_surface_forecast(
                subset,
                longitude=args.longitude,
                latitude=args.latitude,
                valid_at=pd.Timestamp(valid_at).to_pydatetime(),
            )
    values = selected[ARL_VALID_COLUMN].map(decoded)
    selected["arl_grid_longitude"] = values.map(lambda item: item.grid_longitude)
    selected["arl_grid_latitude"] = values.map(lambda item: item.grid_latitude)
    selected["arl_temperature_c_f003"] = values.map(lambda item: item.temperature_c)
    selected["arl_relative_humidity_pct_f003"] = values.map(lambda item: item.relative_humidity_pct)
    selected["arl_wind_u_ms_f003"] = values.map(lambda item: item.wind_u_ms)
    selected["arl_wind_v_ms_f003"] = values.map(lambda item: item.wind_v_ms)
    selected["arl_boundary_layer_height_m_f003"] = values.map(
        lambda item: item.boundary_layer_height_m
    )
    selected = selected.drop(columns=ARL_ARCHIVE_DAY_COLUMN)
    no_future_run = bool(selected[ARL_RUN_COLUMN].le(selected["timestamp"]).all())
    if not no_future_run:
        raise RuntimeError("ARL 特征使用了事件发生后的预报周期")
    missing_values = int(selected[list(ARL_FEATURE_COLUMNS)].isna().sum().sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False)
    manifest = {
        "source": "NOAA ARL READY archived GFS 0.25-degree subset service",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input": str(args.input),
        "output": str(args.output),
        "candidate_rule": "latest six-hour cycle plus f003; keep only rows before its valid time",
        "event_rows": len(selected),
        "event_ids": sorted(int(value) for value in selected["event_id"].unique()),
        "rows_per_event": {
            str(int(event_id)): int(rows)
            for event_id, rows in selected["event_id"].value_counts().sort_index().items()
        },
        "archive_days": [day.isoformat() for day in selected_days],
        "no_future_run_check": no_future_run,
        "missing_feature_values": missing_values,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 {len(selected)} 行、{len(selected_days)} 个 ARL 日文件的 +3 小时预报特征")
    print(f"特征文件: {args.output}")


if __name__ == "__main__":
    main()
