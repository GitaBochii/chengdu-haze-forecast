"""Extract one bounded GEFS shard; publication availability requires separate audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from haze_forecast.operational_gefs import select_operational_gefs_run
from haze_forecast.operational_gefs_dataset import enrich_event_rows_with_operational_gefs
from haze_forecast.operational_gefs_quality import consolidate_verified_shards


def select_predictable_event_rows(frame: pd.DataFrame, *, year: int) -> pd.DataFrame:
    """Keep labelled particle-driven event rows from one source year."""
    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    result = frame.loc[
        timestamp.dt.tz_convert("Asia/Shanghai").dt.year.eq(year)
        & frame["event_id"].gt(0)
        & frame["remaining_hours"].notna()
        & frame["particle_primary"].fillna(False)
    ].copy()
    result["timestamp"] = timestamp.loc[result.index]
    return result.sort_values("timestamp").reset_index(drop=True)


def extraction_quality_summary(
    enriched: pd.DataFrame, requested: pd.DataFrame, source: pd.DataFrame,
) -> dict:
    checked = consolidate_verified_shards([enriched])

    def keys(frame: pd.DataFrame) -> set:
        return set(zip(pd.to_datetime(frame.timestamp, utc=True), frame.event_id, strict=True))

    requested_keys, source_keys = keys(requested), keys(source)
    verified_keys = keys(checked.loc[checked.gefs_row_verified.astype(bool)])
    order_columns = ['timestamp', 'gefs_run_utc', 'gefs_forecast_valid_utc_f006',
                     'gefs_forecast_valid_utc_f024']
    order_safe = False
    if not enriched.empty and set(order_columns).issubset(enriched):
        times = {name: pd.to_datetime(enriched[name], utc=True, errors='coerce')
                 for name in order_columns}
        order_safe = bool(
            times['gefs_run_utc'].le(times['timestamp']).all()
            and times['gefs_forecast_valid_utc_f006'].gt(times['timestamp']).all()
            and times['gefs_forecast_valid_utc_f024'].gt(times['timestamp']).all()
        )
    return {
        'coverage': {
            'complete_rows': len(verified_keys & requested_keys),
            'is_complete': bool(requested_keys and verified_keys == requested_keys == source_keys),
        },
        'run_valid_order_safe': order_safe,
        'publication_availability_verified': False,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--longitude", default=104.0665, type=float)
    parser.add_argument("--latitude", default=30.5723, type=float)
    parser.add_argument("--wgrib2", default=Path("tools/wgrib2/v3.1.3/wgrib2.exe"), type=Path)
    parser.add_argument("--run-offset", default=0, type=int)
    parser.add_argument("--max-runs", default=20, type=int)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.run_offset < 0 or args.max_runs < 1:
        raise ValueError("--run-offset must be non-negative and --max-runs must be positive")
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite an operational GEFS shard or manifest")
    selected = select_predictable_event_rows(pd.read_parquet(args.input), year=args.year)
    runs = sorted({select_operational_gefs_run(value.to_pydatetime())[0] for value in selected["timestamp"]})
    shard_runs = runs[args.run_offset : args.run_offset + args.max_runs]
    if not shard_runs:
        raise ValueError("the requested run offset has no matching GEFS cycle")
    requested = selected.loc[
        selected["timestamp"].map(lambda value: select_operational_gefs_run(value.to_pydatetime())[0])
        .isin(shard_runs)
    ].reset_index(drop=True)
    enriched = enrich_event_rows_with_operational_gefs(
        requested,
        longitude=args.longitude,
        latitude=args.latitude,
        wgrib2_executable=args.wgrib2,
    )
    quality = extraction_quality_summary(enriched, requested, selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input": str(args.input),
        "year": args.year,
        "run_offset": args.run_offset,
        "max_runs": args.max_runs,
        "unique_runs": len(shard_runs),
        "first_run_utc": shard_runs[0].isoformat(),
        "last_run_utc": shard_runs[-1].isoformat(),
        "requested_rows": len(requested),
        "produced_rows": len(enriched),
        **quality,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
