"""Summarise completed NOAA ARL +3 hour batches and recommend the next archive days."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from haze_forecast.arl_dataset import (
    ARL_SOURCE_COLUMN,
    ARL_VALID_COLUMN,
    candidate_arl_rows,
    merge_arl_feature_batches,
    recommend_archive_days,
    validate_arl_feature_frame,
)


def _covered_days(frame: pd.DataFrame) -> list[str]:
    return sorted({value.date().isoformat() for value in pd.to_datetime(frame[ARL_VALID_COLUMN], utc=True)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/city_labeled.parquet"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--pattern", default="arl_f003_event_features_batch_*.parquet")
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/arl_batch_ledger.json")
    )
    parser.add_argument("--start", default="2019-06-01")
    parser.add_argument("--end", default="2021-03-01")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    batch_paths = sorted(args.processed_dir.glob(args.pattern))
    batches = {path.stem: pd.read_parquet(path) for path in batch_paths}
    ledger: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate_window": {"start": args.start, "end": args.end},
        "batches": [
            {
                "batch": name,
                "file": str(next(path for path in batch_paths if path.stem == name)),
                "rows": len(frame),
                "event_ids": sorted(int(value) for value in frame["event_id"].unique()),
                "archive_days": _covered_days(frame),
            }
            for name, frame in batches.items()
        ],
    }
    if batches:
        merged = merge_arl_feature_batches(batches)
        try:
            ledger["merged"] = validate_arl_feature_frame(merged)
        except ValueError as error:
            ledger["merged"] = {"validation_error": str(error)}
        ledger["covered_archive_days"] = _covered_days(merged)
        ledger["covered_event_ids"] = sorted(int(value) for value in merged["event_id"].unique())
        ledger["rows_per_source"] = (
            merged[ARL_SOURCE_COLUMN].value_counts().sort_index().to_dict()
        )
    else:
        merged = None
        ledger["covered_archive_days"] = []
        ledger["covered_event_ids"] = []
        ledger["merged"] = {"rows": 0, "events": 0}

    candidates = candidate_arl_rows(pd.read_parquet(args.input), args.start, args.end)
    ledger["candidate_rows"] = len(candidates)
    ledger["candidate_events"] = int(candidates["event_id"].nunique())
    ledger["next_archive_days"] = recommend_archive_days(
        candidates,
        [date.fromisoformat(day) for day in ledger["covered_archive_days"]],
        ledger["covered_event_ids"],
        limit=args.limit,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")

    merged_summary = ledger["merged"]
    print(
        f"已完成批次 {len(batches)} 个，合并后 {merged_summary.get('rows', 0)} 行、"
        f"{len(ledger['covered_event_ids'])} 个事件"
    )
    if "validation_error" in merged_summary:
        print(f"合并校验失败: {merged_summary['validation_error']}")
    print(f"候选池: {ledger['candidate_rows']} 行 / {ledger['candidate_events']} 个事件")
    print("建议下一批归档日:")
    for item in ledger["next_archive_days"]:
        print(
            f"  {item['archive_day']}  新事件 {item['new_events']} 个 "
            f"{item['new_event_ids']}  行数 {item['rows']}"
        )
    print(f"台账文件: {args.output}")


if __name__ == "__main__":
    main()
