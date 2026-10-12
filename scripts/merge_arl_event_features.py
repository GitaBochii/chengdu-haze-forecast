"""Merge every NOAA ARL +3 hour batch into one validated evaluation frame."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from haze_forecast.arl_dataset import merge_arl_feature_batches, validate_arl_feature_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--pattern", default="arl_f003_event_features_batch_*.parquet")
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/arl_f003_event_features_merged.parquet")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/arl_f003_event_features_merged.json")
    )
    args = parser.parse_args()

    batch_paths = sorted(args.processed_dir.glob(args.pattern))
    if not batch_paths:
        print(f"没有找到匹配 {args.pattern} 的 ARL 批次文件")
        raise SystemExit(1)
    batches: dict[str, pd.DataFrame] = {}
    batch_records: list[dict[str, object]] = []
    for path in batch_paths:
        frame = pd.read_parquet(path)
        batches[path.stem] = frame
        batch_records.append({"batch": path.stem, "file": str(path), "rows": len(frame)})
    try:
        merged = merge_arl_feature_batches(batches)
        summary = validate_arl_feature_frame(merged)
    except ValueError as error:
        print(f"ARL 批次合并校验失败: {error}")
        raise SystemExit(1) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    manifest = {
        "source": "NOAA ARL READY archived GFS 0.25-degree subset service",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "output": str(args.output),
        "batches": batch_records,
        **summary,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已合并 {len(batches)} 个批次: {summary['rows']} 行、{summary['events']} 个事件")
    print(f"合并文件: {args.output}")


if __name__ == "__main__":
    main()
