"""Compare observation-only and ARL +3 hour models with whole events held out."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from haze_forecast.arl_benchmark import (
    ARL_MODEL_NAME,
    BASELINE_MODEL_NAME,
    DECISION_EVENT_MINIMUM,
    compare_feature_sets,
    summarise_per_event_direction,
)
from haze_forecast.arl_dataset import arl_feature_names, validate_arl_feature_frame
from haze_forecast.features import MEASUREMENT_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/arl_f003_event_features_merged.parquet"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/arl_feature_benchmark.json")
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    try:
        validation = validate_arl_feature_frame(frame)
    except ValueError as error:
        print(f"ARL 样本集校验失败: {error}")
        raise SystemExit(1) from error
    forecast_features = arl_feature_names(frame)
    if not forecast_features:
        print("输入样本集没有 ARL +3 小时特征列")
        raise SystemExit(1)

    try:
        payload = compare_feature_sets(
            frame,
            {
                BASELINE_MODEL_NAME: list(MEASUREMENT_COLUMNS),
                ARL_MODEL_NAME: [*MEASUREMENT_COLUMNS, *forecast_features],
            },
        )
    except ValueError as error:
        print(f"无法执行留一事件基准: {error}")
        raise SystemExit(1) from error
    direction = summarise_per_event_direction(pd.DataFrame(payload["per_event"]))
    payload["source"] = "NOAA ARL READY archived GFS 0.25-degree +3 hour subset features"
    payload["created_at_utc"] = datetime.now(UTC).isoformat()
    payload["input"] = str(args.input)
    payload["arl_feature_names"] = forecast_features
    payload["per_event_direction"] = direction
    payload["event_coverage"] = validation
    payload["evidence_level"] = (
        "decision"
        if int(payload["events"]) >= DECISION_EVENT_MINIMUM
        else "exploratory_only"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"样本: {payload['rows']} 行 / {payload['events']} 个事件")
    for record in payload["overall"]:
        print(f"  {record['model']}: MAE {record['mae']:.6f}")
    print(f"增强 MAE - 基线 MAE: {payload['mae_change_arl_minus_baseline']:.6f}")
    print(
        f"每事件方向: 改善 {direction.get('improved', 0)} 个、"
        f"恶化 {direction.get('worsened', 0)} 个、共 {direction.get('events', 0)} 个"
    )
    print(
        "样本集中度: 最大事件行数占比 "
        f"{payload['largest_event_row_share']:.3f} (event {payload['largest_event_id']}); "
        f"最大事件误差占比 {payload['largest_event_error_share']:.3f} "
        f"(event {payload['largest_error_event_id']})"
    )
    print(f"证据等级: {payload['evidence_level']}  决策: {payload['decision']}")
    print(f"报告文件: {args.output}")


if __name__ == "__main__":
    main()
