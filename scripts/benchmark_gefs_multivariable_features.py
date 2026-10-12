"""Benchmark the bounded GEFS multivariable pilot without changing deployable models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from haze_forecast.features import MEASUREMENT_COLUMNS
from haze_forecast.gefs_benchmark import (
    benchmark_gefs_multivariable_features,
    prepare_gefs_benchmark_frame,
    write_gefs_benchmark_artifacts,
)
from haze_forecast.gefs_dataset import GEFS_SURFACE_FEATURE_COLUMNS


def _at_least_three(value: str) -> int:
    parsed = int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("必须至少为 3")
    return parsed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-data", type=Path, default=root / "data/processed/city_labeled.parquet")
    parser.add_argument(
        "--gefs-pilot", type=Path, default=root / "data/processed/gefs_multivariable_event_pilot.parquet"
    )
    parser.add_argument("--output", type=Path, default=root / "artifacts/gefs_multivariable_event_pilot")
    parser.add_argument("--splits", type=_at_least_three, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepared, gefs_features = prepare_gefs_benchmark_frame(
        pd.read_parquet(args.city_data), pd.read_parquet(args.gefs_pilot)
    )
    baseline_features = [name for name in MEASUREMENT_COLUMNS if name in prepared.columns]
    report, predictions = benchmark_gefs_multivariable_features(
        prepared,
        baseline_features=baseline_features,
        gefs_features=list(GEFS_SURFACE_FEATURE_COLUMNS),
        n_splits=args.splits,
        seed=args.seed,
    )
    write_gefs_benchmark_artifacts(report, predictions, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(prepared),
                "gefs_feature_count": len(gefs_features),
                "decision": report["decision"]["decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
