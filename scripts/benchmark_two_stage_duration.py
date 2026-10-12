"""Benchmark a candidate-only two-stage pollution-duration model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from haze_forecast.features import model_feature_names
from haze_forecast.two_stage_benchmark import benchmark_two_stage, write_two_stage_artifacts


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/processed/training_features.parquet")
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts/two_stage_duration_benchmark"
    )
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_parquet(args.data)
    report, predictions = benchmark_two_stage(
        frame, model_feature_names(frame), n_splits=args.splits, seed=args.seed
    )
    write_two_stage_artifacts(report, predictions, args.output)
    print(json.dumps({"output": str(args.output), "decision": report["decision"]["decision"]}))


if __name__ == "__main__":
    main()
