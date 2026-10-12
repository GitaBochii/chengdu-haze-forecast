"""Build and benchmark a candidate using past-only pollution trajectory features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from haze_forecast.features import training_rows
from haze_forecast.trajectory_benchmark import (
    benchmark_trajectory_features,
    write_trajectory_artifacts,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", type=Path, default=root / "data/processed/city_labeled.parquet")
    parser.add_argument(
        "--training-output",
        type=Path,
        default=root / "data/processed/trajectory_training_features.parquet",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts/trajectory_feature_benchmark"
    )
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    training = training_rows(pd.read_parquet(args.labeled))
    args.training_output.parent.mkdir(parents=True, exist_ok=True)
    training.to_parquet(args.training_output, index=False)
    report, predictions = benchmark_trajectory_features(
        training, n_splits=args.splits, seed=args.seed
    )
    write_trajectory_artifacts(report, predictions, args.output)
    print(
        json.dumps(
            {
                "training_rows": len(training),
                "output": str(args.output),
                "decision": report["decision"]["decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
