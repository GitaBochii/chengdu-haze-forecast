"""Benchmark the reviewed history-only candidate models and persist audit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from haze_forecast.features import model_feature_names
from haze_forecast.model_benchmark import (
    benchmark_history_models,
    write_benchmark_artifacts,
)


def _at_least_three(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer of at least 3") from error
    if parsed < 3:
        raise argparse.ArgumentTypeError("must be an integer of at least 3")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/training_features.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/history_model_benchmark"),
    )
    parser.add_argument("--splits", type=_at_least_three, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    feature_names = model_feature_names(frame)
    report, predictions = benchmark_history_models(
        frame,
        feature_names,
        n_splits=args.splits,
        seed=args.seed,
    )
    write_benchmark_artifacts(report, predictions, args.output)
    print(
        json.dumps(
            {
                "data": str(args.data),
                "output_dir": str(args.output),
                "prediction_rows": len(predictions),
                "models": list(report["models"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
