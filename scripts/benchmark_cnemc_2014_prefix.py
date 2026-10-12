"""Evaluate the CNEMC 2014 extension on fixed 2015-2021 validation events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from haze_forecast.features import model_feature_names
from haze_forecast.model_benchmark import (
    benchmark_prefix_data_effect,
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
        "--base-data",
        type=Path,
        default=Path("data/processed/training_features.parquet"),
    )
    parser.add_argument(
        "--prefix-data",
        type=Path,
        default=Path("data/processed/cnemc_chengdu_2014_training_features.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cnemc_2014_prefix_benchmark"),
    )
    parser.add_argument("--splits", type=_at_least_three, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = pd.read_parquet(args.base_data)
    prefix = pd.read_parquet(args.prefix_data).copy()
    feature_names = model_feature_names(base)
    prefix["event_id"] = "cnemc_2014_" + prefix["event_id"].astype(str)
    report, predictions = benchmark_prefix_data_effect(
        base,
        prefix,
        feature_names,
        n_splits=args.splits,
        seed=args.seed,
    )
    write_benchmark_artifacts(report, predictions, args.output)
    print(
        json.dumps(
            {
                "base_data": str(args.base_data),
                "prefix_data": str(args.prefix_data),
                "output_dir": str(args.output),
                "base_rows": len(base),
                "prefix_rows": len(prefix),
                "prediction_rows": len(predictions),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
