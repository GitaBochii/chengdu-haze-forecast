"""Strictly test same-hour station-direction features against the prior spatial candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from haze_forecast.features import model_feature_names, training_rows
from haze_forecast.spatial import enrich_city_with_spatial_features
from haze_forecast.spatial_benchmark import (
    benchmark_spatial_features,
    write_spatial_benchmark_artifacts,
)

DIRECTIONAL_FEATURES = {
    "spatial_wind_direction_consistency",
}
DIRECTIONAL_SUFFIXES = (
    "_upwind_downwind_difference",
    "_gradient_median",
    "_transport_pair_count",
)


def _at_least_three(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer of at least 3") from error
    if parsed < 3:
        raise argparse.ArgumentTypeError("must be an integer of at least 3")
    return parsed


def _is_directional_feature(name: str) -> bool:
    return name in DIRECTIONAL_FEATURES or name.endswith(DIRECTIONAL_SUFFIXES)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-data", type=Path, default=root / "data/processed/city_labeled.parquet")
    parser.add_argument("--station-data", type=Path, default=root / "data/interim/station_hourly.parquet")
    parser.add_argument("--coordinates", type=Path, default=root / "data/raw/data/chengdu/stations.json")
    parser.add_argument("--output", type=Path, default=root / "artifacts/spatial_directional_benchmark")
    parser.add_argument("--splits", type=_at_least_three, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    city_labeled = pd.read_parquet(args.city_data)
    station_hourly = pd.read_parquet(args.station_data)
    coordinates = json.loads(args.coordinates.read_text(encoding="utf-8"))
    features = training_rows(enrich_city_with_spatial_features(city_labeled, station_hourly, coordinates))
    enhanced_features = model_feature_names(features)
    baseline_features = [name for name in enhanced_features if not _is_directional_feature(name)]
    report, predictions = benchmark_spatial_features(
        features,
        baseline_features=baseline_features,
        enhanced_features=enhanced_features,
        n_splits=args.splits,
        seed=args.seed,
    )
    write_spatial_benchmark_artifacts(report, predictions, args.output)
    print(
        json.dumps(
            {
                "output_dir": str(args.output),
                "training_rows": len(features),
                "baseline_feature_count": len(baseline_features),
                "directional_feature_count": len(enhanced_features) - len(baseline_features),
                "decision": report["decision"]["decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
