"""Materialize the complete observed-clock business population without training."""

import json
from pathlib import Path

import pandas as pd

from haze_forecast.features import MEASUREMENT_COLUMNS
from haze_forecast.observed_business import build_observed_business
from scripts.train_recovery_weather_formal import sha

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "2014": "data/interim/cnemc_chengdu_2014_weather.parquet",
    "2015_2021": "data/processed/city_labeled.parquet",
    "2022": "data/interim/cnemc_chengdu_2022_weather.parquet",
    "2023": "data/interim/cnemc_chengdu_2023_weather.parquet",
    "2024_2025": "data/interim/cnemc_chengdu_2024_2025_weather.parquet",
}


def main():
    output = ROOT / "artifacts/task01_recovery_head/observed_business_dataset_v1"
    if output.exists():
        raise FileExistsError(output)
    metadata = ROOT / "artifacts/cnemc_2025_deployment_candidate/model_metadata.json"
    names = json.loads(metadata.read_bytes())["feature_names"]
    if len(names) != 87 or len(set(names)) != 87:
        raise ValueError("expected 87 distinct deployed features")
    paths = [ROOT / p for p in SOURCES.values()] + [metadata]
    paths += [
        ROOT / p
        for p in (
            "src/haze_forecast/observed_business.py",
            "src/haze_forecast/observed_hourly_features.py",
            "src/haze_forecast/observed_recovery_labels.py",
            "src/haze_forecast/features.py",
            "src/haze_forecast/labels.py",
            "scripts/build_observed_business_dataset.py",
        )
    ]
    deployment = [
        ROOT / "app.py",
        *sorted((ROOT / "artifacts/cnemc_2025_deployment_candidate").glob("*")),
    ]
    deployment = [p for p in deployment if p.is_file()]
    hashes = {str(p): sha(p) for p in paths}
    guards = {str(p): sha(p) for p in deployment}
    sources = {key: pd.read_parquet(ROOT / path) for key, path in SOURCES.items()}
    for key, frame in sources.items():
        missing = set(MEASUREMENT_COLUMNS) - set(frame)
        if missing:
            raise ValueError(f"{key} missing measurement columns: {sorted(missing)}")
    full, origins = build_observed_business(sources, names)
    if not full.timestamp.is_unique or not origins.timestamp.is_unique:
        raise ValueError("nonunique timestamps")
    if not origins.label_available_at.ge(origins.timestamp).all():
        raise ValueError("invalid label availability")
    if not origins.lower_bound_hours.gt(0).all():
        raise ValueError("invalid label lower bounds")
    yearly = {}
    for year, grid in full.groupby(full.timestamp.dt.year):
        selected = origins.loc[origins.timestamp.dt.year.eq(year)]
        yearly[str(year)] = {
            "grid_hours": len(grid),
            "input_observed_rows": int(grid.input_hour_observed.sum()),
            "pollution_states": {
                str(k): int(v) for k, v in grid.pollution_state.value_counts().items()
            },
            "eligible_origins": len(selected),
            "episodes": int(selected.episode_id.nunique()),
            "continuity_blocks": int(selected.continuity_block.nunique()),
            "exact_origins": int(selected.exact_hours.notna().sum()),
            "censored_origins": int(selected.exact_hours.isna().sum()),
            "informative_origins": int(selected.informative.sum()),
            "censor_reasons": {str(k): int(v) for k, v in selected.reason.value_counts().items()},
            "measurement_missing_fraction": {
                c: float(grid[c].isna().mean()) for c in MEASUREMENT_COLUMNS
            },
        }
    if hashes != {str(p): sha(p) for p in paths} or guards != {str(p): sha(p) for p in deployment}:
        raise RuntimeError("input or deployment changed during construction")
    output.mkdir(parents=True)
    full.to_parquet(output / "features.parquet", index=False)
    origins.to_parquet(output / "origins.parquet", index=False)
    report = {
        "status": "complete_dataset_no_training",
        "input_sha256": hashes,
        "deployment_sha256": guards,
        "deployment_unchanged": True,
        "feature_names": names,
        "grid_rows": len(full),
        "origin_rows": len(origins),
        "yearly": yearly,
        "sources": {
            k: {
                "path": SOURCES[k],
                "rows": len(f),
                "first_timestamp": str(f.timestamp.min()),
                "last_timestamp": str(f.timestamp.max()),
            }
            for k, f in sources.items()
        },
        "schemas": {
            "features": {c: str(t) for c, t in full.dtypes.items()},
            "origins": {c: str(t) for c, t in origins.dtypes.items()},
        },
        "output_sha256": {
            name: sha(output / name) for name in ("features.parquet", "origins.parquet")
        },
        "protocol": {
            "event_start_hours": 3,
            "stable_recovery_hours": 6,
            "first_two_polluted_hours_not_backfilled": True,
            "aqi_gap_ends_episode": True,
            "aqi_gap_does_not_end_continuity_block": True,
            "source_boundary_does_not_reset": True,
            "origins_selected_without_future_labels": True,
            "measurements_imputed_here": False,
        },
        "limitations": [
            "observed_episode_is_not_proven_physical_event",
            "continuity_blocks_must_control_event_fold_purging",
            "labels_need_availability_cutoff_at_training",
            "exact_metrics_exclude_censored_outcomes",
            "source_o3_8h_vs_o3_semantics_differ",
            "external_raw_measurement_imputation_not_proven_absent",
            "87_feature_schema_preserved_for_comparability_not_harmonized_measurement_claim",
            "no_accuracy_or_deployment_claim",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "grid_rows": len(full),
                "origin_rows": len(origins),
                "yearly": yearly,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
