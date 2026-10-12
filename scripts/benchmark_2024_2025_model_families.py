"""Select a non-AFT model family on 2024, then confirm it once on 2025.

This is a model-family check, not an accuracy claim based on a shuffled split.
Every candidate uses the same 87 deployed-aligned features and the same
event-isolated forward partitions.  The lowest 2024 row MAE is selected before
the one-time 2025 confirmation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from haze_forecast.features import add_missingness_indicators
from haze_forecast.gfs_external_validation import event_equal_mae
from haze_forecast.temporal_validation import event_isolated_year_split
from haze_forecast.training import AFTSurvivalRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "model_family_2024_2025_forward" / "report.json"
PBL_COLUMNS = [
    "boundary_layer_height_m",
    "boundary_layer_height_m_lag_1",
    "boundary_layer_height_m_lag_3",
    "boundary_layer_height_m_lag_24",
    "boundary_layer_height_m_roll_3",
    "boundary_layer_height_m_roll_24",
    "boundary_layer_height_m_change_3",
]
SOURCES = {
    "prefix_2014": DATA / "cnemc_chengdu_2014_training_features.parquet",
    "base": DATA / "training_features.parquet",
    "external_2022": DATA / "cnemc_chengdu_2022_training_features.parquet",
    "external_2023": DATA / "cnemc_chengdu_2023_training_features.parquet",
    "external_2024": DATA / "cnemc_chengdu_2024_training_features.parquet",
    "external_2025": DATA / "cnemc_chengdu_2025_training_features.parquet",
}


def prepare(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    result["event_key"] = source + ":" + result["event_id"].astype(str)
    result = add_missingness_indicators(result, PBL_COLUMNS)
    result["boundary_layer_height_m_change_3_is_missing"] = (
        result["boundary_layer_height_m"].isna() | result["boundary_layer_height_m_lag_3"].isna()
    ).astype(float)
    return result


def build_model(name: str):
    if name == "aft_survival":
        return AFTSurvivalRegressor(seed=42, nthread=1)
    if name == "catboost_mae":
        return CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=500,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=8,
            random_strength=1,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
    if name == "lightgbm_l1":
        return LGBMRegressor(
            objective="regression_l1",
            n_estimators=700,
            learning_rate=0.04,
            num_leaves=24,
            min_child_samples=16,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
    raise ValueError(f"unknown model family: {name}")


def score(train: pd.DataFrame, test: pd.DataFrame, features: list[str], name: str) -> dict[str, float]:
    model = build_model(name)
    model.fit(train[features], train["remaining_hours"])
    prediction = np.asarray(model.predict(test[features]), dtype=float).clip(min=0.0)
    actual = test["remaining_hours"].to_numpy(dtype=float)
    return {
        "row_mae": float(np.abs(prediction - actual).mean()),
        "event_equal_mae": event_equal_mae(actual, prediction, test["event_key"]),
        "prediction_mean": float(prediction.mean()),
        "actual_mean": float(actual.mean()),
    }


def main() -> None:
    metadata = json.loads((ARTIFACTS / "model_metadata.json").read_text(encoding="utf-8"))
    features = list(metadata["feature_names"])
    frames = [prepare(pd.read_parquet(path), source) for source, path in SOURCES.items()]
    all_rows = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    missing = sorted(set(features).difference(all_rows.columns))
    if missing:
        raise ValueError(f"missing deployed-aligned features: {missing}")

    candidates = ("aft_survival", "catboost_mae", "lightgbm_l1")
    train_2024, test_2024 = event_isolated_year_split(all_rows, 2024, event_column="event_key")
    selection: dict[str, dict[str, float]] = {}
    for name in candidates:
        metrics = score(train_2024, test_2024, features, name)
        selection[name] = metrics
        print(f"2024 {name}: row MAE {metrics['row_mae']:.3f}, event MAE {metrics['event_equal_mae']:.3f}")

    selected = min(selection, key=lambda name: selection[name]["row_mae"])
    train_2025, test_2025 = event_isolated_year_split(all_rows, 2025, event_column="event_key")
    confirmation = score(train_2025, test_2025, features, selected)
    aft_2025 = score(train_2025, test_2025, features, "aft_survival")
    report = {
        "protocol": (
            "same 87 features; event-isolated forward partitions; select the model family using 2024 row MAE, "
            "then perform one 2025 confirmation after adding only completed pre-2025 events to training"
        ),
        "selection_year_2024": selection,
        "confirmation_year_2025": {
            "predeclared_from_2024": selected,
            "selected": confirmation,
            "aft_survival_baseline": aft_2025,
            "relative_row_mae_improvement": 1.0 - confirmation["row_mae"] / aft_2025["row_mae"],
            "relative_event_equal_mae_improvement": 1.0
            - confirmation["event_equal_mae"] / aft_2025["event_equal_mae"],
        },
        "partitions": {
            "2024": {"train_rows": len(train_2024), "test_rows": len(test_2024)},
            "2025": {"train_rows": len(train_2025), "test_rows": len(test_2025)},
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"2025 confirmed {selected}: row MAE {confirmation['row_mae']:.3f} vs AFT {aft_2025['row_mae']:.3f}; "
        f"event MAE {confirmation['event_equal_mae']:.3f} vs AFT {aft_2025['event_equal_mae']:.3f}"
    )
    print(f"report: {OUTPUT}")


if __name__ == "__main__":
    main()
