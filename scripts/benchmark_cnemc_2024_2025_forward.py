"""Evaluate whether the newly extended 2024-2025 data reduces forward MAE.

The protocol does not leak a recovery event into its own training partition.
It first selects a recency-weighting rule on 2024 only, then confirms that
single predeclared rule on 2025 after training on events that ended before
2025.  This avoids presenting a lowest-of-many 2025 score as an external MAE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.features import add_missingness_indicators
from haze_forecast.forward_evaluation import recency_sample_weights
from haze_forecast.gfs_external_validation import event_equal_mae
from haze_forecast.temporal_validation import event_isolated_year_split
from haze_forecast.training import AFTSurvivalRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
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
    "external_2024_2025": DATA / "cnemc_chengdu_2024_2025_training_features.parquet",
}
CANDIDATES: dict[str, float | None] = {
    "unweighted": None,
    "recency_half_life_3_years": 3.0,
    "recency_half_life_5_years": 5.0,
    "recency_half_life_8_years": 8.0,
}


def prepare(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="raise")
    prepared["event_key"] = source + ":" + prepared["event_id"].astype(str)
    prepared = add_missingness_indicators(prepared, PBL_COLUMNS)
    change_flag = "boundary_layer_height_m_change_3_is_missing"
    if change_flag not in prepared.columns:
        prepared[change_flag] = (
            prepared["boundary_layer_height_m"].isna()
            | prepared["boundary_layer_height_m_lag_3"].isna()
        ).astype(float)
    return prepared


def score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
    half_life_years: float | None,
    test_year: int,
) -> dict[str, float]:
    weights = None
    if half_life_years is not None:
        reference = pd.Timestamp(f"{test_year}-01-01", tz=train["timestamp"].dt.tz)
        weights = recency_sample_weights(train["timestamp"], reference, half_life_years)
    model = AFTSurvivalRegressor(seed=42, nthread=1).fit(
        train[feature_names], train["remaining_hours"], sample_weight=weights
    )
    predicted = np.asarray(model.predict(test[feature_names]), dtype=float).clip(min=0.0)
    actual = test["remaining_hours"].to_numpy(dtype=float)
    return {
        "row_mae": float(np.abs(predicted - actual).mean()),
        "event_equal_mae": event_equal_mae(actual, predicted, test["event_key"]),
        "prediction_mean": float(predicted.mean()),
        "actual_mean": float(actual.mean()),
    }


def main() -> None:
    with (ARTIFACTS / "model_metadata.json").open(encoding="utf-8") as handle:
        feature_names = json.load(handle)["feature_names"]
    if len(feature_names) != 87:
        raise ValueError(f"expected the deployed 87 features, got {len(feature_names)}")

    frames = [prepare(pd.read_parquet(path), source) for source, path in SOURCES.items()]
    all_rows = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    missing_features = sorted(set(feature_names).difference(all_rows.columns))
    if missing_features:
        raise ValueError(f"extended data is missing model features: {missing_features}")

    result: dict[str, object] = {
        "protocol": (
            "event-isolated forward validation; 2024 selects one recency rule, then 2025 confirms it; "
            "features exactly match the deployed 87-feature AFT model"
        ),
        "data": {
            "rows": len(all_rows),
            "event_count": int(all_rows["event_key"].nunique()),
            "year_counts": {
                str(year): int(count) for year, count in all_rows["timestamp"].dt.year.value_counts().sort_index().items()
            },
        },
        "selection_year_2024": {},
    }
    train_2024, test_2024 = event_isolated_year_split(all_rows, 2024, event_column="event_key")
    for name, half_life in CANDIDATES.items():
        metrics = score(train_2024, test_2024, feature_names, half_life, 2024)
        result["selection_year_2024"][name] = {"half_life_years": half_life, **metrics}
        print(f"2024 {name}: row MAE {metrics['row_mae']:.3f}, event MAE {metrics['event_equal_mae']:.3f}")

    selected_name = min(
        result["selection_year_2024"],
        key=lambda name: result["selection_year_2024"][name]["row_mae"],
    )
    selected_half_life = CANDIDATES[selected_name]
    train_2025, test_2025 = event_isolated_year_split(all_rows, 2025, event_column="event_key")
    confirmation = score(train_2025, test_2025, feature_names, selected_half_life, 2025)
    baseline_2025 = score(train_2025, test_2025, feature_names, None, 2025)
    result["confirmation_year_2025"] = {
        "predeclared_from_2024": selected_name,
        "selected": {"half_life_years": selected_half_life, **confirmation},
        "unweighted_baseline": {"half_life_years": None, **baseline_2025},
        "relative_row_mae_improvement": 1.0 - confirmation["row_mae"] / baseline_2025["row_mae"],
        "relative_event_equal_mae_improvement": 1.0
        - confirmation["event_equal_mae"] / baseline_2025["event_equal_mae"],
    }
    result["partitions"] = {
        "2024": {"train_rows": len(train_2024), "test_rows": len(test_2024), "test_events": int(test_2024["event_key"].nunique())},
        "2025": {"train_rows": len(train_2025), "test_rows": len(test_2025), "test_events": int(test_2025["event_key"].nunique())},
    }
    output = ARTIFACTS / "cnemc_2024_2025_forward" / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"2025 confirmed {selected_name}: row MAE {confirmation['row_mae']:.3f} vs {baseline_2025['row_mae']:.3f}; " f"event MAE {confirmation['event_equal_mae']:.3f} vs {baseline_2025['event_equal_mae']:.3f}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
