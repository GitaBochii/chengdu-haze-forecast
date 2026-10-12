"""Test whether Open-Meteo archive gap filling improves later-year backtests.

This is intentionally an offline-data-quality candidate: the archive fills
weather observations at the prediction timestamp, but is not promoted as a
future operational forecast source.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.features import add_missingness_indicators
from haze_forecast.gfs_external_validation import event_equal_mae
from haze_forecast.temporal_validation import event_isolated_year_split
from haze_forecast.training import AFTSurvivalRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "open_meteo_fill_2024_2025_forward" / "report.json"
PBL_COLUMNS = [
    "boundary_layer_height_m",
    "boundary_layer_height_m_lag_1",
    "boundary_layer_height_m_lag_3",
    "boundary_layer_height_m_lag_24",
    "boundary_layer_height_m_roll_3",
    "boundary_layer_height_m_roll_24",
    "boundary_layer_height_m_change_3",
]


def prepare(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    result["event_key"] = source + ":" + result["event_id"].astype(str)
    result = add_missingness_indicators(result, PBL_COLUMNS)
    result["boundary_layer_height_m_change_3_is_missing"] = (
        result["boundary_layer_height_m"].isna() | result["boundary_layer_height_m_lag_3"].isna()
    ).astype(float)
    return result


def score(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict[str, float]:
    model = AFTSurvivalRegressor(seed=42, nthread=1).fit(train[features], train["remaining_hours"])
    predicted = np.asarray(model.predict(test[features]), dtype=float).clip(min=0.0)
    actual = test["remaining_hours"].to_numpy(dtype=float)
    return {
        "row_mae": float(np.abs(predicted - actual).mean()),
        "event_equal_mae": event_equal_mae(actual, predicted, test["event_key"]),
        "prediction_mean": float(predicted.mean()),
        "actual_mean": float(actual.mean()),
    }


def main() -> None:
    features = json.loads((ARTIFACTS / "model_metadata.json").read_text(encoding="utf-8"))["feature_names"]
    shared = [
        prepare(pd.read_parquet(DATA / "cnemc_chengdu_2014_training_features.parquet"), "prefix_2014"),
        prepare(pd.read_parquet(DATA / "training_features.parquet"), "base"),
        prepare(pd.read_parquet(DATA / "cnemc_chengdu_2022_training_features.parquet"), "external_2022"),
        prepare(pd.read_parquet(DATA / "cnemc_chengdu_2023_training_features.parquet"), "external_2023"),
    ]
    original = {
        2024: prepare(pd.read_parquet(DATA / "cnemc_chengdu_2024_training_features.parquet"), "external_2024"),
        2025: prepare(pd.read_parquet(DATA / "cnemc_chengdu_2025_training_features.parquet"), "external_2025"),
    }
    filled_all = pd.read_parquet(DATA / "cnemc_chengdu_2024_2025_open_meteo_fill_training_features.parquet")
    filled = {
        year: prepare(filled_all.loc[pd.to_datetime(filled_all["timestamp"]).dt.year.eq(year)], f"external_{year}")
        for year in (2024, 2025)
    }
    report: dict[str, object] = {
        "protocol": (
            "event-isolated annual forward test; NOAA observations are retained where present and Open-Meteo archive "
            "fills only missing weather fields; candidate remains offline-only"
        ),
        "years": {},
    }
    for year in (2024, 2025):
        original_rows = pd.concat([*shared, *(original[past] for past in (2024, 2025))], ignore_index=True)
        filled_rows = pd.concat([*shared, *(filled[past] for past in (2024, 2025))], ignore_index=True)
        original_train, original_test = event_isolated_year_split(original_rows, year, event_column="event_key")
        filled_train, filled_test = event_isolated_year_split(filled_rows, year, event_column="event_key")
        original_metrics = score(original_train, original_test, features)
        filled_metrics = score(filled_train, filled_test, features)
        row_improvement = 1.0 - filled_metrics["row_mae"] / original_metrics["row_mae"]
        event_improvement = 1.0 - filled_metrics["event_equal_mae"] / original_metrics["event_equal_mae"]
        report["years"][str(year)] = {
            "original": original_metrics,
            "open_meteo_gap_fill": filled_metrics,
            "relative_row_mae_improvement": row_improvement,
            "relative_event_equal_mae_improvement": event_improvement,
            "test_rows": len(filled_test),
            "test_events": int(filled_test["event_key"].nunique()),
        }
        print(
            f"{year}: row MAE {original_metrics['row_mae']:.3f} -> {filled_metrics['row_mae']:.3f} "
            f"({row_improvement:+.2%}); event MAE {original_metrics['event_equal_mae']:.3f} -> "
            f"{filled_metrics['event_equal_mae']:.3f} ({event_improvement:+.2%})"
        )
    accepted = all(
        values["relative_row_mae_improvement"] > 0
        and values["relative_event_equal_mae_improvement"] > 0
        for values in report["years"].values()
    )
    report["decision"] = (
        "candidate_for_observation_imputation_review"
        if accepted
        else "reject_open_meteo_gap_fill_as_inconsistent"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"decision: {report['decision']}")
    print(f"report: {OUTPUT}")


if __name__ == "__main__":
    main()
