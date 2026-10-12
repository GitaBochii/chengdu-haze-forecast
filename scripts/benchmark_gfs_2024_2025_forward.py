"""Confirm the preselected issued-GFS +6h candidate on 2024 and 2025.

The +6h field set was chosen before this script from the 2022-2023 external
benchmark.  This script therefore makes no parameter/model selection: 2024
and 2025 are both clean forward confirmations on newly acquired observations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.gfs_external_validation import (
    GFS_F006_FEATURES,
    GFS_OBSERVATION_FEATURES,
    event_equal_mae,
    validate_gfs_forecast_timing,
)
from haze_forecast.training import AFTSurvivalRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "artifacts" / "gfs_2024_2025_forward" / "report.json"
SOURCES = {
    2021: DATA / "gfs_event_features_f006_2021_all.parquet",
    2022: DATA / "gfs_event_features_f006_2022_all.parquet",
    2023: DATA / "gfs_event_features_f006_2023_all.parquet",
    2024: DATA / "gfs_event_features_f006_2024_all.parquet",
    2025: DATA / "gfs_event_features_f006_2025_all.parquet",
}


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
    frames: list[pd.DataFrame] = []
    for year, path in SOURCES.items():
        frame = pd.read_parquet(path).copy()
        validate_gfs_forecast_timing(frame)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frame["event_key"] = f"{year}:" + frame["event_id"].astype(str)
        frame["source_year"] = year
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    observation_features = list(GFS_OBSERVATION_FEATURES)
    gfs_features = [*observation_features, *GFS_F006_FEATURES]
    report: dict[str, object] = {
        "protocol": (
            "strict forward confirmation on 2024 and 2025; the issued +6h GFS field set was "
            "preselected from 2022-2023; every row verifies run <= timestamp < forecast valid time"
        ),
        "years": {},
    }
    for year in (2024, 2025):
        train = all_rows.loc[all_rows["source_year"].lt(year)].copy()
        test = all_rows.loc[all_rows["source_year"].eq(year)].copy()
        observation_only = score(train, test, observation_features)
        with_gfs = score(train, test, gfs_features)
        row_improvement = 1.0 - with_gfs["row_mae"] / observation_only["row_mae"]
        event_improvement = 1.0 - with_gfs["event_equal_mae"] / observation_only["event_equal_mae"]
        report["years"][str(year)] = {
            "train_rows": len(train),
            "test_rows": len(test),
            "test_events": int(test["event_key"].nunique()),
            "observation_only": observation_only,
            "plus_issued_gfs_f006": with_gfs,
            "relative_row_mae_improvement": row_improvement,
            "relative_event_equal_mae_improvement": event_improvement,
        }
        print(
            f"{year}: row MAE {observation_only['row_mae']:.3f} -> {with_gfs['row_mae']:.3f} "
            f"({row_improvement:+.2%}); event MAE {observation_only['event_equal_mae']:.3f} -> "
            f"{with_gfs['event_equal_mae']:.3f} ({event_improvement:+.2%})"
        )
    both_improve = all(
        values["relative_row_mae_improvement"] > 0
        and values["relative_event_equal_mae_improvement"] > 0
        for values in report["years"].values()
    )
    report["decision"] = (
        "candidate_for_f024_multi_horizon_confirmation"
        if both_improve
        else "f006_not_consistent_enough_for_default_promotion"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"decision: {report['decision']}")
    print(f"report: {OUTPUT}")


if __name__ == "__main__":
    main()
