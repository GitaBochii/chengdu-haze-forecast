"""Test whether issued GFS +6h weather lowers external-year dissipation MAE.

The benchmark deliberately has no tuning phase: it compares the same AFT model
with current observations only versus the same inputs plus the forecast fields.
2022 and 2023 are strictly forward tests, and each input row is rejected unless
its GFS run existed at the prediction timestamp.
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
OUTPUT = ROOT / "artifacts" / "gfs_external_forward" / "report.json"
SOURCES = {
    2021: DATA / "gfs_event_features_f006_2021_all.parquet",
    2022: DATA / "gfs_event_features_f006_2022_all.parquet",
    2023: DATA / "gfs_event_features_f006_2023_all.parquet",
}


def score(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict[str, float]:
    model = AFTSurvivalRegressor(seed=42, nthread=1).fit(train[features], train["remaining_hours"])
    prediction = np.asarray(model.predict(test[features]), dtype=float).clip(min=0.0)
    actual = test["remaining_hours"].to_numpy(dtype=float)
    return {
        "row_mae": float(np.abs(prediction - actual).mean()),
        "event_equal_mae": event_equal_mae(actual, prediction, test["event_key"]),
        "prediction_mean": float(prediction.mean()),
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
    forecast_features = [*observation_features, *GFS_F006_FEATURES]
    result: dict[str, object] = {
        "protocol": (
            "strict external-year forward test; all GFS runs must satisfy run <= timestamp < valid; "
            "no parameter selection or default-model promotion"
        ),
        "features": {"observation_only": observation_features, "plus_gfs_f006": forecast_features},
        "years": {},
    }
    for year in (2022, 2023):
        train = all_rows.loc[all_rows["source_year"].lt(year)].copy()
        test = all_rows.loc[all_rows["source_year"].eq(year)].copy()
        baseline = score(train, test, observation_features)
        with_gfs = score(train, test, forecast_features)
        row_change = 1.0 - with_gfs["row_mae"] / baseline["row_mae"]
        event_change = 1.0 - with_gfs["event_equal_mae"] / baseline["event_equal_mae"]
        result["years"][str(year)] = {
            "train_rows": len(train),
            "test_rows": len(test),
            "test_events": int(test["event_key"].nunique()),
            "observation_only": baseline,
            "plus_gfs_f006": with_gfs,
            "relative_row_mae_improvement": row_change,
            "relative_event_equal_mae_improvement": event_change,
        }
        print(
            f"{year}: row MAE {baseline['row_mae']:.3f} -> {with_gfs['row_mae']:.3f} "
            f"({row_change:+.2%}); event MAE {baseline['event_equal_mae']:.3f} -> "
            f"{with_gfs['event_equal_mae']:.3f} ({event_change:+.2%})"
        )
    accepted = all(
        values["relative_row_mae_improvement"] > 0
        and values["relative_event_equal_mae_improvement"] > 0
        for values in result["years"].values()
    )
    result["decision"] = (
        "candidate_for_multi_horizon_validation" if accepted else "reject_f006_only_as_insufficient"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结论: {result['decision']}")
    print(f"报告: {OUTPUT}")


if __name__ == "__main__":
    main()
