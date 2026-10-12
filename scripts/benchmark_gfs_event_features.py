"""Compare current observations with and without issued GFS features by held-out event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

from haze_forecast.features import MEASUREMENT_COLUMNS
from haze_forecast.training import constrain_event_predictions


def build_model() -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        iterations=300,
        learning_rate=0.04,
        depth=4,
        l2_leaf_reg=10,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/gfs_event_features_f006_2021.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/gfs_feature_benchmark_2021.json")
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input).sort_values("timestamp").reset_index(drop=True)
    gfs_features = [
        column for column in frame.columns if column.startswith("gfs_") and column.endswith("_f006")
    ]
    if not gfs_features:
        raise ValueError("input does not contain GFS +6 hour feature columns")
    feature_sets = {
        "observations_only": list(MEASUREMENT_COLUMNS),
        "observations_plus_gfs_f006": [*MEASUREMENT_COLUMNS, *gfs_features],
    }
    records: list[dict[str, float | int | str]] = []
    for event_id in sorted(frame["event_id"].unique()):
        train = frame.loc[frame["event_id"].ne(event_id)]
        test = frame.loc[frame["event_id"].eq(event_id)]
        for model_name, feature_names in feature_sets.items():
            model = build_model().fit(train[feature_names], train["remaining_hours"])
            raw_prediction = pd.Series(model.predict(test[feature_names]), index=test.index)
            prediction = constrain_event_predictions(test, raw_prediction)
            records.append(
                {
                    "event_id": int(event_id),
                    "model": model_name,
                    "rows": len(test),
                    "mae": float(mean_absolute_error(test["remaining_hours"], prediction)),
                }
            )
    comparison = pd.DataFrame(records)
    overall = (
        comparison.assign(weighted_error=comparison["mae"] * comparison["rows"])
        .groupby("model", as_index=False)
        .agg(rows=("rows", "sum"), weighted_error=("weighted_error", "sum"))
    )
    overall["mae"] = overall["weighted_error"] / overall["rows"]
    baseline_mae = float(overall.loc[overall["model"].eq("observations_only"), "mae"].iloc[0])
    gfs_mae = float(overall.loc[overall["model"].eq("observations_plus_gfs_f006"), "mae"].iloc[0])
    payload = {
        "protocol": "leave-one-complete-event-out; all GFS cycles must be issued no later than the event timestamp",
        "rows": len(frame),
        "events": int(frame["event_id"].nunique()),
        "gfs_feature_names": gfs_features,
        "overall": overall.to_dict(orient="records"),
        "per_event": comparison.to_dict(orient="records"),
        "mae_change_gfs_minus_baseline": gfs_mae - baseline_mae,
        "decision": "reject_gfs_for_main_model" if gfs_mae >= baseline_mae else "candidate_for_more_data",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["overall"], ensure_ascii=False))
    print(f"决策: {payload['decision']}")


if __name__ == "__main__":
    main()
