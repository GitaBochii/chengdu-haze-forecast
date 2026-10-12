"""Train an isolated 2014-history candidate without replacing default artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.evaluation import regression_metrics
from haze_forecast.features import (
    add_missingness_indicators,
    model_feature_names,
    snapshot_feature_names,
)
from haze_forecast.storage import dump_model_atomic, write_json_atomic
from haze_forecast.training import AFTSurvivalRegressor, event_chronological_split


def _validate_inputs(base: pd.DataFrame, prefix: pd.DataFrame) -> None:
    required = {"timestamp", "event_id", "remaining_hours"}
    missing_base = required - set(base.columns)
    missing_prefix = required - set(prefix.columns)
    if missing_base:
        raise ValueError(f"base data missing columns: {sorted(missing_base)}")
    if missing_prefix:
        raise ValueError(f"historical prefix missing columns: {sorted(missing_prefix)}")
    if base.empty or prefix.empty:
        raise ValueError("base data and historical prefix must both be non-empty")
    if set(base["event_id"]).intersection(set(prefix["event_id"])):
        raise ValueError("historical prefix event identifiers must not overlap base events")
    base_times = pd.to_datetime(base["timestamp"], errors="raise")
    prefix_times = pd.to_datetime(prefix["timestamp"], errors="raise")
    if prefix_times.max() >= base_times.min():
        raise ValueError("historical prefix timestamps must strictly precede the base data")


def _fit_predict(
    train: pd.DataFrame, valid: pd.DataFrame, feature_names: list[str], seed: int
) -> np.ndarray:
    model = AFTSurvivalRegressor(seed=seed, nthread=1).fit(
        train[feature_names], train["remaining_hours"]
    )
    return np.asarray(model.predict(valid[feature_names]), dtype=float).clip(min=0.0)


def build_cnemc_2014_candidate(
    base: pd.DataFrame,
    prefix: pd.DataFrame,
    artifacts_dir: Path,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate 2014 history on the unchanged base holdout, then train final artifacts."""
    _validate_inputs(base, prefix)
    pbl_feature_columns = [
        name for name in base.columns if name.startswith("boundary_layer_height_m")
    ]
    if pbl_feature_columns:
        base = add_missingness_indicators(base, pbl_feature_columns)
        prefix = add_missingness_indicators(prefix, pbl_feature_columns)
    history_features = model_feature_names(base)
    snapshot_features = snapshot_feature_names(base)
    if not history_features or not snapshot_features:
        raise ValueError("base data does not expose required history and snapshot features")
    missing_features = set(history_features) - set(prefix.columns)
    if missing_features:
        raise ValueError(f"historical prefix missing base feature columns: {sorted(missing_features)}")

    train, valid, test = event_chronological_split(base)
    pretest_base = pd.concat([train, valid], ignore_index=True)
    candidate_pretest = pd.concat([prefix, pretest_base], ignore_index=True).sort_values("timestamp")
    if candidate_pretest["timestamp"].max() >= test["timestamp"].min():
        raise ValueError("candidate training time must precede fixed base test time")

    history_test_prediction = _fit_predict(
        candidate_pretest, test, history_features, seed=seed
    )
    snapshot_test_prediction = _fit_predict(
        candidate_pretest, test, snapshot_features, seed=seed
    )
    final_training = pd.concat([prefix, base], ignore_index=True).sort_values("timestamp")
    history_model = AFTSurvivalRegressor(seed=seed, nthread=1).fit(
        final_training[history_features], final_training["remaining_hours"]
    )
    snapshot_model = AFTSurvivalRegressor(seed=seed, nthread=1).fit(
        final_training[snapshot_features], final_training["remaining_hours"]
    )

    metrics = {
        "xgboost_aft": regression_metrics(test["remaining_hours"], history_test_prediction),
        "xgboost_aft_snapshot": regression_metrics(
            test["remaining_hours"], snapshot_test_prediction
        ),
    }
    metadata: dict[str, object] = {
        "model": "XGBoostAFT",
        "aft_model": "XGBoostAFT",
        "aft_model_filename": "aft_model.joblib",
        "aft_snapshot_model_filename": "static_aft_model.joblib",
        "selected_history_model": "XGBoostAFT",
        "selected_history_model_filename": "aft_model.joblib",
        "selected_history_validation_mae": float(metrics["xgboost_aft"]["mae"]),
        "selected_snapshot_model": "XGBoostAFT",
        "selected_snapshot_model_filename": "static_aft_model.joblib",
        "selected_snapshot_validation_mae": float(metrics["xgboost_aft_snapshot"]["mae"]),
        "feature_names": history_features,
        "snapshot_feature_names": snapshot_features,
        "seed": seed,
        "evaluation_protocol": "fixed 2015-2021-style base holdout with 2014 prefix",
        "evaluation_test_event_ids": test["event_id"].drop_duplicates().tolist(),
        "evaluation_rows": len(test),
        "final_training_rows": len(final_training),
        "historical_prefix_rows": len(prefix),
    }
    predictions = pd.DataFrame(
        {
            "timestamp": test["timestamp"],
            "event_id": test["event_id"],
            "actual_remaining_hours": test["remaining_hours"],
            "predicted_remaining_hours": history_test_prediction,
            "snapshot_predicted_remaining_hours": snapshot_test_prediction,
        }
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    dump_model_atomic(artifacts_dir / "aft_model.joblib", history_model)
    dump_model_atomic(artifacts_dir / "static_aft_model.joblib", snapshot_model)
    predictions.to_csv(artifacts_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    write_json_atomic(artifacts_dir / "metrics.json", metrics)
    write_json_atomic(artifacts_dir / "model_metadata.json", metadata)
    return {"metrics": metrics, "metadata": metadata}
