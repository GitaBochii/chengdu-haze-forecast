"""Fit an isolated full-history deployment candidate after external validation is frozen."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.storage import dump_model_atomic, write_json_atomic
from haze_forecast.training import AFTSurvivalRegressor


def _prepare_frames(frames: Mapping[str, pd.DataFrame], feature_names: list[str]) -> pd.DataFrame:
    if not frames:
        raise ValueError("at least one completed-history frame is required")

    prepared: list[pd.DataFrame] = []
    required = {"timestamp", "remaining_hours"}
    for source, frame in frames.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{source} missing required columns: {sorted(missing)}")
        source_frame = frame.copy()
        source_frame["timestamp"] = pd.to_datetime(source_frame["timestamp"], errors="raise")
        for indicator in (name for name in feature_names if name.endswith("_is_missing")):
            source_column = indicator.removesuffix("_is_missing")
            if source_column in source_frame.columns and indicator not in source_frame.columns:
                source_frame[indicator] = source_frame[source_column].isna().astype(float)
        prepared.append(source_frame)

    combined = pd.concat(prepared, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    missing_features = sorted(set(feature_names) - set(combined.columns))
    if missing_features:
        raise ValueError(f"missing deployment features: {missing_features}")
    return combined


def build_full_history_deployment_candidate(
    frames: Mapping[str, pd.DataFrame],
    artifacts_dir: Path,
    *,
    feature_names: list[str],
    snapshot_feature_names: list[str],
    external_mae: float,
    external_metric_source: str,
    seed: int = 42,
) -> dict[str, object]:
    """Refit on all completed history without relabeling its training error as accuracy."""
    if not np.isfinite(external_mae) or external_mae < 0:
        raise ValueError("external_mae must be finite and non-negative")
    if not feature_names or not snapshot_feature_names:
        raise ValueError("history and snapshot features must both be non-empty")

    training = _prepare_frames(frames, feature_names)
    missing_snapshot = sorted(set(snapshot_feature_names) - set(training.columns))
    if missing_snapshot:
        raise ValueError(f"missing deployment snapshot features: {missing_snapshot}")

    history_model = AFTSurvivalRegressor(seed=seed, nthread=1).fit(
        training[feature_names], training["remaining_hours"]
    )
    snapshot_model = AFTSurvivalRegressor(seed=seed, nthread=1).fit(
        training[snapshot_feature_names], training["remaining_hours"]
    )
    fitted_through = training["timestamp"].max().isoformat()
    metrics = {
        "xgboost_aft": {"mae": float(external_mae)},
        "xgboost_aft_snapshot": {"mae": None, "validation_status": "not_independently_validated"},
    }
    metadata: dict[str, object] = {
        "model": "XGBoostAFT",
        "aft_model": "XGBoostAFT",
        "aft_model_filename": "aft_model.joblib",
        "aft_snapshot_model_filename": "static_aft_model.joblib",
        "selected_history_model": "XGBoostAFT",
        "selected_history_model_filename": "aft_model.joblib",
        "selected_history_validation_mae": float(external_mae),
        "selected_snapshot_model": "XGBoostAFT",
        "selected_snapshot_model_filename": "static_aft_model.joblib",
        "selected_snapshot_validation_mae": None,
        "snapshot_validation_status": "not_independently_validated",
        "feature_names": feature_names,
        "snapshot_feature_names": snapshot_feature_names,
        "seed": seed,
        "deployment_role": "full_history_refit_for_future_predictions",
        "accuracy_claim": "external_validation_only",
        "external_metric_source": external_metric_source,
        "final_training_rows": len(training),
        "fitted_through": fitted_through,
        "source_rows": {source: len(frame) for source, frame in frames.items()},
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    dump_model_atomic(artifacts_dir / "aft_model.joblib", history_model)
    dump_model_atomic(artifacts_dir / "static_aft_model.joblib", snapshot_model)
    write_json_atomic(artifacts_dir / "metrics.json", metrics)
    write_json_atomic(artifacts_dir / "model_metadata.json", metadata)
    return {"metrics": metrics, "metadata": metadata, "training": training}
