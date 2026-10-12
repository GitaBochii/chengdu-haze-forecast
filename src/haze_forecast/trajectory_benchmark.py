"""Benchmark process-trajectory features without replacing the production model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.evaluation import regression_metrics
from haze_forecast.features import TRAJECTORY_FEATURE_NAMES, model_feature_names
from haze_forecast.model_benchmark import decide_candidate, fit_predict_candidate
from haze_forecast.storage import write_json_atomic
from haze_forecast.training import rolling_event_splits


def _validate_frame(frame: pd.DataFrame, n_splits: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    required = {"timestamp", "event_id", "remaining_hours", *TRAJECTORY_FEATURE_NAMES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"trajectory benchmark missing columns: {sorted(missing)}")
    if n_splits < 3:
        raise ValueError("trajectory benchmark requires at least three splits")
    prepared = frame.sort_values("timestamp").reset_index(drop=True).copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="raise")
    baseline_features = model_feature_names(prepared)
    enhanced_features = [*baseline_features, *TRAJECTORY_FEATURE_NAMES]
    if not baseline_features:
        raise ValueError("trajectory benchmark requires baseline numeric features")
    return prepared, baseline_features, enhanced_features


def _mae(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual.to_numpy(dtype=float) - predicted)))


def benchmark_trajectory_features(
    frame: pd.DataFrame, n_splits: int = 3, seed: int = 42
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare the current AFT inputs with exactly the added trajectory inputs."""
    prepared, baseline_features, enhanced_features = _validate_frame(frame, n_splits)
    reports: list[dict[str, object]] = []
    records: list[pd.DataFrame] = []
    actuals: dict[str, list[pd.Series]] = {"baseline": [], "with_trajectory": []}
    predictions: dict[str, list[np.ndarray]] = {"baseline": [], "with_trajectory": []}
    fold_mae: dict[str, list[float]] = {"baseline": [], "with_trajectory": []}
    for fold, (train, valid) in enumerate(rolling_event_splits(prepared, n_splits=n_splits), start=1):
        if train["timestamp"].max() >= valid["timestamp"].min():
            raise ValueError("trajectory training time must precede validation time")
        actual = valid["remaining_hours"].reset_index(drop=True)
        fold_values: dict[str, np.ndarray] = {}
        for variant, feature_names in (
            ("baseline", baseline_features),
            ("with_trajectory", enhanced_features),
        ):
            predicted = np.asarray(
                fit_predict_candidate(
                    "xgboost_aft",
                    train[feature_names],
                    train["remaining_hours"],
                    valid[feature_names],
                    seed=seed + fold,
                ),
                dtype=float,
            )
            fold_values[variant] = predicted
            actuals[variant].append(actual)
            predictions[variant].append(predicted)
            fold_mae[variant].append(_mae(actual, predicted))
            records.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "variant": variant,
                        "timestamp": valid["timestamp"].to_numpy(),
                        "event_id": valid["event_id"].to_numpy(),
                        "actual_remaining_hours": actual.to_numpy(),
                        "predicted_remaining_hours": predicted,
                    }
                )
            )
        reports.append(
            {
                "fold": fold,
                "train_event_ids": train["event_id"].drop_duplicates().tolist(),
                "validation_event_ids": valid["event_id"].drop_duplicates().tolist(),
                "train_end_timestamp": train["timestamp"].max().isoformat(),
                "validation_start_timestamp": valid["timestamp"].min().isoformat(),
                "baseline_mae": fold_mae["baseline"][-1],
                "trajectory_mae": fold_mae["with_trajectory"][-1],
            }
        )
    metrics = {
        variant: regression_metrics(
            pd.concat(actuals[variant], ignore_index=True), np.concatenate(predictions[variant])
        )
        for variant in actuals
    }
    report = {
        "protocol": "expanding chronological complete-event rolling validation",
        "trajectory_feature_names": list(TRAJECTORY_FEATURE_NAMES),
        "baseline_feature_count": len(baseline_features),
        "enhanced_feature_count": len(enhanced_features),
        "folds": reports,
        "variants": {
            variant: {"metrics": metrics[variant], "fold_mae": fold_mae[variant]}
            for variant in ("baseline", "with_trajectory")
        },
        "decision": decide_candidate(
            fold_mae["baseline"],
            fold_mae["with_trajectory"],
            baseline_overall_mae=float(metrics["baseline"]["mae"]),
            candidate_overall_mae=float(metrics["with_trajectory"]["mae"]),
        ),
    }
    return report, pd.concat(records, ignore_index=True)


def write_trajectory_artifacts(
    report: dict[str, object], predictions: pd.DataFrame, output_dir: Path
) -> None:
    """Atomically persist isolated trajectory-candidate artifacts."""
    required = {
        "fold",
        "variant",
        "timestamp",
        "event_id",
        "actual_remaining_hours",
        "predicted_remaining_hours",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"trajectory predictions missing columns: {sorted(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_path = output_dir / "fold_predictions.csv.part"
    try:
        write_json_atomic(output_dir / "report.json", report)
        predictions.to_csv(staging_path, index=False, encoding="utf-8-sig")
        staging_path.replace(output_dir / "fold_predictions.csv")
    finally:
        staging_path.unlink(missing_ok=True)

