"""Strict rolling comparison for the local station-network directional features."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.evaluation import regression_metrics
from haze_forecast.features import NON_FEATURE_COLUMNS
from haze_forecast.storage import write_json_atomic
from haze_forecast.training import AFTSurvivalRegressor, rolling_event_splits


def _validate_feature_names(
    frame: pd.DataFrame, feature_names: Sequence[str], label: str
) -> list[str]:
    if isinstance(feature_names, str) or not feature_names:
        raise ValueError(f"{label} features must be non-empty")
    normalized = list(feature_names)
    if len(normalized) != len(set(normalized)) or any(not isinstance(name, str) for name in normalized):
        raise ValueError(f"{label} features must be unique column names")
    forbidden = set(normalized).intersection(NON_FEATURE_COLUMNS)
    if forbidden:
        raise ValueError(f"{label} features include non-feature columns: {sorted(forbidden)}")
    missing = set(normalized) - set(frame.columns)
    if missing:
        raise ValueError(f"missing {label} feature columns: {sorted(missing)}")
    return normalized


def _validate_benchmark_inputs(
    frame: object,
    baseline_features: Sequence[str],
    enhanced_features: Sequence[str],
    n_splits: int,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("spatial benchmark frame must be a pandas DataFrame")  # noqa: TRY004
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 3:
        raise ValueError("spatial benchmark n_splits must be an integer of at least 3")
    required = {"timestamp", "event_id", "remaining_hours"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns for spatial benchmark: {sorted(missing)}")
    baseline = _validate_feature_names(frame, baseline_features, "baseline")
    enhanced = _validate_feature_names(frame, enhanced_features, "enhanced")
    directional = [name for name in enhanced if name not in baseline]
    if not directional:
        raise ValueError("enhanced features must add at least one directional feature")
    if any(name not in enhanced for name in baseline):
        raise ValueError("enhanced features must include every baseline feature")

    prepared = frame.copy()
    try:
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="raise")
        prepared["remaining_hours"] = pd.to_numeric(prepared["remaining_hours"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("spatial benchmark timestamps and targets must be valid") from error
    target = prepared["remaining_hours"].to_numpy(dtype=float)
    if prepared["timestamp"].isna().any() or not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("spatial benchmark labels must be positive and finite")
    return prepared, baseline, enhanced, directional


def _fit_predict_aft(
    train_x: pd.DataFrame, train_y: pd.Series, valid_x: pd.DataFrame, seed: int
) -> np.ndarray:
    model = AFTSurvivalRegressor(num_boost_round=500, seed=seed, nthread=1)
    model.fit(train_x, train_y)
    predicted = np.asarray(model.predict(valid_x), dtype=float)
    if predicted.ndim != 1 or len(predicted) != len(valid_x) or not np.isfinite(predicted).all():
        raise ValueError("spatial AFT predictions must be finite and match validation rows")
    return predicted.clip(min=0.0)


def _mae(actual: pd.Series, predicted: np.ndarray) -> float:
    if len(actual) != len(predicted) or not len(actual):
        raise ValueError("spatial benchmark predictions must match non-empty validation rows")
    return float(np.mean(np.abs(actual.to_numpy(dtype=float) - predicted)))


def _report_value(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def decide_spatial_candidate(
    baseline_fold_mae: Sequence[float],
    candidate_fold_mae: Sequence[float],
    baseline_overall_mae: float,
    candidate_overall_mae: float,
    directional_coverage: float,
) -> dict[str, float | int | str | None]:
    """Apply the existing accuracy gates plus a minimum usable-direction coverage gate."""
    try:
        coverage = float(directional_coverage)
    except (TypeError, ValueError) as error:
        raise ValueError("directional coverage must be a finite ratio between zero and one") from error
    if not np.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError("directional coverage must be a finite ratio between zero and one")

    baseline = np.asarray(baseline_fold_mae, dtype=float)
    candidate = np.asarray(candidate_fold_mae, dtype=float)
    if (
        baseline.ndim != 1
        or candidate.ndim != 1
        or len(baseline) < 3
        or len(baseline) != len(candidate)
        or not np.isfinite(baseline).all()
        or not np.isfinite(candidate).all()
        or (baseline <= 0).any()
        or (candidate < 0).any()
    ):
        raise ValueError("fold MAE values must be finite, aligned, and contain at least three folds")
    try:
        baseline_overall = float(baseline_overall_mae)
        candidate_overall = float(candidate_overall_mae)
    except (TypeError, ValueError) as error:
        raise ValueError("overall MAE values must be numeric scalars") from error
    if (
        not np.isfinite(baseline_overall)
        or not np.isfinite(candidate_overall)
        or baseline_overall <= 0
        or candidate_overall < 0
    ):
        raise ValueError("overall MAE values must be finite and non-negative")
    relative_improvement = float(1 - candidate_overall / baseline_overall)
    relative_changes = candidate / baseline - 1
    improved_folds = int((candidate < baseline).sum())
    worst_fold_relative_change = float(relative_changes.max())
    accuracy_gate = relative_improvement >= 0.03 or np.isclose(
        relative_improvement, 0.03, rtol=0.0, atol=1e-12
    )
    stability_gate = worst_fold_relative_change <= 0.05 or np.isclose(
        worst_fold_relative_change, 0.05, rtol=0.0, atol=1e-12
    )
    decision: dict[str, float | int | str | None] = {
        "decision": (
            "candidate_for_replacement_review"
            if accuracy_gate and improved_folds >= 2 and stability_gate
            else "reject_unstable_or_insufficient_mae_improvement"
        ),
        "baseline_fold_mean_mae": float(baseline.mean()),
        "candidate_fold_mean_mae": float(candidate.mean()),
        "baseline_overall_mae": baseline_overall,
        "candidate_overall_mae": candidate_overall,
        "relative_mae_improvement": relative_improvement,
        "improved_folds": improved_folds,
        "worst_fold_relative_change": worst_fold_relative_change,
        "directional_coverage": coverage,
    }
    if coverage < 0.70:
        decision["decision"] = "reject_insufficient_directional_coverage"
    return decision


def benchmark_spatial_features(
    frame: pd.DataFrame,
    baseline_features: Sequence[str],
    enhanced_features: Sequence[str],
    n_splits: int = 3,
    seed: int = 42,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare baseline and directional AFT inputs on the same complete-event folds."""
    prepared, baseline, enhanced, directional = _validate_benchmark_inputs(
        frame, baseline_features, enhanced_features, n_splits
    )
    folds = rolling_event_splits(prepared, n_splits=n_splits)
    coverage = float(prepared[directional].notna().all(axis=1).mean())
    fold_reports: list[dict[str, object]] = []
    fold_mae: dict[str, list[float]] = {"baseline": [], "directional": []}
    actuals: dict[str, list[pd.Series]] = {"baseline": [], "directional": []}
    predictions: dict[str, list[np.ndarray]] = {"baseline": [], "directional": []}
    records: list[pd.DataFrame] = []

    for fold_number, (train, valid) in enumerate(folds, start=1):
        train_events = [_report_value(value) for value in train["event_id"].drop_duplicates()]
        validation_events = [_report_value(value) for value in valid["event_id"].drop_duplicates()]
        if set(train_events).intersection(validation_events):
            raise ValueError("spatial benchmark training and validation events must be disjoint")
        if train["timestamp"].max() >= valid["timestamp"].min():
            raise ValueError("spatial benchmark training time must precede validation time")
        fold_reports.append(
            {
                "fold": fold_number,
                "train_event_ids": train_events,
                "validation_event_ids": validation_events,
                "train_end_timestamp": train["timestamp"].max().isoformat(),
                "validation_start_timestamp": valid["timestamp"].min().isoformat(),
            }
        )
        actual = valid["remaining_hours"].reset_index(drop=True)
        for variant, feature_names in (("baseline", baseline), ("directional", enhanced)):
            predicted = _fit_predict_aft(
                train[feature_names], train["remaining_hours"], valid[feature_names], seed + fold_number
            )
            actuals[variant].append(actual)
            predictions[variant].append(predicted)
            fold_mae[variant].append(_mae(actual, predicted))
            records.append(
                pd.DataFrame(
                    {
                        "fold": fold_number,
                        "variant": variant,
                        "timestamp": valid["timestamp"].to_numpy(),
                        "event_id": valid["event_id"].to_numpy(),
                        "actual_remaining_hours": actual.to_numpy(),
                        "predicted_remaining_hours": predicted,
                    }
                )
            )

    metrics = {
        variant: regression_metrics(
            pd.concat(actuals[variant], ignore_index=True), np.concatenate(predictions[variant])
        )
        for variant in ("baseline", "directional")
    }
    report = {
        "protocol": "expanding chronological complete-event rolling validation",
        "directional_features": directional,
        "directional_feature_coverage": coverage,
        "folds": fold_reports,
        "models": {
            variant: {"metrics": metrics[variant], "fold_mae": fold_mae[variant]}
            for variant in ("baseline", "directional")
        },
        "decision": decide_spatial_candidate(
            fold_mae["baseline"],
            fold_mae["directional"],
            float(metrics["baseline"]["mae"]),
            float(metrics["directional"]["mae"]),
            coverage,
        ),
    }
    return report, pd.concat(records, ignore_index=True)


def write_spatial_benchmark_artifacts(
    report: dict[str, object], predictions: pd.DataFrame, output_dir: Path
) -> None:
    """Write paired, inspectable spatial benchmark artifacts."""
    required = {"fold", "variant", "event_id", "actual_remaining_hours", "predicted_remaining_hours"}
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("spatial benchmark predictions must be a pandas DataFrame")  # noqa: TRY004
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"spatial benchmark predictions missing columns: {sorted(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    prediction_path = output_dir / "fold_predictions.csv"
    staging_path = output_dir / "fold_predictions.csv.part"
    try:
        write_json_atomic(report_path, report)
        predictions.to_csv(staging_path, index=False, encoding="utf-8-sig")
        staging_path.replace(prediction_path)
    finally:
        staging_path.unlink(missing_ok=True)
