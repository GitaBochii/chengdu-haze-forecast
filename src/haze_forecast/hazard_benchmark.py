"""Auditable rolling comparison of discrete hazards against the current AFT baseline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.discrete_survival import DiscreteHazardRegressor
from haze_forecast.evaluation import regression_metrics
from haze_forecast.features import NON_FEATURE_COLUMNS
from haze_forecast.model_benchmark import decide_candidate
from haze_forecast.storage import write_json_atomic
from haze_forecast.training import AFTSurvivalRegressor, rolling_event_splits


def _validate_inputs(
    frame: object, feature_names: Sequence[str], n_splits: int
) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("hazard benchmark frame must be a pandas DataFrame")  # noqa: TRY004
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 3:
        raise ValueError("hazard benchmark n_splits must be an integer of at least 3")
    if isinstance(feature_names, str) or not feature_names:
        raise ValueError("hazard benchmark feature_names must be non-empty")
    names = list(feature_names)
    if len(names) != len(set(names)) or any(not isinstance(name, str) for name in names):
        raise ValueError("hazard benchmark feature_names must be unique column names")
    forbidden = set(names).intersection(NON_FEATURE_COLUMNS)
    if forbidden:
        raise ValueError(f"hazard benchmark includes non-feature columns: {sorted(forbidden)}")
    required = {"timestamp", "event_id", "remaining_hours"}
    missing = required - set(frame.columns)
    missing_features = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"hazard benchmark missing required columns: {sorted(missing)}")
    if missing_features:
        raise ValueError(f"hazard benchmark missing feature columns: {sorted(missing_features)}")
    prepared = frame.copy()
    try:
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="raise")
        prepared["remaining_hours"] = pd.to_numeric(prepared["remaining_hours"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("hazard benchmark timestamps and targets must be valid") from error
    target = prepared["remaining_hours"].to_numpy(dtype=float)
    if prepared["timestamp"].isna().any() or not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("hazard benchmark labels must be positive and finite")
    if not np.allclose(target, np.rint(target), rtol=0.0, atol=1e-9):
        raise ValueError("hazard benchmark labels must use integer hourly values")
    return prepared, names


def _fit_predict_aft(
    train_x: pd.DataFrame, train_y: pd.Series, valid_x: pd.DataFrame, seed: int
) -> np.ndarray:
    model = AFTSurvivalRegressor(num_boost_round=500, seed=seed, nthread=1)
    model.fit(train_x, train_y)
    prediction = np.asarray(model.predict(valid_x), dtype=float)
    if prediction.ndim != 1 or len(prediction) != len(valid_x) or not np.isfinite(prediction).all():
        raise ValueError("AFT benchmark predictions must be finite and match validation rows")
    return prediction.clip(min=0.0)


def _fit_predict_hazard(
    train_x: pd.DataFrame, train_y: pd.Series, valid_x: pd.DataFrame, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    model = DiscreteHazardRegressor(seed=seed).fit(train_x, train_y)
    prediction, censored = model.predict_with_censoring(valid_x)
    if (
        prediction.ndim != 1
        or len(prediction) != len(valid_x)
        or censored.ndim != 1
        or len(censored) != len(valid_x)
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("hazard benchmark predictions must be finite and match validation rows")
    return prediction.clip(min=0.0), censored.astype(bool)


def _mae(actual: pd.Series, prediction: np.ndarray) -> float:
    if len(actual) != len(prediction) or not len(actual):
        raise ValueError("benchmark predictions must match non-empty validation rows")
    return float(np.mean(np.abs(actual.to_numpy(dtype=float) - prediction)))


def _report_value(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def benchmark_discrete_hazard(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    n_splits: int = 3,
    seed: int = 42,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Benchmark an AFT baseline and a current-snapshot discrete hazard model."""
    prepared, names = _validate_inputs(frame, feature_names, n_splits)
    folds = rolling_event_splits(prepared, n_splits=n_splits)
    fold_reports: list[dict[str, object]] = []
    fold_mae: dict[str, list[float]] = {"xgboost_aft": [], "discrete_hazard": []}
    actuals: dict[str, list[pd.Series]] = {"xgboost_aft": [], "discrete_hazard": []}
    predictions: dict[str, list[np.ndarray]] = {"xgboost_aft": [], "discrete_hazard": []}
    censoring: list[np.ndarray] = []
    records: list[pd.DataFrame] = []

    for fold_number, (train, valid) in enumerate(folds, start=1):
        train_events = [_report_value(value) for value in train["event_id"].drop_duplicates()]
        validation_events = [_report_value(value) for value in valid["event_id"].drop_duplicates()]
        if set(train_events).intersection(validation_events):
            raise ValueError("hazard benchmark training and validation events must be disjoint")
        if train["timestamp"].max() >= valid["timestamp"].min():
            raise ValueError("hazard benchmark training time must precede validation time")
        actual = valid["remaining_hours"].reset_index(drop=True)
        aft_prediction = _fit_predict_aft(
            train[names], train["remaining_hours"], valid[names], seed + fold_number
        )
        hazard_prediction, right_censored = _fit_predict_hazard(
            train[names], train["remaining_hours"], valid[names], seed + fold_number
        )
        fold_reports.append(
            {
                "fold": fold_number,
                "train_event_ids": train_events,
                "validation_event_ids": validation_events,
                "train_end_timestamp": train["timestamp"].max().isoformat(),
                "validation_start_timestamp": valid["timestamp"].min().isoformat(),
                "max_horizon_hours": int(train["remaining_hours"].max()),
                "right_censor_rate": float(right_censored.mean()),
            }
        )
        for model_name, prediction, model_censored in (
            ("xgboost_aft", aft_prediction, np.zeros(len(valid), dtype=bool)),
            ("discrete_hazard", hazard_prediction, right_censored),
        ):
            fold_mae[model_name].append(_mae(actual, prediction))
            actuals[model_name].append(actual)
            predictions[model_name].append(prediction)
            records.append(
                pd.DataFrame(
                    {
                        "fold": fold_number,
                        "model": model_name,
                        "timestamp": valid["timestamp"].to_numpy(),
                        "event_id": valid["event_id"].to_numpy(),
                        "actual_remaining_hours": actual.to_numpy(),
                        "predicted_remaining_hours": prediction,
                        "right_censored": model_censored,
                    }
                )
            )
        censoring.append(right_censored)

    metrics = {
        model_name: regression_metrics(
            pd.concat(actuals[model_name], ignore_index=True), np.concatenate(predictions[model_name])
        )
        for model_name in ("xgboost_aft", "discrete_hazard")
    }
    report = {
        "protocol": "expanding chronological complete-event rolling validation",
        "folds": fold_reports,
        "models": {
            "xgboost_aft": {"metrics": metrics["xgboost_aft"], "fold_mae": fold_mae["xgboost_aft"]},
            "discrete_hazard": {
                "metrics": metrics["discrete_hazard"],
                "fold_mae": fold_mae["discrete_hazard"],
                "right_censor_rate": float(np.concatenate(censoring).mean()),
            },
        },
        "decision": decide_candidate(
            fold_mae["xgboost_aft"],
            fold_mae["discrete_hazard"],
            baseline_overall_mae=float(metrics["xgboost_aft"]["mae"]),
            candidate_overall_mae=float(metrics["discrete_hazard"]["mae"]),
        ),
    }
    return report, pd.concat(records, ignore_index=True)


def write_hazard_benchmark_artifacts(
    report: dict[str, object], predictions: pd.DataFrame, output_dir: Path
) -> None:
    """Publish the report and row-level predictions separately from default artifacts."""
    required = {
        "fold",
        "model",
        "event_id",
        "actual_remaining_hours",
        "predicted_remaining_hours",
        "right_censored",
    }
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("hazard benchmark predictions must be a pandas DataFrame")  # noqa: TRY004
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"hazard benchmark predictions missing columns: {sorted(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_path = output_dir / "fold_predictions.csv.part"
    try:
        write_json_atomic(output_dir / "report.json", report)
        predictions.to_csv(staging_path, index=False, encoding="utf-8-sig")
        staging_path.replace(output_dir / "fold_predictions.csv")
    finally:
        staging_path.unlink(missing_ok=True)
