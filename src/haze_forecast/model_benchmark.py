"""Decision rules for comparing candidate models against the current baseline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from haze_forecast.evaluation import regression_metrics
from haze_forecast.features import NON_FEATURE_COLUMNS
from haze_forecast.storage import write_json_atomic
from haze_forecast.training import AFTSurvivalRegressor, rolling_event_splits

CANDIDATE_MODELS = (
    "xgboost_aft",
    "xgboost_absolute_error",
    "lightgbm_l1",
    "catboost_mae",
)

_LIGHTGBM_L1_PARAMS: dict[str, float | int] = {
    "learning_rate": 0.07915176709957256,
    "num_leaves": 11,
    "min_child_samples": 60,
    "subsample": 0.996836719725523,
    "colsample_bytree": 0.6363225815386635,
    "reg_alpha": 0.5927260996668168,
    "reg_lambda": 0.0018582693681392772,
}


def build_candidate_model(
    model_name: str, seed: int
) -> AFTSurvivalRegressor | xgb.XGBRegressor | LGBMRegressor | CatBoostRegressor:
    """Build one candidate backend with its reviewed, fixed hyperparameters."""
    if model_name == "xgboost_aft":
        return AFTSurvivalRegressor(num_boost_round=500, seed=seed, nthread=1)
    if model_name == "xgboost_absolute_error":
        return xgb.XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=8,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=3,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    if model_name == "lightgbm_l1":
        return LGBMRegressor(
            objective="regression_l1",
            random_state=seed,
            n_estimators=13,
            n_jobs=1,
            verbosity=-1,
            **_LIGHTGBM_L1_PARAMS,
        )
    if model_name == "catboost_mae":
        return CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=500,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=8,
            random_strength=1,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
    raise ValueError(f"unsupported model: {model_name}")


def _validate_candidate_inputs(
    train_x: object, train_y: object, valid_x: object
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if not isinstance(train_x, pd.DataFrame) or not isinstance(valid_x, pd.DataFrame):
        raise ValueError("candidate inputs must use non-empty pandas DataFrames")  # noqa: TRY004
    if not isinstance(train_y, pd.Series):
        raise ValueError(  # noqa: TRY004
            "candidate inputs must use a one-dimensional pandas Series target"
        )
    if train_x.empty or valid_x.empty:
        raise ValueError("candidate inputs must use non-empty feature DataFrames")
    if train_x.columns.has_duplicates or valid_x.columns.has_duplicates:
        raise ValueError("candidate inputs must not contain duplicate feature columns")
    if not train_x.columns.equals(valid_x.columns):
        raise ValueError("candidate inputs must have matching feature columns in the same order")
    if len(train_x) != len(train_y):
        raise ValueError("candidate inputs must have matching training feature and target rows")
    if not train_x.index.equals(train_y.index):
        raise ValueError("candidate inputs must have matching training feature and target indexes")

    try:
        train_x = train_x.astype(float)
        train_y = train_y.astype(float)
        valid_x = valid_x.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("candidate inputs must contain float-convertible values") from error
    if np.isinf(train_x.to_numpy()).any() or np.isinf(valid_x.to_numpy()).any():
        raise ValueError("candidate inputs must not contain infinite feature values")
    if not np.isfinite(train_y.to_numpy()).all():
        raise ValueError("candidate inputs must contain finite training targets")
    return train_x, train_y, valid_x


def fit_predict_candidate(
    model_name: str,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    seed: int = 42,
) -> np.ndarray:
    """Fit one fixed candidate model and return non-negative predictions."""
    train_x, train_y, valid_x = _validate_candidate_inputs(train_x, train_y, valid_x)
    model = build_candidate_model(model_name, seed)
    model.fit(train_x, train_y)
    try:
        predictions = np.asarray(model.predict(valid_x), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("candidate predictions must be numeric") from error
    if (
        predictions.ndim != 1
        or len(predictions) != len(valid_x)
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("candidate predictions must be finite and match validation rows")
    return predictions.clip(min=0.0)


def decide_candidate(
    baseline_fold_mae: Sequence[float],
    candidate_fold_mae: Sequence[float],
    baseline_overall_mae: float | None = None,
    candidate_overall_mae: float | None = None,
) -> dict[str, float | int | str | None]:
    """Return a fold-stable replacement decision, optionally weighted by all validation rows."""
    if len(baseline_fold_mae) != len(candidate_fold_mae) or len(baseline_fold_mae) < 3:
        raise ValueError(
            "baseline and candidate MAE values must have the same length of at least 3"
        )

    baseline = np.asarray(baseline_fold_mae, dtype=float)
    candidate = np.asarray(candidate_fold_mae, dtype=float)
    if baseline.ndim != 1 or candidate.ndim != 1:
        raise ValueError("fold MAE values must be one-dimensional")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("fold MAE values must be finite")
    if (baseline <= 0).any():
        raise ValueError("baseline fold MAE values must be greater than zero")
    if (candidate < 0).any():
        raise ValueError("candidate fold MAE values must be non-negative")

    baseline_fold_mean = float(baseline.mean())
    candidate_fold_mean = float(candidate.mean())
    if (baseline_overall_mae is None) != (candidate_overall_mae is None):
        raise ValueError("baseline and candidate overall MAE values must be provided together")
    if baseline_overall_mae is None:
        comparison_baseline_mae = baseline_fold_mean
        comparison_candidate_mae = candidate_fold_mean
    else:
        try:
            comparison_baseline_mae = float(baseline_overall_mae)
            comparison_candidate_mae = float(candidate_overall_mae)
        except (TypeError, ValueError) as error:
            raise ValueError("overall MAE values must be numeric scalars") from error
        if not np.isfinite(comparison_baseline_mae) or not np.isfinite(comparison_candidate_mae):
            raise ValueError("overall MAE values must be finite")
        if comparison_baseline_mae <= 0:
            raise ValueError("baseline overall MAE must be greater than zero")
        if comparison_candidate_mae < 0:
            raise ValueError("candidate overall MAE must be non-negative")

    relative_improvement = float(1 - comparison_candidate_mae / comparison_baseline_mae)
    relative_changes = candidate / baseline - 1
    improved_folds = int((candidate < baseline).sum())
    worst_fold_relative_change = float(relative_changes.max())
    meets_improvement_gate = relative_improvement >= 0.03 or np.isclose(
        relative_improvement, 0.03, rtol=0.0, atol=1e-12
    )
    meets_stability_gate = worst_fold_relative_change <= 0.05 or np.isclose(
        worst_fold_relative_change, 0.05, rtol=0.0, atol=1e-12
    )
    decision = (
        "candidate_for_replacement_review"
        if meets_improvement_gate and improved_folds >= 2 and meets_stability_gate
        else "reject_unstable_or_insufficient_mae_improvement"
    )
    return {
        "decision": decision,
        "baseline_mean_mae": baseline_fold_mean,
        "candidate_mean_mae": candidate_fold_mean,
        "baseline_fold_mean_mae": baseline_fold_mean,
        "candidate_fold_mean_mae": candidate_fold_mean,
        "baseline_overall_mae": baseline_overall_mae,
        "candidate_overall_mae": candidate_overall_mae,
        "relative_mae_improvement": relative_improvement,
        "improved_folds": improved_folds,
        "worst_fold_relative_change": worst_fold_relative_change,
    }


def _validate_benchmark_inputs(
    frame: object, feature_names: Sequence[str], n_splits: int
) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("benchmark frame must be a pandas DataFrame")  # noqa: TRY004
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 3:
        raise ValueError("benchmark n_splits must be an integer of at least 3")
    if isinstance(feature_names, str) or not feature_names:
        raise ValueError("benchmark feature_names must be non-empty")

    normalized_features = list(feature_names)
    if any(not isinstance(name, str) for name in normalized_features):
        raise ValueError("benchmark feature_names must contain only column names")
    if len(set(normalized_features)) != len(normalized_features):
        raise ValueError("benchmark feature_names must not contain duplicates")
    forbidden_features = set(normalized_features).intersection(NON_FEATURE_COLUMNS)
    if forbidden_features:
        raise ValueError(f"benchmark feature_names include non-feature columns: {sorted(forbidden_features)}")

    required = {"timestamp", "event_id", "remaining_hours"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns for benchmark: {sorted(missing)}")
    missing_features = set(normalized_features) - set(frame.columns)
    if missing_features:
        raise ValueError(f"missing benchmark feature columns: {sorted(missing_features)}")

    prepared = frame.copy()
    try:
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="raise")
        prepared["remaining_hours"] = pd.to_numeric(
            prepared["remaining_hours"], errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("benchmark timestamps and targets must be valid") from error
    if prepared["timestamp"].isna().any():
        raise ValueError("benchmark timestamps must be finite")
    target_values = prepared["remaining_hours"].to_numpy(dtype=float)
    if not np.isfinite(target_values).all() or (target_values <= 0).any():
        raise ValueError("benchmark remaining_hours labels must be positive and finite")
    return prepared, normalized_features


def _mae_for_fold(actual: pd.Series, predicted: np.ndarray) -> float:
    """Calculate one fold's MAE without grouped DataFrame callbacks."""
    if len(actual) != len(predicted) or not len(actual):
        raise ValueError("benchmark predictions must match non-empty validation rows")
    absolute_error = 0.0
    for observed, estimated in zip(actual.to_numpy(dtype=float), predicted, strict=True):
        absolute_error += abs(float(observed) - float(estimated))
    return absolute_error / len(actual)


def _json_safe_report_value(value: object) -> object:
    """Normalize audit metadata without changing the prediction table's source values."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def benchmark_history_models(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    n_splits: int = 3,
    seed: int = 42,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Benchmark all history models using identical expanding complete-event folds."""
    prepared, normalized_features = _validate_benchmark_inputs(frame, feature_names, n_splits)
    folds = rolling_event_splits(prepared, n_splits=n_splits)
    fold_reports: list[dict[str, object]] = []
    prediction_records: list[pd.DataFrame] = []
    model_actuals: dict[str, list[pd.Series]] = {name: [] for name in CANDIDATE_MODELS}
    model_predictions: dict[str, list[np.ndarray]] = {name: [] for name in CANDIDATE_MODELS}
    model_fold_mae: dict[str, list[float]] = {name: [] for name in CANDIDATE_MODELS}

    for fold_number, (train, valid) in enumerate(folds, start=1):
        train_event_ids = [
            _json_safe_report_value(value) for value in train["event_id"].drop_duplicates()
        ]
        validation_event_ids = [
            _json_safe_report_value(value) for value in valid["event_id"].drop_duplicates()
        ]
        if set(train_event_ids).intersection(validation_event_ids):
            raise ValueError("benchmark training and validation events must be disjoint")
        train_end_timestamp = train["timestamp"].max()
        validation_start_timestamp = valid["timestamp"].min()
        if train_end_timestamp >= validation_start_timestamp:
            raise ValueError("benchmark training time must precede validation time")
        fold_reports.append(
            {
                "fold": fold_number,
                "train_event_ids": train_event_ids,
                "validation_event_ids": validation_event_ids,
                "train_end_timestamp": train_end_timestamp.isoformat(),
                "validation_start_timestamp": validation_start_timestamp.isoformat(),
            }
        )

        train_x = train[normalized_features]
        train_y = train["remaining_hours"]
        valid_x = valid[normalized_features]
        actual = valid["remaining_hours"].reset_index(drop=True)
        for model_name in CANDIDATE_MODELS:
            predicted = np.asarray(
                fit_predict_candidate(model_name, train_x, train_y, valid_x, seed=seed + fold_number),
                dtype=float,
            )
            fold_mae = _mae_for_fold(actual, predicted)
            model_actuals[model_name].append(actual)
            model_predictions[model_name].append(predicted)
            model_fold_mae[model_name].append(fold_mae)
            prediction_records.append(
                pd.DataFrame(
                    {
                        "fold": fold_number,
                        "model": model_name,
                        "timestamp": valid["timestamp"].to_numpy(),
                        "event_id": valid["event_id"].to_numpy(),
                        "actual_remaining_hours": actual.to_numpy(),
                        "predicted_remaining_hours": predicted,
                    }
                )
            )

    model_reports: dict[str, dict[str, object]] = {}
    for model_name in CANDIDATE_MODELS:
        actual = pd.concat(model_actuals[model_name], ignore_index=True)
        predicted = np.concatenate(model_predictions[model_name])
        model_reports[model_name] = {
            "metrics": regression_metrics(actual, predicted),
            "fold_mae": model_fold_mae[model_name],
        }

    aft_fold_mae = model_fold_mae["xgboost_aft"]
    for model_name in CANDIDATE_MODELS:
        if model_name != "xgboost_aft":
            aft_metrics = model_reports["xgboost_aft"]["metrics"]
            candidate_metrics = model_reports[model_name]["metrics"]
            model_reports[model_name]["decision_against_xgboost_aft"] = decide_candidate(
                aft_fold_mae,
                model_fold_mae[model_name],
                baseline_overall_mae=float(aft_metrics["mae"]),
                candidate_overall_mae=float(candidate_metrics["mae"]),
            )

    report = {
        "protocol": "expanding chronological complete-event rolling validation",
        "folds": fold_reports,
        "models": model_reports,
    }
    return report, pd.concat(prediction_records, ignore_index=True)


def benchmark_prefix_data_effect(
    base_frame: pd.DataFrame,
    historical_prefix: pd.DataFrame,
    feature_names: Sequence[str],
    n_splits: int = 3,
    seed: int = 42,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Measure whether older labeled data helps on unchanged future-event folds.

    The prefix is deliberately restricted to timestamps before the first baseline
    validation window. This keeps every target and validation row identical
    between variants, so a MAE change can be attributed to the added history.
    """
    base, normalized_features = _validate_benchmark_inputs(
        base_frame, feature_names, n_splits
    )
    prefix, prefix_features = _validate_benchmark_inputs(
        historical_prefix, feature_names, n_splits
    )
    if prefix_features != normalized_features:
        raise ValueError("historical prefix feature names must match the baseline")
    if set(base["event_id"]).intersection(set(prefix["event_id"])):
        raise ValueError("historical prefix event identifiers must not overlap baseline events")

    folds = rolling_event_splits(base, n_splits=n_splits)
    first_validation_timestamp = min(valid["timestamp"].min() for _, valid in folds)
    if prefix["timestamp"].max() >= first_validation_timestamp:
        raise ValueError(
            "historical prefix timestamps must strictly precede every validation window"
        )

    fold_reports: list[dict[str, object]] = []
    prediction_records: list[pd.DataFrame] = []
    baseline_actuals: list[pd.Series] = []
    baseline_predictions: list[np.ndarray] = []
    candidate_actuals: list[pd.Series] = []
    candidate_predictions: list[np.ndarray] = []
    baseline_fold_mae: list[float] = []
    candidate_fold_mae: list[float] = []

    for fold_number, (base_train, valid) in enumerate(folds, start=1):
        candidate_train = pd.concat([prefix, base_train], ignore_index=True).sort_values(
            "timestamp"
        )
        if candidate_train["timestamp"].max() >= valid["timestamp"].min():
            raise ValueError("candidate training time must precede validation time")

        actual = valid["remaining_hours"].reset_index(drop=True)
        baseline_prediction = np.asarray(
            fit_predict_candidate(
                "xgboost_aft",
                base_train[normalized_features],
                base_train["remaining_hours"],
                valid[normalized_features],
                seed=seed + fold_number,
            ),
            dtype=float,
        )
        candidate_prediction = np.asarray(
            fit_predict_candidate(
                "xgboost_aft",
                candidate_train[normalized_features],
                candidate_train["remaining_hours"],
                valid[normalized_features],
                seed=seed + fold_number,
            ),
            dtype=float,
        )
        baseline_mae = _mae_for_fold(actual, baseline_prediction)
        candidate_mae = _mae_for_fold(actual, candidate_prediction)
        baseline_actuals.append(actual)
        baseline_predictions.append(baseline_prediction)
        candidate_actuals.append(actual)
        candidate_predictions.append(candidate_prediction)
        baseline_fold_mae.append(baseline_mae)
        candidate_fold_mae.append(candidate_mae)

        fold_reports.append(
            {
                "fold": fold_number,
                "baseline_train_rows": len(base_train),
                "candidate_train_rows": len(candidate_train),
                "historical_prefix_rows": len(prefix),
                "validation_rows": len(valid),
                "validation_event_ids": [
                    _json_safe_report_value(value)
                    for value in valid["event_id"].drop_duplicates()
                ],
                "validation_start_timestamp": valid["timestamp"].min().isoformat(),
                "baseline_fold_mae": baseline_mae,
                "candidate_fold_mae": candidate_mae,
            }
        )
        for variant, prediction in (
            ("baseline", baseline_prediction),
            ("with_historical_prefix", candidate_prediction),
        ):
            prediction_records.append(
                pd.DataFrame(
                    {
                        "fold": fold_number,
                        "variant": variant,
                        "model": "xgboost_aft",
                        "timestamp": valid["timestamp"].to_numpy(),
                        "event_id": valid["event_id"].to_numpy(),
                        "actual_remaining_hours": actual.to_numpy(),
                        "predicted_remaining_hours": prediction,
                    }
                )
            )

    baseline_metrics = regression_metrics(
        pd.concat(baseline_actuals, ignore_index=True), np.concatenate(baseline_predictions)
    )
    candidate_metrics = regression_metrics(
        pd.concat(candidate_actuals, ignore_index=True), np.concatenate(candidate_predictions)
    )
    report = {
        "protocol": "fixed future-event validation with optional historical prefix data",
        "model": "xgboost_aft",
        "folds": fold_reports,
        "baseline": {"metrics": baseline_metrics, "fold_mae": baseline_fold_mae},
        "with_historical_prefix": {
            "metrics": candidate_metrics,
            "fold_mae": candidate_fold_mae,
            "decision_against_baseline": decide_candidate(
                baseline_fold_mae,
                candidate_fold_mae,
                baseline_overall_mae=float(baseline_metrics["mae"]),
                candidate_overall_mae=float(candidate_metrics["mae"]),
            ),
        },
    }
    return report, pd.concat(prediction_records, ignore_index=True)


def write_benchmark_artifacts(
    report: dict[str, object], predictions: pd.DataFrame, output_dir: Path
) -> None:
    """Publish matching report and fold-prediction artifacts, rolling back on failure."""
    required_columns = {
        "fold",
        "model",
        "event_id",
        "actual_remaining_hours",
        "predicted_remaining_hours",
    }
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("benchmark predictions must be a pandas DataFrame")  # noqa: TRY004
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"benchmark predictions missing columns: {sorted(missing_columns)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    predictions_path = output_dir / "fold_predictions.csv"
    temporary_report = output_dir / "report.json.part"
    temporary_report_staging = output_dir / "report.json.part.part"
    temporary_predictions = output_dir / "fold_predictions.csv.part"
    report_backup = output_dir / "report.json.backup"
    predictions_backup = output_dir / "fold_predictions.csv.backup"
    report_backed_up = False
    predictions_backed_up = False
    report_published = False
    predictions_published = False

    try:
        write_json_atomic(temporary_report, report)
        predictions.to_csv(temporary_predictions, index=False, encoding="utf-8-sig")

        if report_path.exists():
            report_path.replace(report_backup)
            report_backed_up = True
        if predictions_path.exists():
            predictions_path.replace(predictions_backup)
            predictions_backed_up = True

        temporary_predictions.replace(predictions_path)
        predictions_published = True
        temporary_report.replace(report_path)
        report_published = True
    except OSError:
        if report_published:
            report_path.unlink(missing_ok=True)
        if predictions_published:
            predictions_path.unlink(missing_ok=True)
        if report_backed_up:
            report_backup.replace(report_path)
        if predictions_backed_up:
            predictions_backup.replace(predictions_path)
        raise
    finally:
        temporary_report.unlink(missing_ok=True)
        temporary_report_staging.unlink(missing_ok=True)
        temporary_predictions.unlink(missing_ok=True)
        report_backup.unlink(missing_ok=True)
        predictions_backup.unlink(missing_ok=True)
