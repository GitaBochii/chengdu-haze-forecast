"""Time-safe rolling benchmark for GEFSv12 multivariable event features."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.evaluation import regression_metrics
from haze_forecast.features import MEASUREMENT_COLUMNS, NON_FEATURE_COLUMNS, training_rows
from haze_forecast.gefs_dataset import GEFS_SURFACE_FEATURE_COLUMNS
from haze_forecast.storage import write_json_atomic
from haze_forecast.training import AFTSurvivalRegressor, rolling_event_splits

BASELINE_VARIANT = "baseline"
GEFS_VARIANT = "gefs_multivariable"
_REQUIRED_COLUMNS = {"timestamp", "event_id", "remaining_hours"}
_GEFS_TIME_COLUMNS = {"gefs_run_utc", "gefs_forecast_valid_utc"}
_REQUIRED_EVENT_COUNT = 15


def _as_utc_timestamp(series: pd.Series, label: str) -> pd.Series:
    try:
        values = pd.to_datetime(series, utc=True, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain valid timestamps") from error
    if values.isna().any():
        raise ValueError(f"{label} must not contain missing timestamps")
    return values


def _validate_time_safe_gefs_rows(frame: pd.DataFrame) -> pd.DataFrame:
    missing = ({"timestamp", "event_id"} | _GEFS_TIME_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"GEFS benchmark frame missing columns: {sorted(missing)}")
    result = frame.copy()
    result["timestamp"] = _as_utc_timestamp(result["timestamp"], "timestamp")
    result["gefs_run_utc"] = _as_utc_timestamp(result["gefs_run_utc"], "gefs_run_utc")
    result["gefs_forecast_valid_utc"] = _as_utc_timestamp(
        result["gefs_forecast_valid_utc"], "gefs_forecast_valid_utc"
    )
    safe = (
        result["gefs_run_utc"].le(result["timestamp"])
        & result["timestamp"].lt(result["gefs_forecast_valid_utc"])
        & result["gefs_forecast_valid_utc"].ge(result["timestamp"] + pd.Timedelta(6, unit="h"))
    )
    if not safe.all():
        raise ValueError("each GEFS row must satisfy run <= timestamp < valid and be valid at least 6 hours ahead")
    return result


def _gefs_feature_names(frame: pd.DataFrame) -> list[str]:
    missing = set(GEFS_SURFACE_FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            "GEFS benchmark requires the complete GEFS surface feature set: "
            f"missing {sorted(missing)}"
        )
    non_numeric = [
        name
        for name in GEFS_SURFACE_FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[name])
    ]
    if non_numeric:
        raise ValueError(f"GEFS surface feature columns must be numeric: {non_numeric}")
    return list(GEFS_SURFACE_FEATURE_COLUMNS)


def prepare_gefs_benchmark_frame(
    city_labeled: pd.DataFrame, gefs_pilot: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Build observation-only rows first, then time-safe join their GEFS pilot values."""
    if not isinstance(city_labeled, pd.DataFrame) or not isinstance(gefs_pilot, pd.DataFrame):
        raise TypeError("city_labeled and gefs_pilot must both be pandas DataFrames")
    missing_city = _REQUIRED_COLUMNS - set(city_labeled.columns)
    if missing_city:
        raise ValueError(f"city_labeled missing columns: {sorted(missing_city)}")
    missing_pilot = _REQUIRED_COLUMNS.intersection({"timestamp", "event_id"}) - set(gefs_pilot.columns)
    if missing_pilot:
        raise ValueError(f"GEFS pilot missing join columns: {sorted(missing_pilot)}")

    pilot = _validate_time_safe_gefs_rows(gefs_pilot)
    feature_names = _gefs_feature_names(pilot)
    if pilot.duplicated(["timestamp", "event_id"]).any():
        raise ValueError("GEFS pilot must have one row per timestamp and event_id")
    pilot_keys = pd.MultiIndex.from_frame(pilot[["timestamp", "event_id"]])
    observations = training_rows(city_labeled)
    observations["timestamp"] = _as_utc_timestamp(observations["timestamp"], "observation timestamp")
    observation_keys = pd.MultiIndex.from_frame(observations[["timestamp", "event_id"]])
    observations = observations.loc[observation_keys.isin(pilot_keys)].copy()
    selected_keys = pd.MultiIndex.from_frame(observations[["timestamp", "event_id"]])
    if selected_keys.has_duplicates:
        raise ValueError("matched observation-only training rows must have one row per timestamp and event_id")
    if len(selected_keys) != len(pilot_keys) or set(selected_keys) != set(pilot_keys):
        raise ValueError(
            "GEFS pilot must have a one-to-one match with observation-only training rows"
        )
    joined = observations.merge(
        pilot[["timestamp", "event_id", *_GEFS_TIME_COLUMNS, *feature_names]],
        on=["timestamp", "event_id"],
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError("no observation-only training rows matched the GEFS pilot")
    return _validate_time_safe_gefs_rows(joined), feature_names


def _validate_features(frame: pd.DataFrame, names: Sequence[str], label: str) -> list[str]:
    if isinstance(names, str) or not names:
        raise ValueError(f"{label} features must be a non-empty sequence")
    result = list(names)
    if len(result) != len(set(result)) or any(not isinstance(name, str) for name in result):
        raise ValueError(f"{label} features must be unique column names")
    forbidden = set(result).intersection(NON_FEATURE_COLUMNS)
    if forbidden:
        raise ValueError(f"{label} features include non-feature columns: {sorted(forbidden)}")
    missing = set(result) - set(frame.columns)
    if missing:
        raise ValueError(f"missing {label} feature columns: {sorted(missing)}")
    return result


def _validate_exact_event_count(frame: pd.DataFrame) -> None:
    event_ids = frame["event_id"].drop_duplicates()
    if len(event_ids) != _REQUIRED_EVENT_COUNT:
        raise ValueError("GEFS benchmark requires exactly 15 independent events")


def _measurement_baseline_features(frame: pd.DataFrame) -> list[str]:
    return [name for name in MEASUREMENT_COLUMNS if name in frame.columns]


def _fit_predict_aft(
    train_x: pd.DataFrame, train_y: pd.Series, valid_x: pd.DataFrame, seed: int
) -> np.ndarray:
    model = AFTSurvivalRegressor(num_boost_round=500, seed=seed, nthread=1)
    model.fit(train_x, train_y)
    predicted = np.asarray(model.predict(valid_x), dtype=float)
    if predicted.ndim != 1 or len(predicted) != len(valid_x) or not np.isfinite(predicted).all():
        raise ValueError("GEFS AFT predictions must be finite and match validation rows")
    return predicted.clip(min=0.0)


def _mae(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual.to_numpy(dtype=float) - predicted)))


def _event_mae(predictions: pd.DataFrame, variant: str) -> pd.DataFrame:
    selected = predictions.loc[predictions["variant"].eq(variant)].copy()
    selected["absolute_error"] = (
        selected["actual_remaining_hours"] - selected["predicted_remaining_hours"]
    ).abs()
    return (
        selected.groupby("event_id", as_index=False)
        .agg(rows=("event_id", "size"), mae=("absolute_error", "mean"), absolute_error=("absolute_error", "sum"))
        .sort_values("event_id")
        .reset_index(drop=True)
    )


def _validate_paired_per_event_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {"event_id", "mae", "absolute_error"}
    if not isinstance(frame, pd.DataFrame) or required - set(frame.columns):
        raise ValueError(f"{label} per-event errors must include {sorted(required)}")
    if frame.empty or frame["event_id"].isna().any() or frame["event_id"].duplicated().any():
        raise ValueError(
            "baseline and candidate per-event errors need unique event_id values and the same event_id set"
        )
    return frame


def decide_gefs_multivariable_candidate(
    baseline_mae: float, candidate_mae: float, baseline_per_event: pd.DataFrame, candidate_per_event: pd.DataFrame
) -> dict[str, object]:
    """Apply the predeclared accuracy, event-majority and concentration gates."""
    if baseline_mae < 0 or candidate_mae < 0 or not np.isfinite([baseline_mae, candidate_mae]).all():
        raise ValueError("overall MAE values must be finite and non-negative")
    baseline_per_event = _validate_paired_per_event_frame(baseline_per_event, "baseline")
    candidate_per_event = _validate_paired_per_event_frame(candidate_per_event, "candidate")
    if set(baseline_per_event["event_id"]) != set(candidate_per_event["event_id"]):
        raise ValueError(
            "baseline and candidate per-event errors need unique event_id values and the same event_id set"
        )
    merged = baseline_per_event.merge(
        candidate_per_event, on="event_id", suffixes=("_baseline", "_gefs"), validate="one_to_one"
    )
    relative_improvement = float(1 - candidate_mae / baseline_mae) if baseline_mae else 0.0
    improved_events = int(merged["mae_gefs"].lt(merged["mae_baseline"]).sum())
    total_error = float(merged["absolute_error_gefs"].sum())
    largest_share = float(merged["absolute_error_gefs"].max() / total_error) if total_error else 0.0
    gates = {
        "overall_mae_improvement_at_least_5pct": relative_improvement >= 0.05,
        "majority_validation_events_improved": improved_events > len(merged) / 2,
        "largest_candidate_absolute_error_share_at_most_0_40": largest_share <= 0.40,
    }
    if not gates["overall_mae_improvement_at_least_5pct"]:
        decision = "reject_insufficient_mae_improvement"
    elif not gates["majority_validation_events_improved"]:
        decision = "reject_insufficient_event_improvement"
    elif not gates["largest_candidate_absolute_error_share_at_most_0_40"]:
        decision = "reject_concentrated_candidate_error"
    else:
        decision = "candidate_for_replacement_review"
    return {
        "decision": decision,
        "relative_mae_improvement": relative_improvement,
        "improved_validation_events": improved_events,
        "validation_events": len(merged),
        "largest_candidate_absolute_error_share": largest_share,
        "gates": gates,
    }


def benchmark_gefs_multivariable_features(
    frame: pd.DataFrame,
    *,
    baseline_features: Sequence[str],
    gefs_features: Sequence[str],
    n_splits: int = 3,
    seed: int = 42,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare observation-only and GEFS-augmented models on identical event folds."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("GEFS benchmark frame must be a pandas DataFrame")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 3:
        raise ValueError("GEFS benchmark needs at least three rolling folds")
    prepared = _validate_time_safe_gefs_rows(frame)
    missing = _REQUIRED_COLUMNS - set(prepared.columns)
    if missing:
        raise ValueError(f"GEFS benchmark frame missing columns: {sorted(missing)}")
    _validate_exact_event_count(prepared)
    baseline = _validate_features(prepared, baseline_features, "baseline")
    expected_baseline = _measurement_baseline_features(prepared)
    if baseline != expected_baseline:
        raise ValueError("baseline features must be the existing observation measurement columns")
    gefs = _validate_features(prepared, gefs_features, "GEFS")
    if gefs != list(GEFS_SURFACE_FEATURE_COLUMNS):
        raise ValueError("GEFS features must be the complete GEFS surface feature set")
    if set(baseline).intersection(gefs):
        raise ValueError("baseline and GEFS feature columns must not overlap")
    if prepared[gefs].isna().any().any() or not np.isfinite(prepared[gefs].to_numpy(dtype=float)).all():
        raise ValueError("GEFS benchmark frame has missing GEFS feature values")
    target = pd.to_numeric(prepared["remaining_hours"], errors="coerce")
    if target.isna().any() or (target <= 0).any() or not np.isfinite(target).all():
        raise ValueError("GEFS benchmark labels must be positive and finite")
    prepared["remaining_hours"] = target
    folds = rolling_event_splits(prepared, n_splits=n_splits)
    records: list[pd.DataFrame] = []
    fold_reports: list[dict[str, object]] = []
    fold_mae: dict[str, list[float]] = {BASELINE_VARIANT: [], GEFS_VARIANT: []}
    model_features = {
        BASELINE_VARIANT: baseline,
        GEFS_VARIANT: [*baseline, *gefs],
    }
    for fold, (train, valid) in enumerate(folds, start=1):
        train_events = [int(value) for value in train["event_id"].drop_duplicates()]
        validation_events = [int(value) for value in valid["event_id"].drop_duplicates()]
        if set(train_events).intersection(validation_events):
            raise ValueError("GEFS benchmark training and validation events must be disjoint")
        if train["timestamp"].max() >= valid["timestamp"].min():
            raise ValueError("GEFS benchmark training time must precede validation time")
        fold_reports.append(
            {
                "fold": fold,
                "train_event_ids": train_events,
                "validation_event_ids": validation_events,
                "train_end_timestamp": train["timestamp"].max().isoformat(),
                "validation_start_timestamp": valid["timestamp"].min().isoformat(),
            }
        )
        actual = valid["remaining_hours"].reset_index(drop=True)
        for variant, names in model_features.items():
            predicted = _fit_predict_aft(train[names], train["remaining_hours"], valid[names], seed + fold)
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
    predictions = pd.concat(records, ignore_index=True)
    per_event = {
        variant: _event_mae(predictions, variant) for variant in (BASELINE_VARIANT, GEFS_VARIANT)
    }
    metrics = {
        variant: regression_metrics(
            predictions.loc[predictions["variant"].eq(variant), "actual_remaining_hours"],
            predictions.loc[predictions["variant"].eq(variant), "predicted_remaining_hours"],
        )
        for variant in (BASELINE_VARIANT, GEFS_VARIANT)
    }
    decision = decide_gefs_multivariable_candidate(
        float(metrics[BASELINE_VARIANT]["mae"]),
        float(metrics[GEFS_VARIANT]["mae"]),
        per_event[BASELINE_VARIANT],
        per_event[GEFS_VARIANT],
    )
    return {
        "protocol": "expanding chronological complete-event rolling validation",
        "time_safety": {"run_lte_timestamp_lt_valid": True, "valid_at_least_six_hours_ahead": True},
        "gefs_feature_names": gefs,
        "folds": fold_reports,
        "models": [
            {
                "variant": variant,
                "metrics": metrics[variant],
                "fold_mae": fold_mae[variant],
                "per_event_mae": per_event[variant].to_dict(orient="records"),
            }
            for variant in (BASELINE_VARIANT, GEFS_VARIANT)
        ],
        "decision": decision,
    }, predictions


def write_gefs_benchmark_artifacts(
    report: dict[str, object], predictions: pd.DataFrame, output_dir: Path
) -> None:
    """Write the benchmark report and paired fold predictions for review."""
    required = {"fold", "variant", "timestamp", "event_id", "actual_remaining_hours", "predicted_remaining_hours"}
    if not isinstance(predictions, pd.DataFrame) or required - set(predictions.columns):
        raise ValueError("GEFS benchmark predictions are missing required audit columns")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "report.json", report)
    staging = output_dir / "fold_predictions.csv.part"
    try:
        predictions.to_csv(staging, index=False, encoding="utf-8-sig")
        staging.replace(output_dir / "fold_predictions.csv")
    finally:
        staging.unlink(missing_ok=True)
