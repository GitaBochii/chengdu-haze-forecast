"""Evaluate state-grouped residual calibration without temporal leakage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.event_residual_calibration import (
    EventResidualCalibrator,
    out_of_fold_event_predictions,
    select_candidate,
)
from haze_forecast.features import add_missingness_indicators
from haze_forecast.gfs_external_validation import event_equal_mae
from haze_forecast.temporal_validation import event_isolated_year_split
from haze_forecast.training import AFTSurvivalRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts" / "event_residual_calibration_benchmark"
STRICT_PREDICTIONS = ROOT / "artifacts" / "strict_history_backtest" / "predictions.parquet"
PBL_COLUMNS = [
    "boundary_layer_height_m",
    "boundary_layer_height_m_lag_1",
    "boundary_layer_height_m_lag_3",
    "boundary_layer_height_m_lag_24",
    "boundary_layer_height_m_roll_3",
    "boundary_layer_height_m_roll_24",
    "boundary_layer_height_m_change_3",
]
SOURCES = {
    "prefix_2014": DATA / "cnemc_chengdu_2014_training_features.parquet",
    "base": DATA / "training_features.parquet",
    "external_2022": DATA / "cnemc_chengdu_2022_training_features.parquet",
    "external_2023": DATA / "cnemc_chengdu_2023_training_features.parquet",
    "external_2024_2025": DATA / "cnemc_chengdu_2024_2025_training_features.parquet",
}
CANDIDATES = {
    "none": None,
    "state_30_30": (30, 30),
    "state_30_60": (30, 60),
    "state_60_60": (60, 60),
    "state_60_120": (60, 120),
}


def prepare(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Apply the deployed missing-data schema with source-scoped events."""
    prepared = frame.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="raise")
    prepared["event_key"] = source + ":" + prepared["event_id"].astype(str)
    prepared = add_missingness_indicators(prepared, PBL_COLUMNS)
    change_flag = "boundary_layer_height_m_change_3_is_missing"
    if change_flag not in prepared.columns:
        prepared[change_flag] = (
            prepared["boundary_layer_height_m"].isna()
            | prepared["boundary_layer_height_m_lag_3"].isna()
        ).astype(float)
    return prepared


def metrics(actual: pd.Series, prediction: np.ndarray, event_key: pd.Series) -> dict[str, float]:
    """Return row- and event-balanced absolute errors."""
    observed = actual.to_numpy(dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    return {
        "row_mae": float(np.abs(predicted - observed).mean()),
        "event_equal_mae": event_equal_mae(observed, predicted, event_key),
        "prediction_mean": float(predicted.mean()),
        "actual_mean": float(observed.mean()),
    }


def build_context(
    train: pd.DataFrame, test: pd.DataFrame, feature_names: list[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit a future-facing AFT model and independent historical OOF residual rows."""
    model = AFTSurvivalRegressor(seed=42, nthread=1).fit(
        train[feature_names], train["remaining_hours"]
    )
    base_test = np.asarray(model.predict(test[feature_names]), dtype=float).clip(min=0.0)
    oof = out_of_fold_event_predictions(
        train,
        feature_names=feature_names,
        model_factory=lambda: AFTSurvivalRegressor(seed=42, nthread=1),
        minimum_train_rows=100,
        fold_count=4,
    )
    if oof.empty:
        raise ValueError("training partition produced no residual calibration rows")
    return base_test, oof


def calibrated_prediction(
    candidate: tuple[int, int] | None,
    base_prediction: np.ndarray,
    test: pd.DataFrame,
    oof: pd.DataFrame,
) -> np.ndarray:
    """Leave baseline predictions unchanged or apply one fixed historical calibrator."""
    if candidate is None:
        return np.asarray(base_prediction, dtype=float)
    minimum_state_rows, shrinkage_rows = candidate
    calibrator = EventResidualCalibrator(minimum_state_rows, shrinkage_rows).fit(
        oof, oof["base_prediction"].to_numpy(dtype=float), oof["remaining_hours"].to_numpy(dtype=float)
    )
    return calibrator.predict(base_prediction, test)


def score_partition(
    train: pd.DataFrame, test: pd.DataFrame, feature_names: list[str]
) -> dict[str, dict[str, float]]:
    """Score every calibration candidate using one common base-model context."""
    base_prediction, oof = build_context(train, test, feature_names)
    return {
        name: {
            **metrics(
                test["remaining_hours"],
                calibrated_prediction(candidate, base_prediction, test, oof),
                test["event_key"],
            ),
            "oof_rows": len(oof),
        }
        for name, candidate in CANDIDATES.items()
    }


def strict_history_metrics(candidate: tuple[int, int] | None, frame: pd.DataFrame) -> dict[str, object]:
    """Recalibrate stored strict predictions from preceding years only."""
    result: dict[str, object] = {}
    for year in range(2022, 2026):
        year_start = pd.Timestamp(f"{year}-01-01", tz=frame["timestamp"].dt.tz)
        train = frame[frame["timestamp"].lt(year_start)]
        test = frame[frame["timestamp"].dt.year.eq(year)]
        if train.empty or test.empty:
            raise ValueError(f"strict history data is incomplete for {year}")
        baseline = test["strict_prediction_hours"].to_numpy(dtype=float)
        corrected = calibrated_prediction(candidate, baseline, test, train)
        result[str(year)] = {
            "baseline": metrics(test["remaining_hours"], baseline, test["event_key"]),
            "calibrated": metrics(test["remaining_hours"], corrected, test["event_key"]),
        }
    return result


def main() -> None:
    metadata_path = ROOT / "artifacts" / "cnemc_2025_deployment_candidate" / "model_metadata.json"
    feature_names = list(json.loads(metadata_path.read_text(encoding="utf-8"))["feature_names"])
    if len(feature_names) != 87:
        raise ValueError(f"expected the deployed 87 features, got {len(feature_names)}")

    rows = pd.concat(
        [prepare(pd.read_parquet(path), source) for source, path in SOURCES.items()],
        ignore_index=True,
    ).sort_values("timestamp")
    missing_features = set(feature_names) - set(rows.columns)
    if missing_features:
        raise ValueError(f"extended data is missing model features: {sorted(missing_features)}")

    partitions: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {
        year: event_isolated_year_split(rows, year, event_column="event_key") for year in range(2022, 2026)
    }
    annual_scores = {
        str(year): score_partition(train, test, feature_names)
        for year, (train, test) in partitions.items()
    }
    selection = {name: values["row_mae"] for name, values in annual_scores["2024"].items()}
    confirmation = {name: values["row_mae"] for name, values in annual_scores["2025"].items()}
    selected_2024 = min(selection, key=selection.__getitem__)
    accepted = select_candidate(selection, confirmation, baseline_name="none")
    selected_candidate = CANDIDATES[selected_2024]

    strict_frame = pd.read_parquet(STRICT_PREDICTIONS).copy()
    strict_frame["timestamp"] = pd.to_datetime(strict_frame["timestamp"], errors="raise")
    strict_scores = strict_history_metrics(selected_candidate, strict_frame)
    strict_improved_years = sum(
        values["calibrated"]["row_mae"] < values["baseline"]["row_mae"]
        for values in strict_scores.values()
    )
    strict_not_worse = all(
        values["calibrated"]["row_mae"] <= values["baseline"]["row_mae"]
        for values in strict_scores.values()
    )
    accepted = accepted if strict_not_worse and strict_improved_years >= 2 else None

    reason = (
        "accepted"
        if accepted is not None
        else "2025_confirmation_or_strict_history_stability_gate_not_met"
    )
    report = {
        "protocol": (
            "2024 selects state-residual calibration from event-isolated AFT forecasts; "
            "2025 confirms without retuning. Residual rows are chronological event-batch out-of-fold "
            "predictions. Strict-history calibration uses only prior-year strict predictions."
        ),
        "feature_count": len(feature_names),
        "candidates": {name: value for name, value in CANDIDATES.items() if value is not None},
        "selection_year_2024": annual_scores["2024"],
        "selected_from_2024": selected_2024,
        "confirmation_year_2025": annual_scores["2025"],
        "annual_forward_metrics": annual_scores,
        "strict_history_2022_2025": strict_scores,
        "decision": {
            "status": "accept" if accepted is not None else "reject",
            "selected_candidate": accepted,
            "reason": reason,
            "strict_improved_years": strict_improved_years,
            "strict_not_worse_all_years": strict_not_worse,
            "deployment_artifact_changed": False,
        },
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = ARTIFACTS / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"2024 selected: {selected_2024} ({selection[selected_2024]:.3f} MAE)")
    print(f"2025 baseline: {confirmation['none']:.3f} MAE")
    print(f"2025 selected: {confirmation[selected_2024]:.3f} MAE")
    print(f"decision: {report['decision']['status']} ({reason})")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
