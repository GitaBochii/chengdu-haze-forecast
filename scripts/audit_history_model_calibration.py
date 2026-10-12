"""Audit AFT and L1 history models with a pre-test median-residual calibration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import pandas as pd
from sklearn.metrics import mean_absolute_error

from haze_forecast.calibration import (
    MedianResidualCalibrator,
    prediction_bias_summary,
    remaining_time_segment_metrics,
)
from haze_forecast.evaluation import regression_metrics
from haze_forecast.features import model_feature_names
from haze_forecast.training import (
    AFTSurvivalRegressor,
    _build_model,
    _matrix,
    constrain_event_predictions,
    event_chronological_split,
    rolling_event_splits,
)


class Regressor(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> Regressor: ...

    def predict(self, features: pd.DataFrame): ...


def _constrained(frame: pd.DataFrame, prediction) -> pd.Series:
    raw = pd.Series(prediction, index=frame.index, dtype=float).clip(lower=0.0)
    return constrain_event_predictions(frame, raw)


def _evaluate_candidate(
    name: str,
    factory,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    valid_model: Regressor = factory().fit(_matrix(train, feature_names), train["remaining_hours"])
    valid_unconstrained = pd.Series(
        valid_model.predict(_matrix(valid, feature_names)), index=valid.index, dtype=float
    ).clip(lower=0.0)
    valid_state_constrained = _constrained(valid, valid_unconstrained)
    raw_calibrator = MedianResidualCalibrator.fit(valid["remaining_hours"], valid_unconstrained)
    state_calibrator = MedianResidualCalibrator.fit(valid["remaining_hours"], valid_state_constrained)

    combined = pd.concat([train, valid], ignore_index=True)
    test_model: Regressor = factory().fit(
        _matrix(combined, feature_names), combined["remaining_hours"]
    )
    test_unconstrained = pd.Series(
        test_model.predict(_matrix(test, feature_names)), index=test.index, dtype=float
    ).clip(lower=0.0)
    test_state_constrained = _constrained(test, test_unconstrained)
    variants = {
        "raw": (valid_unconstrained, test_unconstrained),
        "median_residual_calibrated": (
            raw_calibrator.predict(valid_unconstrained),
            raw_calibrator.predict(test_unconstrained),
        ),
        "state_constrained": (valid_state_constrained, test_state_constrained),
        "state_constrained_calibrated": (
            _constrained(valid, state_calibrator.predict(valid_state_constrained)),
            _constrained(test, state_calibrator.predict(test_state_constrained)),
        ),
    }
    report: dict[str, object] = {
        "model": name,
        "calibrators": {
            "raw": asdict(raw_calibrator),
            "state_constrained": asdict(state_calibrator),
        },
        "variants": {},
    }
    test_predictions = test[["timestamp", "event_id", "remaining_hours"]].rename(
        columns={"remaining_hours": "actual_remaining_hours"}
    )
    for variant, (valid_prediction, test_prediction) in variants.items():
        validation_mae = float(mean_absolute_error(valid["remaining_hours"], valid_prediction))
        test_metrics = regression_metrics(test["remaining_hours"], test_prediction)
        report["variants"][variant] = {
            "validation_mae": validation_mae,
            "test_metrics": test_metrics,
            "validation_bias": prediction_bias_summary(valid["remaining_hours"], valid_prediction),
            "test_bias": prediction_bias_summary(test["remaining_hours"], test_prediction),
            "test_segments": remaining_time_segment_metrics(test["remaining_hours"], test_prediction),
        }
        test_predictions[f"{name}_{variant}_hours"] = test_prediction.to_numpy()
    return report, test_predictions


def _event_fit_calibration_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a chronological fit/calibration split without splitting an event."""
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    event_ids = ordered["event_id"].drop_duplicates().tolist()
    if len(event_ids) < 3:
        raise ValueError("at least three events are required for calibration")
    calibration_start = min(max(1, int(len(event_ids) * 0.8)), len(event_ids) - 1)
    fit_ids = event_ids[:calibration_start]
    calibration_ids = event_ids[calibration_start:]
    return (
        ordered.loc[ordered["event_id"].isin(fit_ids)].reset_index(drop=True),
        ordered.loc[ordered["event_id"].isin(calibration_ids)].reset_index(drop=True),
    )


def _rolling_calibration_audit(
    factory,
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    splits: int,
) -> dict[str, object]:
    records: list[pd.DataFrame] = []
    offsets: list[float] = []
    for fold, (outer_train, outer_valid) in enumerate(rolling_event_splits(frame, n_splits=splits), start=1):
        fit, calibration = _event_fit_calibration_split(outer_train)
        calibration_model: Regressor = factory().fit(
            _matrix(fit, feature_names), fit["remaining_hours"]
        )
        calibration_prediction = pd.Series(
            calibration_model.predict(_matrix(calibration, feature_names)),
            index=calibration.index,
            dtype=float,
        ).clip(lower=0.0)
        calibrator = MedianResidualCalibrator.fit(
            calibration["remaining_hours"], calibration_prediction
        )
        outer_model: Regressor = factory().fit(
            _matrix(outer_train, feature_names), outer_train["remaining_hours"]
        )
        raw_prediction = pd.Series(
            outer_model.predict(_matrix(outer_valid, feature_names)),
            index=outer_valid.index,
            dtype=float,
        ).clip(lower=0.0)
        records.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "event_id": outer_valid["event_id"],
                    "actual": outer_valid["remaining_hours"],
                    "raw": raw_prediction,
                    "calibrated": calibrator.predict(raw_prediction),
                }
            )
        )
        offsets.append(calibrator.offset_hours)
    joined = pd.concat(records, ignore_index=True)
    raw_metrics = regression_metrics(joined["actual"], joined["raw"])
    calibrated_metrics = regression_metrics(joined["actual"], joined["calibrated"])
    return {
        "folds": splits,
        "rows": len(joined),
        "events": int(joined["event_id"].nunique()),
        "calibration_offsets_hours": offsets,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "mae_change_calibrated_minus_raw": calibrated_metrics["mae"] - raw_metrics["mae"],
        "decision": (
            "reject_median_residual_calibration"
            if calibrated_metrics["mae"] >= raw_metrics["mae"]
            else "candidate_for_more_validation"
        ),
        "raw_bias": prediction_bias_summary(joined["actual"], joined["raw"]),
        "calibrated_bias": prediction_bias_summary(joined["actual"], joined["calibrated"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/training_features.parquet")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("artifacts/model_metadata.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/model_calibration_audit")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rolling-splits", type=int, default=3)
    args = parser.parse_args()
    if args.rolling_splits < 2:
        raise ValueError("--rolling-splits must be at least 2")

    frame = pd.read_parquet(args.input)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    feature_names = model_feature_names(frame)
    train, valid, test = event_chronological_split(frame)
    lightgbm_params = metadata["best_params"]
    lightgbm_iterations = int(metadata["best_iteration"])
    candidates = {
        "lightgbm_l1": lambda: _build_model(
            lightgbm_params, args.seed, n_estimators=lightgbm_iterations
        ),
        "xgboost_aft": lambda: AFTSurvivalRegressor(seed=args.seed),
    }
    reports: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for name, factory in candidates.items():
        report, predictions = _evaluate_candidate(name, factory, train, valid, test, feature_names)
        reports.append(report)
        prediction_frames.append(predictions.drop(columns=["timestamp", "event_id", "actual_remaining_hours"]))
    rolling_reports = {
        name: _rolling_calibration_audit(
            factory, pd.concat([train, valid], ignore_index=True), feature_names, splits=args.rolling_splits
        )
        for name, factory in candidates.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_predictions = pd.concat(
        [test[["timestamp", "event_id", "remaining_hours"]].rename(
            columns={"remaining_hours": "actual_remaining_hours"}
        ), *prediction_frames],
        axis=1,
    )
    output_predictions.to_csv(
        args.output_dir / "heldout_predictions.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "protocol": (
            "chronological completed-event holdout; each calibrator is fit only on pre-test validation "
            "events; raw and state-constrained deployment paths are reported separately"
        ),
        "rows": {"train": len(train), "validation": len(valid), "test": len(test)},
        "events": {
            "train": int(train["event_id"].nunique()),
            "validation": int(valid["event_id"].nunique()),
            "test": int(test["event_id"].nunique()),
        },
        "candidates": reports,
        "rolling_pretest_audit": rolling_reports,
        "warning": "Exploratory audit only. Do not replace artifacts or Streamlit model selection from this file.",
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for report in reports:
        variants = report["variants"]
        raw_mae = variants["raw"]["test_metrics"]["mae"]
        calibrated_mae = variants["median_residual_calibrated"]["test_metrics"]["mae"]
        print(f"{report['model']}: raw={raw_mae:.4f}, calibrated={calibrated_mae:.4f}")


if __name__ == "__main__":
    main()
