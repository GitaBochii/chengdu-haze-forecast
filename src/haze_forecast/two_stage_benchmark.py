"""Two-stage gated duration candidates evaluated outside the production predictor."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

from haze_forecast.evaluation import regression_metrics
from haze_forecast.model_benchmark import decide_candidate
from haze_forecast.storage import write_json_atomic
from haze_forecast.training import AFTSurvivalRegressor, rolling_event_splits

GATE_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def _build_gate(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=8,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        n_jobs=1,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )


def select_gate_threshold(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    """Choose a stable binary recovery threshold using training-only calibration rows."""
    probability_values = np.asarray(probabilities, dtype=float)
    label_values = np.asarray(labels, dtype=bool)
    if probability_values.ndim != 1 or label_values.ndim != 1:
        raise ValueError("gate probabilities and labels must be one-dimensional")
    if len(probability_values) != len(label_values) or not len(probability_values):
        raise ValueError("gate probabilities and labels must be equally sized and non-empty")
    if not np.isfinite(probability_values).all() or (probability_values < 0).any() or (
        probability_values > 1
    ).any():
        raise ValueError("gate probabilities must be finite values between zero and one")
    if label_values.min() == label_values.max():
        return 0.5
    scores = [
        (float(balanced_accuracy_score(label_values, probability_values >= threshold)), float(threshold))
        for threshold in GATE_THRESHOLDS
    ]
    best_score = max(score for score, _ in scores)
    tied = [threshold for score, threshold in scores if np.isclose(score, best_score)]
    return min(tied, key=lambda threshold: (abs(threshold - 0.5), threshold))


def _calibrated_gate_threshold(
    features: pd.DataFrame, short_labels: pd.Series, event_ids: pd.Series | None, seed: int
) -> float:
    if event_ids is None:
        return 0.5
    events = pd.Series(event_ids, index=features.index)
    ordered_events = events.drop_duplicates().tolist()
    calibration_count = max(1, int(np.ceil(len(ordered_events) * 0.2)))
    if len(ordered_events) - calibration_count < 2:
        return 0.5
    calibration_events = set(ordered_events[-calibration_count:])
    calibration_mask = events.isin(calibration_events)
    if short_labels.loc[calibration_mask].nunique() < 2:
        return 0.5
    gate = _build_gate(seed).fit(
        features.loc[~calibration_mask], short_labels.loc[~calibration_mask].astype(int)
    )
    probabilities = gate.predict_proba(features.loc[calibration_mask])[:, 1]
    return select_gate_threshold(probabilities, short_labels.loc[calibration_mask])


class TwoStageDurationRegressor:
    """Use a 24-hour recovery gate to route rows to short or long AFT regressors."""

    def __init__(self, horizon_hours: float = 24, min_partition_rows: int = 80, seed: int = 42):
        self.horizon_hours = horizon_hours
        self.min_partition_rows = min_partition_rows
        self.seed = seed
        self.uses_fallback = True
        self.threshold = 0.5
        self.baseline: AFTSurvivalRegressor | None = None
        self.gate: XGBClassifier | None = None
        self.short_model: AFTSurvivalRegressor | None = None
        self.long_model: AFTSurvivalRegressor | None = None

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        event_ids: pd.Series | None = None,
    ) -> TwoStageDurationRegressor:
        values = pd.Series(target, index=features.index, dtype=float)
        self.baseline = AFTSurvivalRegressor(num_boost_round=500, seed=self.seed, nthread=1).fit(
            features, values
        )
        short = values.le(self.horizon_hours)
        if short.sum() < self.min_partition_rows or (~short).sum() < self.min_partition_rows:
            return self
        self.threshold = _calibrated_gate_threshold(features, short, event_ids, self.seed)
        self.gate = _build_gate(self.seed).fit(features, short.astype(int))
        self.short_model = AFTSurvivalRegressor(
            num_boost_round=500, seed=self.seed, nthread=1
        ).fit(features.loc[short], values.loc[short])
        self.long_model = AFTSurvivalRegressor(
            num_boost_round=500, seed=self.seed, nthread=1
        ).fit(features.loc[~short], values.loc[~short])
        self.uses_fallback = False
        return self

    def predict_baseline(self, features: pd.DataFrame) -> np.ndarray:
        if self.baseline is None:
            raise ValueError("two-stage model must be fitted before prediction")
        return np.asarray(self.baseline.predict(features), dtype=float).clip(min=0.0)

    def gate_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        if self.uses_fallback or self.gate is None:
            return np.full(len(features), np.nan)
        return np.asarray(self.gate.predict_proba(features)[:, 1], dtype=float)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.uses_fallback:
            return self.predict_baseline(features)
        if self.short_model is None or self.long_model is None:
            raise ValueError("two-stage duration models are unavailable")
        probabilities = self.gate_probabilities(features)
        choose_short = probabilities >= self.threshold
        prediction = np.asarray(self.long_model.predict(features), dtype=float)
        if choose_short.any():
            prediction[choose_short] = self.short_model.predict(features.loc[choose_short])
        return prediction.clip(min=0.0)


def _validate_benchmark_inputs(
    frame: pd.DataFrame, feature_names: Sequence[str], n_splits: int
) -> tuple[pd.DataFrame, list[str]]:
    required = {"timestamp", "event_id", "remaining_hours"}
    missing = required - set(frame.columns)
    names = list(feature_names)
    missing_features = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"two-stage benchmark missing required columns: {sorted(missing)}")
    if not names or missing_features:
        raise ValueError(f"two-stage benchmark missing feature columns: {sorted(missing_features)}")
    if n_splits < 3:
        raise ValueError("two-stage benchmark requires at least three splits")
    prepared = frame.sort_values("timestamp").reset_index(drop=True).copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="raise")
    prepared["remaining_hours"] = pd.to_numeric(prepared["remaining_hours"], errors="raise")
    if prepared["remaining_hours"].isna().any() or prepared["remaining_hours"].le(0).any():
        raise ValueError("two-stage benchmark targets must be positive and finite")
    return prepared, names


def _mae(actual: pd.Series, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(actual.to_numpy(dtype=float) - prediction)))


def _prediction_rows(
    fold: int,
    valid: pd.DataFrame,
    model_name: str,
    prediction: np.ndarray,
    gate_probability: np.ndarray,
    selected_band: np.ndarray,
    used_fallback: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": fold,
            "model": model_name,
            "timestamp": valid["timestamp"].to_numpy(),
            "event_id": valid["event_id"].to_numpy(),
            "actual_remaining_hours": valid["remaining_hours"].to_numpy(dtype=float),
            "predicted_remaining_hours": prediction,
            "gate_probability": gate_probability,
            "selected_duration_band": selected_band,
            "used_fallback": used_fallback,
        }
    )


def benchmark_two_stage(
    frame: pd.DataFrame, feature_names: Sequence[str], n_splits: int = 3, seed: int = 42
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare a gated two-stage candidate against the same-fold AFT baseline."""
    prepared, names = _validate_benchmark_inputs(frame, feature_names, n_splits)
    fold_reports: list[dict[str, object]] = []
    baseline_fold_mae: list[float] = []
    candidate_fold_mae: list[float] = []
    records: list[pd.DataFrame] = []
    for fold, (train, valid) in enumerate(rolling_event_splits(prepared, n_splits), start=1):
        if train["timestamp"].max() >= valid["timestamp"].min():
            raise ValueError("two-stage training time must precede validation time")
        model = TwoStageDurationRegressor(seed=seed + fold).fit(
            train[names], train["remaining_hours"], train["event_id"]
        )
        baseline = model.predict_baseline(valid[names])
        candidate = model.predict(valid[names])
        probabilities = model.gate_probabilities(valid[names])
        bands = np.where(probabilities >= model.threshold, "short", "long")
        if model.uses_fallback:
            bands = np.full(len(valid), "baseline_fallback")
        baseline_mae = _mae(valid["remaining_hours"], baseline)
        candidate_mae = _mae(valid["remaining_hours"], candidate)
        baseline_fold_mae.append(baseline_mae)
        candidate_fold_mae.append(candidate_mae)
        fold_reports.append(
            {
                "fold": fold,
                "train_event_ids": train["event_id"].drop_duplicates().tolist(),
                "validation_event_ids": valid["event_id"].drop_duplicates().tolist(),
                "train_end_timestamp": train["timestamp"].max().isoformat(),
                "validation_start_timestamp": valid["timestamp"].min().isoformat(),
                "gate_threshold": model.threshold,
                "used_fallback": model.uses_fallback,
                "baseline_mae": baseline_mae,
                "candidate_mae": candidate_mae,
            }
        )
        records.extend(
            [
                _prediction_rows(
                    fold,
                    valid,
                    "xgboost_aft",
                    baseline,
                    np.full(len(valid), np.nan),
                    np.full(len(valid), "baseline"),
                    False,
                ),
                _prediction_rows(
                    fold,
                    valid,
                    "two_stage_duration",
                    candidate,
                    probabilities,
                    bands,
                    model.uses_fallback,
                ),
            ]
        )
    predictions = pd.concat(records, ignore_index=True)
    model_metrics = {
        name: regression_metrics(
            predictions.loc[predictions["model"].eq(name), "actual_remaining_hours"],
            predictions.loc[predictions["model"].eq(name), "predicted_remaining_hours"],
        )
        for name in ("xgboost_aft", "two_stage_duration")
    }
    report = {
        "protocol": "expanding chronological complete-event rolling validation",
        "folds": fold_reports,
        "models": {
            "xgboost_aft": {"metrics": model_metrics["xgboost_aft"], "fold_mae": baseline_fold_mae},
            "two_stage_duration": {
                "metrics": model_metrics["two_stage_duration"],
                "fold_mae": candidate_fold_mae,
            },
        },
        "decision": decide_candidate(
            baseline_fold_mae,
            candidate_fold_mae,
            baseline_overall_mae=float(model_metrics["xgboost_aft"]["mae"]),
            candidate_overall_mae=float(model_metrics["two_stage_duration"]["mae"]),
        ),
    }
    return report, predictions


def decide_external_years(yearly_mae: dict[int, tuple[float, float]]) -> dict[str, object]:
    """Apply the predeclared two-year external replacement gate."""
    if len(yearly_mae) != 2:
        raise ValueError("two external year MAE pairs are required")
    improvements = {
        year: float(1 - candidate / baseline)
        for year, (baseline, candidate) in yearly_mae.items()
        if baseline > 0 and candidate >= 0
    }
    if len(improvements) != 2:
        raise ValueError("external MAE values must have positive baselines and non-negative candidates")
    average = float(np.mean(list(improvements.values())))
    accepted = average >= 0.03 and min(improvements.values()) >= -0.05
    return {
        "relative_mae_improvement_by_year": improvements,
        "average_relative_mae_improvement": average,
        "decision": "accept_for_ui_review" if accepted else "reject_no_default_replacement",
    }


def write_two_stage_artifacts(
    report: dict[str, object], predictions: pd.DataFrame, output_dir: Path
) -> None:
    """Write candidate-only benchmark artifacts without touching default-model files."""
    required = {
        "fold",
        "model",
        "timestamp",
        "event_id",
        "actual_remaining_hours",
        "predicted_remaining_hours",
        "gate_probability",
        "selected_duration_band",
        "used_fallback",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"two-stage predictions missing columns: {sorted(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_path = output_dir / "fold_predictions.csv.part"
    try:
        write_json_atomic(output_dir / "report.json", report)
        predictions.to_csv(staging_path, index=False, encoding="utf-8-sig")
        staging_path.replace(output_dir / "fold_predictions.csv")
    finally:
        staging_path.unlink(missing_ok=True)
