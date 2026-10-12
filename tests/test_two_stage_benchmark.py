from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.two_stage_benchmark import (
    TwoStageDurationRegressor,
    benchmark_two_stage,
    decide_external_years,
    select_gate_threshold,
    write_two_stage_artifacts,
)


def test_two_stage_falls_back_when_a_duration_partition_is_too_small(monkeypatch):
    class ConstantAft:
        def __init__(self, **_kwargs):
            self.value = 0.0

        def fit(self, _features, target):
            self.value = float(pd.Series(target).median())
            return self

        def predict(self, features):
            return np.full(len(features), self.value)

    monkeypatch.setattr("haze_forecast.two_stage_benchmark.AFTSurvivalRegressor", ConstantAft)
    features = pd.DataFrame({"aqi": [110.0, 120.0, 130.0, 140.0]})
    target = pd.Series([3.0, 5.0, 7.0, 48.0])

    model = TwoStageDurationRegressor(min_partition_rows=3).fit(features, target)

    assert model.uses_fallback is True
    assert model.predict(features).tolist() == model.predict_baseline(features).tolist()


def test_select_gate_threshold_returns_half_when_scores_tie():
    probabilities = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([True, True, False, False])

    assert select_gate_threshold(probabilities, labels) == 0.5


def _event_frame() -> pd.DataFrame:
    rows = []
    for event_id in range(1, 9):
        for step, remaining in enumerate((30.0, 10.0)):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2020-01-01", tz="Asia/Shanghai")
                    + pd.Timedelta(hours=(event_id - 1) * 3 + step),
                    "event_id": event_id,
                    "remaining_hours": remaining,
                    "aqi": 110.0 + event_id,
                }
            )
    return pd.DataFrame(rows)


def test_benchmark_records_disjoint_outer_train_and_validation_events():
    report, predictions = benchmark_two_stage(_event_frame(), ["aqi"], n_splits=3)

    for fold in report["folds"]:
        assert not set(fold["train_event_ids"]).intersection(fold["validation_event_ids"])
        assert fold["train_end_timestamp"] < fold["validation_start_timestamp"]
    assert set(predictions["model"]) == {"xgboost_aft", "two_stage_duration"}


def test_external_year_gate_rejects_average_improvement_below_three_percent():
    result = decide_external_years({2022: (10.0, 9.8), 2023: (20.0, 19.5)})

    assert result["decision"] == "reject_no_default_replacement"


def test_benchmark_script_uses_isolated_default_output():
    text = Path("scripts/benchmark_two_stage_duration.py").read_text(encoding="utf-8")

    assert "artifacts/two_stage_duration_benchmark" in text
    assert "training_features.parquet" in text


def test_write_two_stage_artifacts_creates_only_isolated_report_and_predictions(tmp_path):
    report = {"decision": {"decision": "reject_no_default_replacement"}}
    predictions = pd.DataFrame(
        {
            "fold": [1],
            "model": ["two_stage_duration"],
            "timestamp": [pd.Timestamp("2020-01-01", tz="Asia/Shanghai")],
            "event_id": [1],
            "actual_remaining_hours": [12.0],
            "predicted_remaining_hours": [10.0],
            "gate_probability": [0.6],
            "selected_duration_band": ["short"],
            "used_fallback": [False],
        }
    )

    write_two_stage_artifacts(report, predictions, tmp_path)

    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "fold_predictions.csv").exists()
