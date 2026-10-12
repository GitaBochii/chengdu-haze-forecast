import json

import numpy as np
import pandas as pd
import pytest

from haze_forecast.spatial_benchmark import (
    benchmark_spatial_features,
    decide_spatial_candidate,
    write_spatial_benchmark_artifacts,
)


def test_decide_spatial_candidate_requires_coverage_and_stable_improvement():
    accepted = decide_spatial_candidate(
        baseline_fold_mae=[10.0, 10.0, 10.0],
        candidate_fold_mae=[9.0, 9.4, 9.6],
        baseline_overall_mae=10.0,
        candidate_overall_mae=9.3,
        directional_coverage=0.8,
    )
    low_coverage = decide_spatial_candidate(
        baseline_fold_mae=[10.0, 10.0, 10.0],
        candidate_fold_mae=[9.0, 9.4, 9.6],
        baseline_overall_mae=10.0,
        candidate_overall_mae=9.3,
        directional_coverage=0.69,
    )

    assert accepted["decision"] == "candidate_for_replacement_review"
    assert low_coverage["decision"] == "reject_insufficient_directional_coverage"


def test_benchmark_spatial_features_uses_identical_event_folds_without_target_leakage(monkeypatch):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=12, freq="h"),
            "event_id": np.repeat(np.arange(1, 7), 2),
            "remaining_hours": np.arange(12, 0, -1, dtype=float),
            "history_feature": np.arange(12, 0, -1, dtype=float),
            "spatial_pm2_5_gradient_median": np.ones(12),
        }
    )
    calls = []

    def fake_predict(train_x, train_y, valid_x, seed):
        calls.append((train_x.columns.tolist(), train_y.name, valid_x.columns.tolist(), seed))
        offset = 2.0 if len(train_x.columns) == 1 else 1.0
        return valid_x["history_feature"].to_numpy() + offset

    monkeypatch.setattr("haze_forecast.spatial_benchmark._fit_predict_aft", fake_predict)

    report, predictions = benchmark_spatial_features(
        frame,
        baseline_features=["history_feature"],
        enhanced_features=["history_feature", "spatial_pm2_5_gradient_median"],
        n_splits=3,
        seed=10,
    )

    assert report["protocol"] == "expanding chronological complete-event rolling validation"
    assert report["directional_feature_coverage"] == 1.0
    assert len(calls) == 6
    assert all(call[1] == "remaining_hours" for call in calls)
    assert {call[0][0] for call in calls} == {"history_feature"}
    assert all(
        set(fold["train_event_ids"]).isdisjoint(fold["validation_event_ids"])
        for fold in report["folds"]
    )
    assert set(predictions["variant"]) == {"baseline", "directional"}
    assert report["decision"]["decision"] == "candidate_for_replacement_review"


def test_write_spatial_benchmark_artifacts_exports_auditable_files(tmp_path):
    predictions = pd.DataFrame(
        {
            "fold": [1],
            "variant": ["baseline"],
            "event_id": [1],
            "actual_remaining_hours": [4.0],
            "predicted_remaining_hours": [4.5],
        }
    )
    output = tmp_path / "spatial"

    write_spatial_benchmark_artifacts({"protocol": "test"}, predictions, output)

    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == {"protocol": "test"}
    exported = pd.read_csv(output / "fold_predictions.csv", encoding="utf-8-sig")
    assert exported.to_dict(orient="records") == predictions.to_dict(orient="records")


@pytest.mark.parametrize("coverage", [-0.1, 1.1, np.nan])
def test_decide_spatial_candidate_rejects_invalid_coverage(coverage):
    with pytest.raises(ValueError, match="coverage"):
        decide_spatial_candidate([10.0] * 3, [9.0] * 3, 10.0, 9.0, coverage)
