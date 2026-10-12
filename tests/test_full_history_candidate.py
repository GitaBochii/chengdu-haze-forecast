from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from haze_forecast.full_history_candidate import build_full_history_deployment_candidate


class _MedianAFT:
    fit_rows: ClassVar[list[int]] = []

    def __init__(self, **_kwargs):
        self.median = 0.0

    def fit(self, _features, target):
        self.median = float(target.median())
        self.fit_rows.append(len(target))
        return self

    def predict(self, features):
        return np.full(len(features), self.median)


def _frame(year: int, event_id: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(f"{year}-01-01", periods=2, freq="h", tz="Asia/Shanghai"),
            "event_id": [event_id, event_id],
            "remaining_hours": [2.0, 1.0],
            "aqi": [150.0, 120.0],
            "pm2_5": [100.0, 80.0],
            "boundary_layer_height_m": [np.nan, np.nan],
        }
    )


def test_full_history_candidate_refits_on_every_completed_year_and_keeps_external_metric(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("haze_forecast.full_history_candidate.AFTSurvivalRegressor", _MedianAFT)
    _MedianAFT.fit_rows = []

    result = build_full_history_deployment_candidate(
        {
            "history": _frame(2021, 1),
            "external_2025": _frame(2025, 1),
        },
        tmp_path,
        feature_names=["aqi", "pm2_5", "boundary_layer_height_m", "boundary_layer_height_m_is_missing"],
        snapshot_feature_names=["aqi", "pm2_5"],
        external_mae=26.267,
        external_metric_source="2025 event-isolated forward validation",
    )

    assert _MedianAFT.fit_rows == [4, 4]
    assert result["metadata"]["fitted_through"] == "2025-01-01T01:00:00+08:00"
    assert result["metadata"]["final_training_rows"] == 4
    assert result["metadata"]["selected_history_validation_mae"] == 26.267
    assert result["metadata"]["accuracy_claim"] == "external_validation_only"
    assert result["metrics"]["xgboost_aft"]["mae"] == 26.267
    assert result["metadata"]["selected_snapshot_validation_mae"] is None
    assert result["metrics"]["xgboost_aft_snapshot"]["mae"] is None
    assert (tmp_path / "aft_model.joblib").exists()
    assert (tmp_path / "static_aft_model.joblib").exists()
    assert (tmp_path / "model_metadata.json").exists()


def test_full_history_candidate_rejects_missing_required_features(tmp_path):
    with pytest.raises(ValueError, match="missing deployment features"):
        build_full_history_deployment_candidate(
            {"history": _frame(2021, 1)},
            tmp_path,
            feature_names=["aqi", "not_available"],
            snapshot_feature_names=["aqi"],
            external_mae=1.0,
            external_metric_source="test",
        )
