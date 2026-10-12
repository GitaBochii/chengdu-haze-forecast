from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from haze_forecast.cnemc_candidate import build_cnemc_2014_candidate


class _MedianAFT:
    fit_rows: ClassVar[list[int]] = []

    def __init__(self, **_kwargs):
        self.median = 0.0

    def fit(self, features, target):
        self.median = float(target.median())
        self.fit_rows.append(len(features))
        return self

    def predict(self, features):
        return np.full(len(features), self.median)


def _base_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=8, freq="h", tz="Asia/Shanghai")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "event_id": list(range(1, 9)),
            "remaining_hours": np.arange(8, 0, -1, dtype=float),
            "aqi": np.linspace(110.0, 180.0, 8),
            "pm2_5": np.linspace(80.0, 120.0, 8),
        }
    )


def test_build_candidate_uses_unchanged_base_test_events_and_writes_isolated_artifacts(
    monkeypatch, tmp_path
):
    base = _base_frame()
    prefix = pd.DataFrame(
        {
            "timestamp": pd.date_range("2014-01-01", periods=2, freq="h", tz="Asia/Shanghai"),
            "event_id": ["cnemc_1", "cnemc_2"],
            "remaining_hours": [4.0, 3.0],
            "aqi": [130.0, 140.0],
            "pm2_5": [90.0, 100.0],
        }
    )
    _MedianAFT.fit_rows = []
    monkeypatch.setattr("haze_forecast.cnemc_candidate.AFTSurvivalRegressor", _MedianAFT)

    result = build_cnemc_2014_candidate(base, prefix, tmp_path)

    assert result["metrics"]["xgboost_aft"]["mae"] >= 0
    assert result["metadata"]["evaluation_protocol"] == "fixed 2015-2021-style base holdout with 2014 prefix"
    assert result["metadata"]["evaluation_test_event_ids"] == [8]
    assert _MedianAFT.fit_rows == [9, 9, 10, 10]
    assert (tmp_path / "aft_model.joblib").exists()
    assert (tmp_path / "static_aft_model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "model_metadata.json").exists()
    assert (tmp_path / "test_predictions.csv").exists()


def test_build_candidate_rejects_a_prefix_that_is_not_older_than_base():
    base = _base_frame()
    invalid_prefix = base.iloc[[0]].assign(event_id="cnemc_1")

    with pytest.raises(ValueError, match="strictly precede the base data"):
        build_cnemc_2014_candidate(base, invalid_prefix, Path("unused"))
