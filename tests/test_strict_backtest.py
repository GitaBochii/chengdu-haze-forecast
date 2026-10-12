from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from haze_forecast.strict_backtest import rolling_event_backtest


class _MeanModel:
    fit_rows: ClassVar[list[int]] = []

    def fit(self, _features, target):
        self.fit_rows.append(len(target))
        self.value = float(target.mean())
        return self

    def predict(self, features):
        return np.full(len(features), self.value)


def _events() -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=6, freq="h", tz="Asia/Shanghai")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "event_key": ["first", "first", "second", "second", "third", "third"],
            "remaining_hours": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            "aqi": [150.0, 120.0, 170.0, 150.0, 190.0, 180.0],
        }
    )


def test_rolling_event_backtest_uses_only_completed_preceding_events():
    _MeanModel.fit_rows = []

    result = rolling_event_backtest(
        _events(),
        feature_names=["aqi"],
        minimum_train_rows=2,
        model_factory=_MeanModel,
    )

    assert _MeanModel.fit_rows == [2, 4]
    assert result["event_key"].tolist() == ["second", "second", "third", "third"]
    assert result["strict_prediction_hours"].tolist() == [1.5, 1.5, 2.5, 2.5]
    assert result["train_rows"].tolist() == [2, 2, 4, 4]
    assert result["train_end_timestamp"].tolist() == [
        pd.Timestamp("2020-01-01 01:00", tz="Asia/Shanghai"),
        pd.Timestamp("2020-01-01 01:00", tz="Asia/Shanghai"),
        pd.Timestamp("2020-01-01 03:00", tz="Asia/Shanghai"),
        pd.Timestamp("2020-01-01 03:00", tz="Asia/Shanghai"),
    ]


def test_rolling_event_backtest_requires_a_source_scoped_event_identifier():
    with pytest.raises(ValueError, match="event_key"):
        rolling_event_backtest(
            _events().drop(columns="event_key"),
            feature_names=["aqi"],
            model_factory=_MeanModel,
        )
