from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from haze_forecast.event_residual_calibration import (
    EventResidualCalibrator,
    event_state_keys,
    out_of_fold_event_predictions,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ongoing_pollution_hours": [2.0, 8.0, 4.0, 10.0],
            "aqi_change_3": [-30.0, -3.0, 5.0, 5.0],
            "aqi_change_24": [-40.0, -5.0, 8.0, 3.0],
            "wind_speed_ms": [4.0, 1.0, 2.0, 1.5],
            "relative_humidity_pct": [45.0, 80.0, 65.0, 70.0],
            "month_sin": [0.0, 0.0, 0.0, 0.0],
            "month_cos": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_event_state_keys_use_only_supplied_current_features():
    keys = event_state_keys(_features())

    assert keys.tolist() == [
        "early_falling_dry",
        "late_flat_humid",
        "early_rising_humid",
        "late_rising_humid",
    ]


def test_event_state_keys_require_only_the_features_used_for_state_assignment():
    keys = event_state_keys(
        _features().drop(columns=["aqi_change_24", "wind_speed_ms", "month_sin", "month_cos"])
    )

    assert keys.iloc[0] == "early_falling_dry"


def test_calibrator_shrinks_small_state_to_global_residual():
    features = _features()
    calibrator = EventResidualCalibrator(minimum_state_rows=3, shrinkage_rows=3).fit(
        features,
        base_prediction=np.array([10.0, 10.0, 10.0, 10.0]),
        actual=np.array([16.0, 14.0, 8.0, 8.0]),
    )

    corrected = calibrator.predict(np.array([10.0]), features.iloc[[0]])

    assert corrected[0] == pytest.approx(11.0)


def test_calibrator_rejects_misaligned_inputs_and_clips_negative_outputs():
    with pytest.raises(ValueError, match="aligned"):
        EventResidualCalibrator().fit(_features(), np.array([1.0]), np.array([1.0]))

    fitted = EventResidualCalibrator().fit(_features(), np.zeros(4), np.full(4, -2.0))

    assert fitted.predict(np.array([0.0]), _features().iloc[[0]]).tolist() == [0.0]


class _MeanModel:
    def fit(self, _features: pd.DataFrame, target: pd.Series):
        self.value = float(target.mean())
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.value)


class _CountingMeanModel(_MeanModel):
    fit_rows: ClassVar[list[int]] = []

    def fit(self, features: pd.DataFrame, target: pd.Series):
        self.fit_rows.append(len(target))
        return super().fit(features, target)


def test_out_of_fold_predictions_never_fit_the_predicted_event():
    frame = _features().assign(
        timestamp=pd.date_range("2024-01-01", periods=4, freq="h", tz="Asia/Shanghai"),
        event_key=["a", "a", "b", "b"],
        remaining_hours=[4.0, 3.0, 8.0, 7.0],
    )

    predictions = out_of_fold_event_predictions(
        frame,
        feature_names=list(_features().columns),
        model_factory=_MeanModel,
        minimum_train_rows=2,
    )

    assert predictions["event_key"].tolist() == ["b", "b"]
    assert predictions["base_prediction"].tolist() == [3.5, 3.5]


def test_out_of_fold_predictions_can_use_chronological_event_batches():
    _CountingMeanModel.fit_rows = []
    rows = pd.concat([_features().iloc[[index % 4]] for index in range(6)], ignore_index=True)
    frame = rows.assign(
        timestamp=pd.date_range("2024-01-01", periods=6, freq="h", tz="Asia/Shanghai"),
        event_key=["a", "b", "c", "d", "e", "f"],
        remaining_hours=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )

    predictions = out_of_fold_event_predictions(
        frame,
        feature_names=list(_features().columns),
        model_factory=_CountingMeanModel,
        minimum_train_rows=1,
        fold_count=2,
    )

    assert _CountingMeanModel.fit_rows == [2, 4]
    assert predictions["event_key"].tolist() == ["c", "d", "e", "f"]
