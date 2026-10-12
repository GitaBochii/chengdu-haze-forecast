import numpy as np
import pandas as pd
import pytest

from haze_forecast.discrete_survival import DiscreteHazardRegressor, expand_event_hours


def test_expand_event_hours_repeats_only_current_features_and_marks_final_horizon():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h"),
            "event_id": [1, 1],
            "remaining_hours": [3.0, 1.0],
            "pm2_5": [120.0, 80.0],
        }
    )

    expanded = expand_event_hours(frame, ["pm2_5"])

    assert expanded["horizon_hours"].tolist() == [1.0, 2.0, 3.0, 1.0]
    assert expanded["event_ends_in_horizon"].tolist() == [0, 0, 1, 1]
    assert expanded["pm2_5"].tolist() == [120.0, 120.0, 120.0, 80.0]
    assert "remaining_hours" not in expanded.columns


def test_expand_event_hours_rejects_non_integer_or_forbidden_labels():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h"),
            "event_id": [1, 1],
            "remaining_hours": [2.5, 1.0],
            "pm2_5": [120.0, 80.0],
        }
    )

    with pytest.raises(ValueError, match="integer"):
        expand_event_hours(frame, ["pm2_5"])
    with pytest.raises(ValueError, match="non-feature"):
        expand_event_hours(frame.assign(remaining_hours=[2.0, 1.0]), ["remaining_hours"])


def test_discrete_hazard_predict_returns_median_or_maximum_horizon_for_right_censoring(monkeypatch):
    model = DiscreteHazardRegressor(max_horizon_hours=4)
    model.feature_names = ["pm2_5"]
    monkeypatch.setattr(
        model,
        "_predict_hazards",
        lambda values: np.array([[0.1, 0.2, 0.8, 0.1], [0.1, 0.1, 0.1, 0.1]]),
    )

    prediction, censored = model.predict_with_censoring(pd.DataFrame({"pm2_5": [100.0, 80.0]}))

    assert prediction.tolist() == [3.0, 4.0]
    assert censored.tolist() == [False, True]


def test_discrete_hazard_requires_fit_before_prediction():
    model = DiscreteHazardRegressor(max_horizon_hours=4)

    with pytest.raises(ValueError, match="fitted"):
        model.predict(pd.DataFrame({"pm2_5": [100.0]}))
