import numpy as np
import pandas as pd
import pytest

from haze_forecast import training
from haze_forecast.training import (
    chronological_split,
    constrain_event_predictions,
    empirical_interval,
    event_chronological_split,
    rolling_event_splits,
    rolling_time_splits,
)


def test_chronological_split_never_places_later_timestamp_in_training():
    frame = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=10, freq="h")})

    train, valid, test = chronological_split(frame)

    assert train["timestamp"].max() < valid["timestamp"].min() < test["timestamp"].min()


def test_empirical_interval_uses_nonnegative_lower_bound():
    low, high = empirical_interval(2.0, 3.0)

    assert low == 0.0
    assert high == 5.0


def test_rolling_time_splits_keep_each_validation_window_after_its_training_window():
    frame = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=12, freq="h")})

    splits = rolling_time_splits(frame, n_splits=3)

    assert len(splits) == 3
    assert all(train["timestamp"].max() < valid["timestamp"].min() for train, valid in splits)


def test_event_chronological_split_keeps_each_pollution_event_in_one_partition():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=20, freq="h"),
            "event_id": [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4,
        }
    )

    train, valid, test = event_chronological_split(frame)

    partitions = [set(partition["event_id"]) for partition in (train, valid, test)]
    assert all(left.isdisjoint(right) for index, left in enumerate(partitions) for right in partitions[index + 1 :])
    assert train["timestamp"].max() < valid["timestamp"].min() < test["timestamp"].min()


def test_rolling_event_splits_never_share_an_event_between_train_and_validation():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=24, freq="h"),
            "event_id": [event_id for event_id in range(1, 7) for _ in range(4)],
        }
    )

    splits = rolling_event_splits(frame, n_splits=3)

    assert len(splits) == 3
    assert all(set(train["event_id"]).isdisjoint(valid["event_id"]) for train, valid in splits)
    assert all(train["timestamp"].max() < valid["timestamp"].min() for train, valid in splits)


def test_event_constraint_prevents_upward_remaining_time_steps_within_one_event():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=4, freq="h"),
            "event_id": [1, 1, 1, 2],
        }
    )

    constrained = constrain_event_predictions(frame, pd.Series([5.0, 8.0, 2.0, 7.0]))

    assert constrained.tolist() == [5.0, 4.0, 2.0, 7.0]


def test_aft_bounds_represent_exact_observed_dissipation_times():
    assert hasattr(training, "aft_label_bounds")

    lower, upper = training.aft_label_bounds(pd.Series([0.0, 2.5, 8.0]))

    assert np.all(lower > 0)
    assert lower.tolist() == upper.tolist()


def test_aft_survival_regressor_returns_positive_dissipation_times():
    assert hasattr(training, "AFTSurvivalRegressor")

    features = pd.DataFrame({"aqi": [120.0, 135.0, 150.0, 165.0], "wind_speed_ms": [3.0, 2.0, 1.0, 0.5]})
    model = training.AFTSurvivalRegressor(num_boost_round=5, seed=42)
    model.fit(features, pd.Series([1.0, 3.0, 5.0, 7.0]))

    predicted = model.predict(features.iloc[:2])

    assert predicted.shape == (2,)
    assert np.all(predicted > 0)


@pytest.mark.parametrize("nthread", [None, 1])
def test_aft_survival_regressor_accepts_optional_xgboost_thread_count(nthread):
    model = training.AFTSurvivalRegressor(num_boost_round=5, seed=42, nthread=nthread)

    assert model.nthread == nthread


def test_aft_survival_regressor_accepts_sample_weights():
    features = pd.DataFrame({"aqi": [120.0, 135.0, 150.0, 165.0], "wind_speed_ms": [3.0, 2.0, 1.0, 0.5]})
    target = pd.Series([1.0, 3.0, 5.0, 7.0])

    model = training.AFTSurvivalRegressor(num_boost_round=5, seed=42)
    model.fit(features, target, sample_weight=np.array([1.0, 1.0, 1.0, 1.0]))

    assert np.all(model.predict(features) > 0)


def test_aft_survival_regressor_rejects_malformed_sample_weights():
    features = pd.DataFrame({"aqi": [120.0, 135.0, 150.0, 165.0], "wind_speed_ms": [3.0, 2.0, 1.0, 0.5]})
    target = pd.Series([1.0, 3.0, 5.0, 7.0])
    model = training.AFTSurvivalRegressor(num_boost_round=5, seed=42)

    with pytest.raises(ValueError, match="one entry per row"):
        model.fit(features, target, sample_weight=np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="finite and non-negative"):
        model.fit(features, target, sample_weight=np.array([1.0, -1.0, 1.0, 1.0]))


def test_aft_survival_regressor_returns_xgboost_hours_without_exponentiating_them():
    features = pd.DataFrame({"aqi": [120.0, 135.0, 150.0, 165.0], "wind_speed_ms": [3.0, 2.0, 1.0, 0.5]})
    model = training.AFTSurvivalRegressor(num_boost_round=5, seed=42).fit(
        features, pd.Series([1.0, 3.0, 5.0, 7.0])
    )

    raw_hours = model.booster.predict(training.xgb.DMatrix(features.astype(float)))

    assert np.allclose(model.predict(features), raw_hours)
