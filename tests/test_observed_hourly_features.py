import importlib

import numpy as np
import pandas as pd
import pytest

from haze_forecast.features import build_features
from haze_forecast.labels import add_particle_primary


def build(frame):
    module = importlib.import_module("haze_forecast.observed_hourly_features")
    return module.build_observed_hourly_features(frame)


def measurements():
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=60, freq="h", tz="Asia/Shanghai"),
        "aqi": np.arange(60, dtype=float) + 110,
        "pm2_5": np.full(60, 120.0),
        "pm10": np.full(60, 200.0),
    })


def test_lag_is_elapsed_hours_and_grid_not_imputed():
    frame = measurements().drop(index=5)
    result = build(frame)
    assert len(result) == 60
    assert np.isnan(result.loc[5, "aqi"])
    assert not result.loc[5, "input_hour_observed"]
    assert result.loc[6, "input_hour_observed"]
    assert result.loc[24, "aqi_lag_24"] == 110
    assert np.isnan(result.loc[29, "aqi_lag_24"])
    assert result.loc[30, "aqi_lag_24"] == 116


def test_rolling_excludes_current_hour():
    frame = measurements()
    frame.loc[30, "aqi"] = 9999
    result = build(frame)
    assert result.loc[30, "aqi_roll_3"] == 138
    assert result.loc[30, "aqi_roll_24"] == pytest.approx(127.5)


def test_particle_window_requires_24_observed_hours():
    result = build(measurements().drop(index=5))
    assert np.isnan(result.loc[24, "pm2_5_iaqi"])
    assert not result.loc[24, "particle_primary"]
    assert result.loc[29, "particle_primary"]


def test_future_measurements_cannot_change_past_features():
    frame = measurements()
    expected = build(frame).iloc[:35]
    frame.loc[35:, ["aqi", "pm2_5", "pm10"]] = 0
    pd.testing.assert_frame_equal(expected, build(frame).iloc[:35])


def test_continuous_features_equal_original_and_input_unchanged():
    frame = measurements()
    original = frame.copy(deep=True)
    result = build(frame)
    expected = build_features(add_particle_primary(frame))
    pd.testing.assert_frame_equal(result[expected.columns], expected)
    pd.testing.assert_frame_equal(frame, original)
    assert result.input_hour_observed.all()


def test_discards_labels_and_derived_inputs():
    frame = measurements()
    expected = build(frame)
    for name in ["event_id", "remaining_hours", "particle_primary", "aqi_lag_24", "future_target"]:
        frame[name] = 999
    pd.testing.assert_frame_equal(build(frame), expected)


@pytest.mark.parametrize("kind", ["duplicate", "naive", "nonhour", "missing_time", "inf", "negative_inf", "bad_numeric", "empty", "missing_measurement", "duplicate_column"])
def test_rejects_invalid_input(kind):
    frame = measurements()
    if kind == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif kind == "naive":
        frame["timestamp"] = frame.timestamp.dt.tz_localize(None)
    elif kind == "nonhour":
        frame["timestamp"] += pd.Timedelta(minutes=1)
    elif kind == "missing_time":
        frame.loc[0, "timestamp"] = pd.NaT
    elif kind in {"inf", "negative_inf"}:
        frame.loc[0, "aqi"] = np.inf if kind == "inf" else -np.inf
    elif kind == "bad_numeric":
        frame["aqi"] = "bad"
    elif kind == "empty":
        frame = frame.iloc[:0]
    elif kind == "missing_measurement":
        frame = frame.drop(columns="pm10")
    elif kind == "duplicate_column":
        frame = pd.concat([frame, frame[["aqi"]]], axis=1)
    with pytest.raises(ValueError):
        build(frame)


def test_missing_aqi_stays_unknown_and_does_not_imply_clean():
    frame = measurements()
    frame.loc[30, "aqi"] = np.nan
    result = build(frame)
    assert np.isnan(result.loc[30, "aqi"])
    assert result.loc[30, "input_hour_observed"]
    assert "remaining_hours" not in result


def test_unsorted_input_is_sorted_without_changing_values():
    frame = measurements()
    pd.testing.assert_frame_equal(build(frame.iloc[::-1]), build(frame))
