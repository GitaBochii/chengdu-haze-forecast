import numpy as np
import pandas as pd

from haze_forecast.features import (
    add_missingness_indicators,
    add_pollution_trajectory_features,
    build_features,
    snapshot_feature_names,
)


def test_rolling_feature_excludes_current_and_future_observations():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=4, freq="h"),
            "pm2_5": [10.0, 20.0, 30.0, 1000.0],
            "wind_u_ms": [1.0] * 4,
            "wind_v_ms": [0.0] * 4,
            "remaining_hours": [3.0, 2.0, 1.0, 0.0],
        }
    )

    result = build_features(frame)

    assert result.loc[2, "pm2_5_lag_1"] == 20.0
    assert result.loc[2, "pm2_5_roll_3"] == 15.0


def test_pollution_duration_uses_only_the_current_and_previous_hours():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=4, freq="h"),
            "aqi": [110, 120, 130, 90],
            "particle_primary": [True, True, True, False],
        }
    )

    result = build_features(frame)

    assert result["ongoing_pollution_hours"].tolist() == [1.0, 2.0, 3.0, 0.0]


def test_trajectory_features_reset_after_clean_or_nonconsecutive_hour():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2021-01-01 00:00",
                    "2021-01-01 01:00",
                    "2021-01-01 02:00",
                    "2021-01-01 04:00",
                ]
            ),
            "aqi": [120.0, 160.0, 130.0, 140.0],
            "pm2_5": [80.0, 110.0, 90.0, 100.0],
            "pm10": [160.0, 190.0, 170.0, 180.0],
            "particle_primary": [True, True, True, True],
        }
    )

    result = add_pollution_trajectory_features(frame)

    assert result["pollution_age_hours"].tolist() == [1.0, 2.0, 3.0, 1.0]
    assert result["event_peak_aqi"].tolist() == [120.0, 160.0, 160.0, 140.0]
    assert result["aqi_from_event_peak"].tolist() == [0.0, 0.0, -30.0, 0.0]
    assert result["aqi_cumulative_excess"].tolist() == [20.0, 80.0, 110.0, 40.0]


def test_snapshot_features_exclude_all_history_dependent_columns():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=2, freq="h"),
            "aqi": [120.0, 130.0],
            "pm2_5": [80.0, 90.0],
            "wind_u_ms": [1.0, 1.0],
            "wind_v_ms": [0.0, 0.0],
            "remaining_hours": [2.0, 1.0],
        }
    )

    names = snapshot_feature_names(build_features(frame))

    assert {"aqi", "pm2_5", "hour_sin", "wind_speed_ms"}.issubset(names)
    assert not any("_lag_" in name or "_roll_" in name or "_change_" in name for name in names)


def test_add_missingness_indicators_preserves_source_values_and_marks_missing_rows():
    frame = pd.DataFrame({"boundary_layer_height_m": [250.0, np.nan]})

    result = add_missingness_indicators(frame, ["boundary_layer_height_m"])

    assert result["boundary_layer_height_m"].tolist()[0] == 250.0
    assert result["boundary_layer_height_m_is_missing"].tolist() == [0.0, 1.0]
