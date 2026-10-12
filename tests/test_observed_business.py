import numpy as np
import pandas as pd

from haze_forecast.observed_business import build_business_states, build_observed_business


def state_frame(aqi, pm25=None, pm10=None):
    n = len(aqi)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="h", tz="Asia/Shanghai"),
            "aqi": aqi,
            "pm2_5_iaqi": [150.0] * n if pm25 is None else pm25,
            "pm10_iaqi": [150.0] * n if pm10 is None else pm10,
        }
    )


def test_confirmation_is_not_backfilled():
    result = build_business_states(state_frame([150.0] * 5))
    assert result.eligible.tolist() == [False, False, True, True, True]
    assert result.episode_id.tolist() == [0, 0, 1, 1, 1]


def test_six_clean_closes_and_block_changes_next_hour_once():
    result = build_business_states(state_frame([150.0] * 3 + [50.0] * 8 + [150.0] * 3))
    assert result.episode_id.iloc[7] == 1
    assert result.episode_id.iloc[8] == 0
    assert result.continuity_block.iloc[8] == 1
    assert result.continuity_block.iloc[9] == 2
    assert result.continuity_block.iloc[-1] == 2
    assert result.episode_id.iloc[-1] == 2


def test_gap_breaks_episode_not_continuity_block():
    result = build_business_states(state_frame([150.0] * 3 + [np.nan] + [150.0] * 3))
    assert result.episode_id.tolist() == [0, 0, 1, 0, 0, 0, 2]
    assert result.continuity_block.nunique() == 1
    assert result.pollution_state.iloc[3] == "unknown"


def test_three_valued_logic_and_pm_unknown_does_not_close():
    result = build_business_states(
        state_frame(
            [150.0] * 7 + [50.0],
            [150.0, 150.0, 150.0, np.nan, np.nan, 50.0, 50.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 150.0, 50.0, np.nan, np.nan],
        )
    )
    assert result.pollution_state.tolist() == ["confirmed_pollution"] * 3 + [
        "unknown",
        "confirmed_pollution",
        "confirmed_not_pollution",
        "unknown",
        "confirmed_not_pollution",
    ]
    assert result.episode_id.iloc[3:8].eq(1).all()
    assert not result.eligible.iloc[3]


def test_prefix_invariance():
    frame = state_frame([150.0] * 3 + [50.0] * 5 + [150.0] * 3 + [np.nan] + [150.0] * 4)
    full = build_business_states(frame)
    for n in range(1, len(frame)):
        pd.testing.assert_frame_equal(build_business_states(frame.iloc[:n]), full.iloc[:n])


def test_dataset_zero_eligible_and_source_grid():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=30, freq="h", tz="Asia/Shanghai"),
            "aqi": 50.0,
            "pm2_5": 20.0,
            "pm10": 30.0,
        }
    )
    full, origins = build_observed_business({"a": frame.iloc[:15], "b": frame.iloc[16:]}, [])
    assert len(full) == 30 and origins.empty
    assert not full.input_hour_observed.iloc[15]
    assert pd.isna(full.source.iloc[15])
    assert "lower_bound_hours" in origins


def test_dataset_retains_censored_tail_and_measurement_only():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=40, freq="h", tz="Asia/Shanghai"),
            "aqi": 150.0,
            "pm2_5": 150.0,
            "pm10": 200.0,
            "event_id": 999,
            "remaining_hours": 0.0,
        }
    )
    full, origins = build_observed_business({"a": frame.iloc[:27], "b": frame.iloc[27:]}, [])
    assert len(origins) == 15
    assert origins.exact_hours.isna().all()
    assert not origins.informative.iloc[-1]
    assert full.eligible.iloc[27]
    assert "remaining_hours" not in full and "event_id" not in full


def test_duplicate_sources_rejected():
    import pytest

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="h", tz="Asia/Shanghai"),
            "aqi": 150.0,
            "pm2_5": 150.0,
            "pm10": 200.0,
        }
    )
    with pytest.raises(ValueError, match="unique"):
        build_observed_business({"a": frame, "b": frame}, [])
