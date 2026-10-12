import numpy as np
import pandas as pd
import pytest

from haze_forecast.observed_recovery_labels import observed_recovery_bounds


def frame(values):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-01-01", periods=len(values), freq="h", tz="Asia/Shanghai"
            ),
            "aqi": values,
        }
    )


def test_exact_timestamp_duration_and_availability():
    data = frame([150] * 7 + [80] * 6)
    result = observed_recovery_bounds(data, data.timestamp.iloc[:3])
    assert result.exact_hours.tolist() == [7, 6, 5]
    assert result.lower_bound_hours.equals(result.upper_bound_hours)
    assert result.label_available_at.eq(data.timestamp.iloc[-1]).all()
    assert result.reason.eq("exact").all()


@pytest.mark.parametrize("explicit", [False, True])
def test_two_hour_gap_stops_observation(explicit):
    data = frame([150] * 8 + [80] * 8)
    gap = data.timestamp.iloc[8]
    if explicit:
        data.loc[8:9, "aqi"] = np.nan
    else:
        data = data.drop(index=[8, 9])
    row = observed_recovery_bounds(data, data.timestamp.iloc[:1]).iloc[0]
    assert row.observed_prefix_hours == 7
    assert row.lower_bound_hours == 3
    assert np.isinf(row.upper_bound_hours)
    assert np.isnan(row.exact_hours)
    assert row.informative
    assert row.reason == "gap"
    assert row.label_available_at == gap


def test_long_gap_hides_apparent_recovery():
    data = frame([150] * 4 + [80] * 20).drop(index=range(4, 17))
    row = observed_recovery_bounds(data, data.timestamp.iloc[:1]).iloc[0]
    assert row.reason == "gap"
    assert row.lower_bound_hours == 1
    assert not row.informative
    assert row.observed_prefix_hours == 3


@pytest.mark.parametrize("count", range(6))
def test_short_prefix_not_informative(count):
    data = frame([150] + [80] * count)
    row = observed_recovery_bounds(data, data.timestamp.iloc[:1]).iloc[0]
    assert not row.informative
    assert row.lower_bound_hours == 1
    assert row.reason == "end_of_data"
    assert row.label_available_at == data.timestamp.iloc[-1]


def test_exact_followed_by_missing_still_exact():
    data = frame([150] + [80] * 6 + [np.nan, 150])
    row = observed_recovery_bounds(data, data.timestamp.iloc[:1]).iloc[0]
    assert row.exact_hours == 1
    assert row.informative
    assert row.label_available_at == data.timestamp.iloc[6]
    assert row.observed_prefix_hours == 6


def test_end_of_data_censoring_and_origin_at_end():
    data = frame([150] * 10)
    rows = observed_recovery_bounds(data, data.timestamp.iloc[[0, 9]])
    assert rows.lower_bound_hours.tolist() == [5, 1]
    assert rows.observed_prefix_hours.tolist() == [9, 0]
    assert rows.label_available_at.eq(data.timestamp.iloc[-1]).all()


def test_sorted_input_preserves_origin_order_and_does_not_mutate():
    data = frame([150] * 3 + [80] * 6)
    original = data.copy(deep=True)
    rows = observed_recovery_bounds(data.iloc[::-1], data.timestamp.iloc[[1, 0]])
    assert rows.exact_hours.tolist() == [2, 3]
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("bad", ["duplicate", "naive", "minute", "nat", "infinity", "empty"])
def test_invalid_frame_rejected(bad):
    data = frame([150] * 8)
    origins = data.timestamp.iloc[:1].copy()
    if bad == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    elif bad == "naive":
        data["timestamp"] = data.timestamp.dt.tz_localize(None)
    elif bad == "minute":
        data.loc[0, "timestamp"] += pd.Timedelta(minutes=1)
    elif bad == "nat":
        data.loc[0, "timestamp"] = pd.NaT
    elif bad == "infinity":
        data.loc[0, "aqi"] = np.inf
    else:
        data = data.iloc[:0]
    with pytest.raises(ValueError):
        observed_recovery_bounds(data, origins)


@pytest.mark.parametrize("bad", ["duplicate", "naive", "minute", "outside", "empty"])
def test_invalid_origins_rejected(bad):
    data = frame([150] * 8)
    origin = data.timestamp.iloc[0]
    origins = {
        "duplicate": [origin, origin],
        "naive": [origin.tz_localize(None)],
        "minute": [origin + pd.Timedelta(minutes=1)],
        "outside": [origin - pd.Timedelta(hours=1)],
        "empty": [],
    }[bad]
    with pytest.raises(ValueError):
        observed_recovery_bounds(data, origins)


@pytest.mark.parametrize("stable", [0, -1, 1.5, True])
def test_invalid_stability_rejected(stable):
    data = frame([150, 80])
    with pytest.raises(ValueError):
        observed_recovery_bounds(data, data.timestamp.iloc[:1], stable_hours=stable)


def test_custom_stability_and_threshold_inclusive():
    data = frame([150, 100, 100])
    row = observed_recovery_bounds(data, data.timestamp.iloc[:1], stable_hours=2).iloc[0]
    assert row.exact_hours == 1
    assert row.label_available_at == data.timestamp.iloc[2]


def test_contiguous_pollution_agrees_with_legacy_labels():
    from haze_forecast.labels import label_pollution_events

    data = frame([150] * 9 + [80] * 6)
    data["particle_primary"] = data.aqi.gt(100)
    old = label_pollution_events(data)
    origins = old.loc[old.remaining_hours.notna(), "timestamp"]
    result = observed_recovery_bounds(data, origins)
    assert result.exact_hours.tolist() == old.remaining_hours.dropna().tolist()


def test_origin_inside_missing_grid_hour_is_allowed():
    data = frame([150, np.nan] + [80] * 6)
    origin = data.timestamp.iloc[1]
    result = observed_recovery_bounds(data.drop(index=1), [origin]).iloc[0]
    assert result.exact_hours == 1


def test_matches_naive_scan_randomized():
    rng = np.random.default_rng(42)
    for _ in range(20):
        values = rng.choice([80.0, 100.0, 150.0, np.nan], size=50)
        data = frame(values)
        stable = int(rng.integers(1, 7))
        result = observed_recovery_bounds(data, data.timestamp, stable_hours=stable)
        for position, row in result.iterrows():
            boundary = position + 1
            while boundary < len(values) and not np.isnan(values[boundary]):
                boundary += 1
            candidate = None
            for start in range(position + 1, boundary - stable + 1):
                if np.all(values[start : start + stable] <= 100):
                    candidate = start
                    break
            assert row.observed_prefix_hours == boundary - position - 1
            if candidate is not None:
                assert row.exact_hours == candidate - position
                assert row.reason == "exact"
                assert row.label_available_at == data.timestamp.iloc[candidate + stable - 1]
            else:
                assert np.isnan(row.exact_hours)
                assert row.lower_bound_hours == max(0, boundary - position - stable) + 1
                assert row.reason == ("gap" if boundary < len(values) else "end_of_data")
