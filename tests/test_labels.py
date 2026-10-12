import pandas as pd
import pytest

from haze_forecast.labels import add_particle_primary, label_pollution_events


def test_label_requires_six_stable_clean_hours_after_event():
    aqi = [80, 110, 115, 120, 90, 95, 88, 89, 91, 92, 93]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=len(aqi), freq="h"),
            "aqi": aqi,
            "particle_primary": [False, True, True, True, False, False, False, False, False, False, False],
        }
    )

    labeled = label_pollution_events(frame)

    assert labeled.loc[1, "remaining_hours"] == 3
    assert labeled.loc[3, "event_id"] == 1
    assert labeled.loc[4, "event_id"] == 0


def test_label_does_not_end_an_event_on_a_short_lived_improvement():
    aqi = [110, 115, 120, 90, 91, 92, 150, 155, 160, 90, 91, 92, 93, 94, 95]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=len(aqi), freq="h"),
            "aqi": aqi,
            "particle_primary": [True, True, True, False, False, False, True, True, True, False, False, False, False, False, False],
        }
    )

    labeled = label_pollution_events(frame)

    assert labeled.loc[0, "remaining_hours"] == 9
    assert labeled.loc[6, "event_id"] == 1


def test_high_aqi_with_a_non_particle_primary_pollutant_is_not_clean():
    aqi = [110, 115, 120, 130, 131, 132, 133, 134, 135, 90, 91, 92, 93, 94, 95]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=len(aqi), freq="h"),
            "aqi": aqi,
            "particle_primary": [True, True, True, False, False, False, False, False, False, False, False, False, False, False, False],
        }
    )

    labeled = label_pollution_events(frame)

    assert labeled.loc[0, "remaining_hours"] == 9


def test_particle_dissipation_target_ends_when_particles_are_no_longer_primary():
    aqi = [110, 115, 120, 130, 131, 132, 133, 134, 135]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=len(aqi), freq="h"),
            "aqi": aqi,
            "particle_primary": [True, True, True, False, False, False, False, False, False],
        }
    )

    labeled = label_pollution_events(frame, require_aqi_recovery=False)

    assert labeled.loc[0, "remaining_hours"] == 3


def test_particle_dissipation_target_does_not_treat_missing_particle_iaqi_as_clean():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=9, freq="h"),
            "aqi": [110, 115, 120, 130, 131, 132, 133, 134, 135],
            "particle_primary": [True, True, True, False, False, False, False, False, False],
            "pm2_5_iaqi": [120, 120, 120] + [float("nan")] * 6,
            "pm10_iaqi": [120, 120, 120] + [float("nan")] * 6,
        }
    )

    labeled = label_pollution_events(frame, require_aqi_recovery=False)

    assert labeled["event_id"].eq(0).all()


def test_particle_primary_accepts_a_particle_iaqi_above_100_without_exact_aqi_match():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=24, freq="h"),
            "aqi": [160] * 24,
            "pm2_5": [118] * 24,
            "pm10": [150] * 24,
        }
    )

    result = add_particle_primary(frame)

    assert result.loc[23, "pm2_5_iaqi"] == pytest.approx(154)
    assert result.loc[23, "particle_primary"]
