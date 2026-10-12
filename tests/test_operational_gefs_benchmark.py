from haze_forecast.operational_gefs_benchmark import (
    extracted_keys_are_complete,
    select_2023_variant,
    should_accept_for_deployment,
)


def test_extraction_completeness_requires_every_expected_event_timestamp():
    expected = {(1, "2024-01-01T00:00:00+00:00"), (1, "2024-01-01T01:00:00+00:00")}
    observed = {(1, "2024-01-01T00:00:00+00:00")}

    assert extracted_keys_are_complete(expected, observed) is False


def test_selects_the_lowest_2023_row_mae_without_seeing_future_years():
    assert select_2023_variant({"aft": 38.4, "mean_6": 37.9, "ensemble_6_24": 36.2}) == "ensemble_6_24"


def test_rejects_when_either_confirmation_year_or_event_mae_does_not_improve():
    metrics = {
        2024: {"row_improved": True, "event_improved": True},
        2025: {"row_improved": True, "event_improved": False},
    }

    assert should_accept_for_deployment(metrics) is False
