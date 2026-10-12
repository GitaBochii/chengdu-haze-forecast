from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from haze_forecast.gefs_features import select_reforecast_run_and_lead, summarise_ensemble


def test_reforecast_selection_uses_daily_run_and_keeps_target_weather_in_future():
    issue_time = datetime(2015, 1, 1, 16, tzinfo=UTC)

    run_time, lead_hours = select_reforecast_run_and_lead(issue_time, horizon_hours=6)

    assert run_time == datetime(2015, 1, 1, tzinfo=UTC)
    assert lead_hours == 24
    assert run_time + timedelta(hours=lead_hours) >= issue_time + timedelta(hours=6)


def test_reforecast_selection_rejects_out_of_coverage_or_unsupported_horizon():
    with pytest.raises(ValueError):
        select_reforecast_run_and_lead(datetime(2020, 1, 1, tzinfo=UTC), horizon_hours=6)
    with pytest.raises(ValueError):
        select_reforecast_run_and_lead(datetime(2015, 1, 1, tzinfo=UTC), horizon_hours=0)


def test_ensemble_summary_reports_distribution_without_member_order_dependence():
    summary = summarise_ensemble([5.0, 1.0, 7.0, 3.0, 9.0])

    assert summary["median"] == 5.0
    assert summary["std"] == pytest.approx(np.std([5.0, 1.0, 7.0, 3.0, 9.0], ddof=0))
    assert summary["p10"] == pytest.approx(1.8)
    assert summary["p90"] == pytest.approx(8.2)


def test_ensemble_summary_rejects_incomplete_member_vector():
    with pytest.raises(ValueError, match="five"):
        summarise_ensemble([1.0, 2.0, 3.0, 4.0])
