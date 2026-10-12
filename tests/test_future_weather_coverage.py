import numpy as np
import pandas as pd
import pytest

from scripts.audit_future_weather_coverage import align_coverage


def sample():
    issue = pd.Timestamp('2023-01-01T04:00:00Z')
    origins = pd.DataFrame({'timestamp': [issue, issue + pd.Timedelta(hours=6)]})
    forecast = pd.DataFrame({'timestamp': [issue], 'gefs_run_utc': [issue - pd.Timedelta(hours=4)],
                             'gefs_forecast_valid_utc_f006': [issue + pd.Timedelta(hours=2)],
                             'gefs_forecast_valid_utc_f024': [issue + pd.Timedelta(hours=20)],
                             'f006': [np.nan], 'f024': [4.]})
    return origins, forecast


def test_exact_origin_join_preserves_missing_and_actual_lead():
    origins, forecasts = sample()
    result = align_coverage(origins, forecasts, {6: ['f006'], 24: ['f024']})
    assert len(result) == 2
    assert result.matched.tolist() == [True, False]
    assert result.available_6.tolist() == [False, False]
    assert result.available_24.tolist() == [True, False]
    assert result.ahead_6.iloc[0] == 2
    assert result.ahead_24.iloc[0] == 20


def test_duplicate_forecast_origin_rejected():
    origins, forecasts = sample()
    with pytest.raises(ValueError):
        align_coverage(origins, pd.concat([forecasts, forecasts]), {6: ['f006'], 24: ['f024']})
