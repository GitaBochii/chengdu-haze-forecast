import pandas as pd
import pytest

from haze_forecast.neighbor_city_data import parse_city_day, required_days


def test_required_days_uses_beijing_calendar_after_lag():
    origins = pd.Series(pd.to_datetime(['2022-01-02T17:00:00Z']))
    assert required_days(origins) == ['20220101']
    with pytest.raises(ValueError):
        required_days(pd.Series(pd.to_datetime(['2022-01-01'])))


def test_parser_preserves_missing_hours_and_cities():
    raw = b'date,hour,type,Chengdu\n20220101,0,PM2.5,12\n20220101,0,PM2.5_24h,99\n'
    result = parse_city_day(raw, '20220101', ['Chengdu', 'Missing'])
    assert len(result) == 24
    assert result.loc[0, 'Chengdu_PM2.5'] == 12
    assert result['Missing_PM2.5'].isna().all()
    assert result['Chengdu_PM2.5'].notna().sum() == 1
    assert str(result.timestamp.dt.tz) == 'Asia/Shanghai'


def test_parser_rejects_duplicate_and_wrong_dates():
    raw = b'date,hour,type,C\n20220101,0,AQI,1\n20220101,0,AQI,2\n'
    with pytest.raises(ValueError):
        parse_city_day(raw, '20220101', ['C'])
    with pytest.raises(ValueError):
        parse_city_day(raw, '20220102', ['C'])
