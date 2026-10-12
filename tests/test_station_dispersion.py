import numpy as np
import pandas as pd
import pytest

from haze_forecast.station_dispersion import station_dispersion


def test_distribution_excludes_missing_and_preserves_threshold():
    data = pd.DataFrame({'timestamp': pd.to_datetime(['2020-01-01']*4, utc=True),
                         'node_id': ['a', 'b', 'c', 'd'], 'aqi': [80., 100., 120., np.nan]})
    result = station_dispersion(data).iloc[0]
    assert result.station_aqi_q25_offset == -10
    assert result.station_aqi_q75_offset == 10
    assert result.station_aqi_polluted_fraction == 1/3
    assert result.station_aqi_count == 3


def test_missing_and_single_station_are_not_deleted():
    times = pd.to_datetime(['2020-01-01', '2020-01-02'], utc=True)
    result = station_dispersion(pd.DataFrame({'timestamp': times, 'node_id': ['a', 'a'], 'aqi': [np.nan, 105.]}))
    assert len(result) == 2
    assert result.iloc[0].station_aqi_count == 0
    assert np.isnan(result.iloc[0].station_aqi_polluted_fraction)
    assert result.iloc[1].station_aqi_q75_offset == 0


def test_future_stations_cannot_change_past():
    data = pd.DataFrame({'timestamp': pd.to_datetime(['2020-01-01','2020-01-02'], utc=True),
                         'node_id': ['a','b'], 'aqi': [101., 900.]})
    pd.testing.assert_frame_equal(station_dispersion(data.iloc[:1]), station_dispersion(data).iloc[:1])


def test_duplicates_rejected():
    data = pd.DataFrame({'timestamp': pd.to_datetime(['2020-01-01']*2, utc=True),
                         'node_id': ['a','a'], 'aqi': [80., 90.]})
    with pytest.raises(ValueError):
        station_dispersion(data)
