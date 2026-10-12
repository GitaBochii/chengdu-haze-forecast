import pandas as pd
import pytest

from haze_forecast.regional_air_history import (
    aggregate_history_city_hourly,
    complete_history_hours,
    normalize_history_interim,
    parse_history_daily,
)


@pytest.fixture
def metadata():
    return pd.DataFrame({'station_id': ['A', 'B'], 'city': ['chengdu'] * 2,
                         'longitude': [104., 105.], 'latitude': [30., 31.]})


def test_exact_ozone_type_and_legacy_interface_unchanged(metadata):
    raw = pd.DataFrame({'date': [20220101] * 4, 'hour': [0] * 4,
                        'type': ['O3', 'O3_8h', 'O3_8h_24h', 'AQI'],
                        'A': [900, 80, 700, 20], 'B': [901, 100, 701, None]})
    result = parse_history_daily(raw, metadata)
    assert result.o3_8h.tolist() == [80, 100]
    assert 'o3' not in result
    from haze_forecast.regional_air_data import parse_station_daily
    assert parse_station_daily(raw, metadata).o3.tolist() == [900, 901]
    city = aggregate_history_city_hourly(result)
    assert city.o3_8h.iloc[0] == 90
    assert city.aqi_station_count.iloc[0] == 1


def test_interim_ozone_and_cross_year_gaps(metadata):
    raw = pd.DataFrame({'timestamp': ['2021-12-31 23:00'], 'station_id': ['A'],
                        'aqi': [30], 'pm2_5': [15], 'pm10': [20], 'so2': [1],
                        'no2': [2], 'co': [0.5], 'o3': [80]})
    frame = normalize_history_interim(raw, metadata)
    result = complete_history_hours(frame, metadata, '2021-12-31', '2022-01-01')
    assert len(result) == 96
    assert str(result.timestamp.dt.tz) == 'Asia/Shanghai'
    assert result.o3_8h.notna().sum() == 1
    assert result.aqi.notna().sum() == 1
    assert result.longitude.notna().all()
    assert result.node_id.equals(result.station_id)


def test_duplicate_observations_fail_even_when_equal(metadata):
    raw = pd.DataFrame({'date': [20220101] * 2, 'hour': [0] * 2,
                        'type': ['O3_8h'] * 2, 'A': [80, 80]})
    with pytest.raises(ValueError, match='duplicate'):
        parse_history_daily(raw, metadata)
    frame = parse_history_daily(raw.iloc[:1], metadata)
    with pytest.raises(ValueError, match='duplicate'):
        complete_history_hours(pd.concat([frame, frame]), metadata, '2022-01-01', '2022-01-01')


def test_missing_eight_hour_ozone_not_replaced_by_hourly(metadata):
    raw = pd.DataFrame({'date': [20220101], 'hour': [0], 'type': ['O3'], 'A': [900]})
    assert parse_history_daily(raw, metadata).o3_8h.isna().all()


def test_unknown_station_is_not_silently_dropped(metadata):
    raw = pd.DataFrame({'timestamp': ['2021-01-01'], 'station_id': ['unknown'],
                        **{key: [1.] for key in ['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3']}})
    with pytest.raises(ValueError, match='unknown station'):
        normalize_history_interim(raw, metadata)


def test_outside_grid_fails_instead_of_clipping(metadata):
    raw = pd.DataFrame({'date': [20220101], 'hour': [0], 'type': ['AQI'], 'A': [30]})
    with pytest.raises(ValueError, match='outside'):
        complete_history_hours(parse_history_daily(raw, metadata), metadata, '2023-01-01', '2023-01-01')


def test_source_reader_records_hash_and_exact_ozone(tmp_path, metadata):
    import hashlib

    from scripts.build_regional_air_history import read_daily_source
    payload = b'date,hour,type,A\n20220101,0,O3,900\n20220101,0,O3_8h,80\n'
    frame, report = read_daily_source(payload, 'china_sites_20220101.csv', metadata)
    assert frame.o3_8h.dropna().tolist() == [80]
    assert report['sha256'] == hashlib.sha256(payload).hexdigest()
    assert report['ozone_type_present'] is True
    assert report['observed_hours']['o3_8h'] == 1


def test_baseline_comparison_rejects_changed_values(tmp_path):
    from scripts.build_regional_air_history import compare_city_baseline
    frame = pd.DataFrame({'timestamp': pd.date_range('2022-01-01', periods=2, freq='h', tz='Asia/Shanghai'),
                          'aqi': [30., None], 'pm2_5': [15., None], 'pm10': [25., None]})
    path = tmp_path / 'baseline.parquet'
    frame.to_parquet(path, index=False)
    assert compare_city_baseline(frame, path)['equal'] is True
    changed = frame.copy()
    changed.loc[0, 'aqi'] = 31.
    with pytest.raises(AssertionError):
        compare_city_baseline(changed, path)


def test_output_guard_allows_weather_but_refuses_panel_overwrite(tmp_path):
    from scripts.build_regional_air_history import guard_outputs
    output, audit = tmp_path / 'output', tmp_path / 'audit'
    output.mkdir()
    (output / 'weather_proxy.parquet').touch()
    guard_outputs(output, audit)
    (output / 'city_hourly.parquet').touch()
    with pytest.raises(FileExistsError, match='city_hourly'):
        guard_outputs(output, audit)
