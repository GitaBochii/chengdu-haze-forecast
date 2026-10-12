import numpy as np
import pandas as pd
import pytest


def run_features(frame):
    from haze_forecast.observed_pollution_run import build_observed_pollution_run

    return build_observed_pollution_run(frame)


def hours(aqi, pm25=None, pm10=None):
    count = len(aqi)
    return pd.DataFrame({
        'timestamp': pd.date_range('2018-01-01', periods=count, freq='h', tz='Asia/Shanghai'),
        'aqi': np.asarray(aqi, dtype=float),
        'pm2_5_iaqi': np.full(count, 150.) if pm25 is None else np.asarray(pm25, dtype=float),
        'pm10_iaqi': np.full(count, 150.) if pm10 is None else np.asarray(pm10, dtype=float),
    })


def test_left_boundary_unknown_and_observed_start():
    frame = hours([150., 150., np.nan, 150., 150., 50., 150., 150., 50.])
    result = run_features(frame)
    np.testing.assert_array_equal(result.observed_particle_run_hours,
                                  [1., 2., np.nan, 1., 2., 0., 1., 2., 0.])
    np.testing.assert_array_equal(result.particle_run_start_observed,
                                  [0., 0., np.nan, 0., 0., np.nan, 1., 1., np.nan])


def test_unknown_after_clean_does_not_confirm_a_new_start():
    frame = hours([50., np.nan, 150., 150.])
    result = run_features(frame)
    np.testing.assert_array_equal(result.observed_particle_run_hours, [0., np.nan, 1., 2.])
    np.testing.assert_array_equal(result.particle_run_start_observed, [np.nan, np.nan, 0., 0.])


def test_single_observed_particle_can_confirm_pollution():
    frame = hours([50., 150., 150., 150., 150.],
                  [np.nan, 150., np.nan, 50., 150.],
                  [np.nan, np.nan, 150., np.nan, np.nan])
    result = run_features(frame)
    np.testing.assert_array_equal(result.observed_particle_run_hours, [0., 1., 2., np.nan, 1.])
    np.testing.assert_array_equal(result.particle_run_start_observed, [np.nan, 1., 1., np.nan, 0.])


def test_threshold_100_and_confirmed_nonparticle_high_aqi():
    frame = hours([100., 150., 150., 150., 150., 150.],
                  [150., 100., 100., 100.01, np.nan, 150.],
                  [150., 100., np.nan, 100., np.nan, np.nan])
    result = run_features(frame)
    np.testing.assert_array_equal(result.observed_particle_run_hours, [0., 0., np.nan, 1., np.nan, 1.])
    np.testing.assert_array_equal(result.particle_run_start_observed, [np.nan, np.nan, np.nan, 0., np.nan, 0.])
    certain = run_features(hours([150., 150.], [100., 101.], [100., np.nan]))
    np.testing.assert_array_equal(certain.observed_particle_run_hours, [0., 1.])
    np.testing.assert_array_equal(certain.particle_run_start_observed, [np.nan, 1.])


@pytest.mark.parametrize('observed_start', [False, True])
def test_long_runs_are_not_capped_at_24_or_168_hours(observed_start):
    prefix = [50.] if observed_start else []
    result = run_features(hours(prefix + [150.] * 240))
    pollution = result.iloc[len(prefix):]
    np.testing.assert_array_equal(pollution.observed_particle_run_hours, np.arange(1., 241.))
    np.testing.assert_array_equal(pollution.particle_run_start_observed, np.full(240, float(observed_start)))


def test_every_prefix_is_unchanged():
    frame = hours([150.] * 4 + [np.nan] + [150.] * 3 + [50.] + [150.] * 5)
    complete = run_features(frame)
    for length in range(1, len(frame)):
        pd.testing.assert_frame_equal(run_features(frame.iloc[:length]), complete.iloc[:length])


def test_changing_future_measurements_cannot_change_prior_features():
    frame = hours([50., 150., 150., 150., np.nan, 50., 150., 150.])
    before = run_features(frame)
    changed = frame.copy()
    changed.loc[4:, ['aqi', 'pm2_5_iaqi', 'pm10_iaqi']] = 500.
    pd.testing.assert_frame_equal(before.iloc[:4], run_features(changed).iloc[:4])


def test_ignores_future_labels_and_preserves_source_index_without_mutation():
    frame = hours([50., 150., 150., np.nan])
    frame.index = [9, 2, 8, 3]
    expected = run_features(frame)
    contaminated = frame.assign(
        remaining_hours=[np.inf, -np.inf, 9999., 0.],
        lower_bound_hours='future label must not be parsed',
        continuity_block=9999, episode_id=9999, pollution_state='confirmed_pollution',
        observed_particle_run_hours=9999, particle_run_start_observed=9999,
    )
    original = contaminated.copy(deep=True)
    actual = run_features(contaminated)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(contaminated, original)
    assert actual.index.equals(frame.index)
    assert list(actual.columns) == ['timestamp', 'observed_particle_run_hours', 'particle_run_start_observed']
    assert actual.observed_particle_run_hours.dtype == np.dtype(float)
    assert actual.particle_run_start_observed.dtype == np.dtype(float)


@pytest.mark.parametrize('column', ['aqi', 'pm2_5_iaqi', 'pm10_iaqi'])
def test_nullable_numeric_missing_values_remain_unknown_when_insufficient(column):
    frame = hours([150., 150.], [150., 50.], [50., 50.])
    frame[column] = frame[column].astype('Float64')
    frame.loc[1, column] = pd.NA
    result = run_features(frame)
    assert pd.isna(result.observed_particle_run_hours.iloc[1])
    assert pd.isna(result.particle_run_start_observed.iloc[1])


@pytest.mark.parametrize('invalid_case', [
    'not_frame', 'empty', 'missing_column', 'duplicate_columns', 'duplicate_timestamp',
    'unsorted', 'gap', 'naive', 'missing_timestamp', 'non_hour', 'subsecond',
    'string_timestamp', 'nonnumeric', 'numeric_strings', 'infinite', 'negative_infinite',
    'complex', 'boolean',
])
def test_rejects_invalid_inputs_without_repairing_or_sorting(invalid_case):
    frame = hours([150., 150., 150.])
    if invalid_case == 'not_frame':
        frame = []
    elif invalid_case == 'empty':
        frame = frame.iloc[:0]
    elif invalid_case == 'missing_column':
        frame = frame.drop(columns='pm10_iaqi')
    elif invalid_case == 'duplicate_columns':
        frame = pd.concat([frame, frame[['aqi']]], axis=1)
    elif invalid_case == 'duplicate_timestamp':
        frame.loc[1, 'timestamp'] = frame.timestamp.iloc[0]
    elif invalid_case == 'unsorted':
        frame = frame.iloc[[1, 0, 2]]
    elif invalid_case == 'gap':
        frame = frame.iloc[[0, 2]]
    elif invalid_case == 'naive':
        frame['timestamp'] = frame.timestamp.dt.tz_localize(None)
    elif invalid_case == 'missing_timestamp':
        frame.loc[1, 'timestamp'] = pd.NaT
    elif invalid_case == 'non_hour':
        frame['timestamp'] = pd.date_range('2018-01-01 00:30', periods=3, freq='h', tz='Asia/Shanghai')
    elif invalid_case == 'subsecond':
        frame['timestamp'] = pd.date_range('2018-01-01 00:00:00.000000001', periods=3, freq='h', tz='Asia/Shanghai')
    elif invalid_case == 'string_timestamp':
        frame['timestamp'] = frame.timestamp.astype(str)
    elif invalid_case in ('nonnumeric', 'numeric_strings'):
        frame['aqi'] = ['bad' if invalid_case == 'nonnumeric' else '150'] * 3
    elif invalid_case in ('infinite', 'negative_infinite'):
        frame.loc[1, 'pm10_iaqi'] = np.inf if invalid_case == 'infinite' else -np.inf
    elif invalid_case == 'complex':
        frame['aqi'] = np.array([150 + 1j] * 3)
    elif invalid_case == 'boolean':
        frame['aqi'] = True
    with pytest.raises(ValueError):
        run_features(frame)
