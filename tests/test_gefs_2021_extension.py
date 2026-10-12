import importlib

import numpy as np
import pandas as pd
import pytest


def module():
    return importlib.import_module('scripts.benchmark_gefs_2021_extension')


def example():
    m = module()
    timestamps = pd.to_datetime(['2020-12-31', '2021-01-02', '2021-12-30', '2022-01-02'], utc=True)
    train = pd.DataFrame({'timestamp': timestamps, 'lower_bound_hours': [1., 2., 3., 4.],
                          'upper_bound_hours': [1., 4., 3., 8.], 'history': [8., 7., 6., 5.]})
    for field in m.FORECAST:
        train[field] = [9., np.nan, np.nan, 10.]
    features = pd.DataFrame({'timestamp': timestamps[1:3]})
    for field in m.FORECAST:
        features[field] = [3., np.nan]
    return train, features


def test_fill_only_2021_preserves_every_other_input_and_label():
    train, features = example()
    original = train.copy(deep=True)
    result = module().extend_training(train, features, expected_rows=2)
    pd.testing.assert_frame_equal(train, original)
    pd.testing.assert_frame_equal(result.loc[[0, 3]], original.loc[[0, 3]])
    pd.testing.assert_frame_equal(result.drop(columns=module().FORECAST), original.drop(columns=module().FORECAST))
    assert result.loc[1, module().FORECAST].eq(3).all()
    assert result.loc[2, module().FORECAST].isna().all()


def test_refuse_nonempty_original_forecast():
    train, features = example()
    train.loc[1, module().FORECAST[-1]] = 0.
    with pytest.raises(ValueError, match='original 2021 forecast'):
        module().extend_training(train, features, expected_rows=2)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'outside'])
def test_refuse_changed_population(change):
    train, features = example()
    if change == 'missing':
        features = features.iloc[:1]
    elif change == 'duplicate':
        features.loc[1, 'timestamp'] = features.loc[0, 'timestamp']
    else:
        features.loc[1, 'timestamp'] = train.loc[3, 'timestamp']
    with pytest.raises(ValueError):
        module().extend_training(train, features, expected_rows=2)


@pytest.mark.parametrize('count', [0, 100, 102])
def test_incomplete_audit_refused(count):
    audit = {'verified_runs': count, 'planned_runs': 101, 'verified_grib_messages': count * 9,
             'all_planned_runs_verified': count == 101, 'verified': []}
    with pytest.raises(ValueError, match='audit population'):
        module().validate_audit(audit, ['run'] * 101, 'protocol', 'script')


def test_stale_audit_refused():
    runs = [str(i) for i in range(101)]
    audit = {'verified_runs': 101, 'planned_runs': 101, 'verified_grib_messages': 909,
             'all_planned_runs_verified': True, 'protocol_sha256': 'stale',
             'audit_script_sha256': 'script', 'verified': [{'run_at': r} for r in runs]}
    with pytest.raises(ValueError, match='stale audit'):
        module().validate_audit(audit, runs, 'protocol', 'script')


@pytest.mark.parametrize('change', ['none', 'script', 'duplicate', 'missing'])
def test_audit_exact_inventory_and_auditor_identity(change):
    runs = [str(i) for i in range(101)]
    audit = {'verified_runs': 101, 'planned_runs': 101, 'verified_grib_messages': 909,
             'all_planned_runs_verified': True, 'protocol_sha256': 'protocol',
             'audit_script_sha256': 'script',
             'verified': [{'run_at': r, 'complete_sha256': 'hash' + r} for r in runs]}
    if change == 'script':
        audit['audit_script_sha256'] = 'stale'
    elif change == 'duplicate':
        audit['verified'][-1] = audit['verified'][0]
    elif change == 'missing':
        audit['verified'].pop()
    if change == 'none':
        assert module().validate_audit(audit, runs, 'protocol', 'script') == {r: 'hash' + r for r in runs}
    else:
        with pytest.raises(ValueError):
            module().validate_audit(audit, runs, 'protocol', 'script')


def test_matching_release_inclusive_expiration_exclusive():
    m = module()
    run = pd.Timestamp('2021-01-01', tz='UTC')
    snapshots = pd.DataFrame([{'run_at': run, 'known_at': run + pd.Timedelta('4h'),
                               'valid_at': run + pd.Timedelta('24h'), **dict.fromkeys(m.FIELDS, 1.)}])
    origins = pd.DatetimeIndex([run + pd.Timedelta(x) for x in ['3h', '4h', '24h']])
    features = m.match_2021(snapshots, origins)
    assert features[m.FIELDS[0]].isna().tolist() == [True, False, True]
    assert features.loc[1, 'forecast_known_age_hours'] == 0


def test_expanded_cohort_may_omit_forecast_features_but_not_change_origins():
    train, _ = example()
    original = train[['timestamp', 'history'] + module().FORECAST].copy()
    expanded = original[['timestamp', 'history']].copy()
    result = module().frozen_inference(original, expanded, ['history'] + module().FORECAST)
    pd.testing.assert_frame_equal(result, original)
    expanded.loc[0, 'history'] += 1
    with pytest.raises(AssertionError):
        module().frozen_inference(original, expanded, ['history'] + module().FORECAST)
