import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_endpoint_l1 import save_native_model, select_frozen_year


def cohort():
    timestamps = pd.to_datetime([
        '2016-01-01', '2017-12-30', '2017-12-31', '2018-01-01', '2018-02-01',
        '2019-01-01',
    ]).tz_localize('Asia/Shanghai')
    data = pd.DataFrame({
        'timestamp': timestamps, 'continuity_block': [0, 1, 2, 2, 3, 4],
        'label_available_at': timestamps + pd.to_timedelta([6, 72, 6, 6, 6, 6], unit='h'),
        'lower_bound_hours': 1., 'upper_bound_hours': 1., 'exact_hours': 1.,
        'feature': np.arange(6, dtype=float),
    })
    frozen = data.iloc[[3, 4]].assign(temporal_prediction=[2., 3.])
    members = data.iloc[[0]][['timestamp', 'continuity_block', 'label_available_at']]
    return data, frozen, members


def test_frozen_year_maturity_and_process_isolation():
    data, frozen, members = cohort()
    train, valid = select_frozen_year(data, frozen, members, 2018)
    pd.testing.assert_frame_equal(train, data.iloc[[0]])
    pd.testing.assert_frame_equal(valid, frozen)


def test_reject_changed_frozen_features():
    data, frozen, members = cohort()
    frozen.loc[frozen.index[0], 'feature'] += 1
    with pytest.raises(AssertionError):
        select_frozen_year(data, frozen, members, 2018)


def test_reject_missing_validation_origin():
    data, frozen, members = cohort()
    with pytest.raises(AssertionError):
        select_frozen_year(data, frozen.iloc[:1], members, 2018)


def test_reject_training_membership_change():
    data, frozen, members = cohort()
    with pytest.raises(AssertionError):
        select_frozen_year(data, frozen, members.iloc[:0], 2018)


def test_native_export_roundtrips_unicode_path(tmp_path):
    from lightgbm import Booster

    from haze_forecast.interval_endpoint_regression import make_endpoint_model

    frame = pd.DataFrame({'x': [1., 2., 3.]})
    model = make_endpoint_model(len(frame)).set_params(n_estimators=2)
    model.fit(frame, [5., 10., 15.])
    parent = tmp_path / '\u5927\u521b'
    parent.mkdir()
    path = parent / '\u6a21\u578b.txt'
    save_native_model(model, path)
    restored = Booster(model_str=path.read_text(encoding='utf-8'))
    np.testing.assert_array_equal(model.predict(frame), restored.predict(frame))
