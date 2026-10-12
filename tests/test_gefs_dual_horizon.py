import importlib

import numpy as np
import pandas as pd
import pytest


def merge():
    assert importlib.util.find_spec('scripts.benchmark_gefs_dual_horizon') is not None
    return importlib.import_module('scripts.benchmark_gefs_dual_horizon').combine


def test_preserve_short_horizon_and_missing_rows():
    short = pd.DataFrame({'timestamp': [1, 2], 'x': [3., 4.], 'forecast_known_age_hours': [1., 2.]})
    long = pd.DataFrame({'timestamp': [1, 2], 'gefs_mean_temperature_c_f072': [10., np.nan],
                         'forecast_known_age_hours': [5., np.nan]})
    before = short.copy()
    out, names = merge()(short, long, ['x', 'forecast_known_age_hours'],
                          ['gefs_mean_temperature_c_f072', 'forecast_known_age_hours'])
    pd.testing.assert_frame_equal(out[short.columns], before)
    assert names[-1] == 'forecast_known_age_hours_f072'
    assert out.iloc[0][names[-1]] == 5
    assert pd.isna(out.iloc[1][names[-1]])
    pd.testing.assert_frame_equal(short, before)


def test_origin_misalignment_rejected():
    short = pd.DataFrame({'timestamp': [1, 2], 'x': [3., 4.]})
    long = pd.DataFrame({'timestamp': [2, 1], 'y': [5., 6.]})
    with pytest.raises(ValueError, match='origin'):
        merge()(short, long, ['x'], ['y'])


def test_column_collision_rejected():
    frame = pd.DataFrame({'timestamp': [1], 'x': [3.]})
    with pytest.raises(ValueError, match='collision'):
        merge()(frame, frame, ['x'], ['x'])
