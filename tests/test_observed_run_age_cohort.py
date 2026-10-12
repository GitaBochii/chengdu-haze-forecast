import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_observed_run_age import RUN_FEATURES, attach_run_features


def frames():
    times = pd.date_range('2018-01-01', periods=3, freq='h', tz='Asia/Shanghai')
    original = pd.DataFrame({'timestamp': times[[2, 0]], 'old_feature': [7, 8]}, index=[9, 4])
    run = pd.DataFrame({'timestamp': times, RUN_FEATURES[0]: [1., 2., 3.],
                        RUN_FEATURES[1]: [0., 0., 0.]})
    return original, run


def test_join_preserves_original_values_order_and_index():
    original, run = frames()
    result = attach_run_features(original, run)
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert result[RUN_FEATURES[0]].tolist() == [3., 1.]
    assert result[RUN_FEATURES[1]].tolist() == [0., 0.]
    assert set(original.columns) == {'timestamp', 'old_feature'}


@pytest.mark.parametrize('failure', ['missing', 'duplicate_run', 'duplicate_origin', 'empty',
                                   'unknown', 'nonpolluted', 'invalid_flag', 'collision'])
def test_join_does_not_drop_or_repair_invalid_origins(failure):
    original, run = frames()
    if failure == 'missing':
        run = run.iloc[:2]
    elif failure == 'duplicate_run':
        run = pd.concat([run, run.iloc[:1]], ignore_index=True)
    elif failure == 'duplicate_origin':
        original = pd.concat([original, original.iloc[:1]])
    elif failure == 'empty':
        original = original.iloc[:0]
    elif failure == 'unknown':
        run.loc[0, RUN_FEATURES[0]] = np.nan
    elif failure == 'nonpolluted':
        run.loc[0, RUN_FEATURES[0]] = 0.
    elif failure == 'invalid_flag':
        run.loc[0, RUN_FEATURES[1]] = 0.5
    else:
        original[RUN_FEATURES[0]] = 123.
    with pytest.raises(ValueError):
        attach_run_features(original, run)
