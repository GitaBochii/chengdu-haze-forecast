import numpy as np
import pandas as pd
import pytest

from audit_results import independent_scores, reconstruct, compare_tree


def test_scores_day_equal_and_horizon():
    target = np.zeros((3, 24, 3))
    pred = np.broadcast_to(np.array([1., 3., 8.])[:, None, None], target.shape)
    scores = independent_scores(target, pred, ['a', 'a', 'b'])
    for channel in scores.values():
        assert channel['mae'] == 4
        assert channel['day_equal_mae'] == 5
        assert channel['rmse'] == pytest.approx(np.sqrt(74/3))
        assert channel['mae_h24'] == 4


def test_reconstruct_excludes_invalid_and_cross_year():
    ts = pd.date_range('2023-12-28', periods=240, freq='h', tz='Asia/Shanghai')
    frame = pd.DataFrame({'timestamp': ts, 'pm2_5': 1., 'pm10': 2., 'aqi': 3.})
    frame.loc[180, 'pm10'] = np.nan
    meta, history, target = reconstruct(frame)
    expected = [i for i in range(23, 216) if ts[i-23].year == ts[i+24].year
                and not i-23 <= 180 <= i+24]
    assert meta.source_index.tolist() == expected
    assert target.shape == history.shape == (len(expected), 24, 3)
    assert not meta.polluted.any()


def test_compare_rejects_changed_metric_and_missing_key():
    compare_tree({'x': 1.}, {'x': 1. + 1e-12})
    with pytest.raises(AssertionError):
        compare_tree({'x': 1.}, {'x': 1.01})
    with pytest.raises(AssertionError):
        compare_tree({'x': 1.}, {})


def test_scores_reject_negative_or_wrong_shape():
    target = np.ones((3, 24, 3))
    with pytest.raises(AssertionError):
        independent_scores(target, -target, ['a'] * 3)
    with pytest.raises(AssertionError):
        independent_scores(target, target[:, :1], ['a'] * 3)
