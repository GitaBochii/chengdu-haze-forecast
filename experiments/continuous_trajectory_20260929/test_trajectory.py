import numpy as np
import pandas as pd
import pytest

from trajectory import windows, FEATURES, baselines, score, save_booster, load_booster


def frame(start='2023-12-28', periods=160):
    p = pd.DataFrame({'timestamp': pd.date_range(start, periods=periods, freq='h', tz='Asia/Shanghai')})
    for i, name in enumerate(FEATURES):
        p[name] = np.arange(periods, dtype=float) + i
    return p


def test_windows_have_exact_future_and_no_boundary_overlap():
    p = frame()
    w = windows(p)
    for i, origin in enumerate(w['indices']):
        np.testing.assert_array_equal(w['history'][i], p[list(FEATURES)].to_numpy()[origin-23:origin+1])
        np.testing.assert_array_equal(w['target'][i], p[list(FEATURES)].to_numpy()[origin+1:origin+25])
        assert p.timestamp.iloc[origin-23].year == p.timestamp.iloc[origin+24].year


def test_clean_days_kept_and_missing_labels_not_filled():
    p = frame('2025-05-01', 100)
    p.loc[:, list(FEATURES)] = 20.
    p.loc[60, 'aqi'] = np.nan
    w = windows(p)
    assert len(w['indices']) > 0
    assert not w['polluted'].any()
    assert all(not (i-23 <= 60 <= i+24) for i in w['indices'])


def test_gap_and_duplicates_rejected():
    p = frame()
    with pytest.raises(ValueError): windows(p.drop(index=30))
    with pytest.raises(ValueError): windows(pd.concat([p, p.iloc[:1]]))


def test_baselines_and_metrics():
    history = np.arange(72, dtype='float32').reshape(1,24,3)
    b = baselines(history)
    np.testing.assert_array_equal(b['daily_repeat'], history)
    np.testing.assert_array_equal(b['persistence'][0,0], history[0,-1])
    y = np.zeros((2,24,3))
    result = score(y, y+2, ['2025-01-01','2025-01-02'])
    assert result['aqi']['mae'] == result['aqi']['rmse'] == 2
    assert result['aqi']['mae_h24'] == 2


def test_model_roundtrip_with_unicode_path(tmp_path):
    import lightgbm as lgb
    x = np.arange(100).reshape(50,2)
    model = lgb.LGBMRegressor(n_estimators=2, min_child_samples=2, verbosity=-1, n_jobs=1).fit(x, x[:,0])
    path = tmp_path / '\u6a21\u578b.txt'
    save_booster(model.booster_, path)
    restored = load_booster(path)
    np.testing.assert_array_equal(model.booster_.predict(x), restored.predict(x))
