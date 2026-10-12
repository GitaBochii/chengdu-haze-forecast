import numpy as np
import pandas as pd
import pytest

from verify_independent import HOUR, decision, greedy, grouping, metrics, rebuild, recovery, select, probability_schema, detached_metrics


def test_recovery_censoring_and_rebound():
    curves = np.full((4, 24), 150.)
    curves[1, :6] = 90
    curves[1, 6:] = 200
    curves[2, :5] = 90
    curves[3, 18:] = 100
    assert recovery(curves).tolist() == [20, 1, 20, 19]
    curves[0, 0] = np.nan
    with pytest.raises(ValueError):
        recovery(curves)


def test_missing_and_partition_break_groups():
    dates = pd.date_range('2019-12-31 18:00', periods=12, freq='h', tz='Asia/Shanghai')
    core = np.full((12, 3), 50.)
    core[2] = np.nan
    ids = grouping(dates, core)
    assert ids[0] != ids[3] and ids[5] != ids[6] and ids[2] == ''


def test_process_includes_sixth_clean_hour_and_short_rebound():
    dates = pd.date_range('2020-01-01', periods=12, freq='h', tz='Asia/Shanghai')
    core = np.full((12, 3), 50.)
    core[[0, 3], 2] = 200
    ids = grouping(dates, core)
    assert len(set(ids[:10])) == 1 and ids[10] != ids[9]


def test_issue_pm_missing_does_not_break_aqi_process():
    dates = pd.date_range('2020-01-01', periods=12, freq='h', tz='Asia/Shanghai')
    core = np.full((12, 3), 150.)
    core[3, 0] = np.nan
    assert len(set(grouping(dates, core))) == 1


def test_window_boundary_and_missing_no_fill():
    dates = pd.date_range('2019-12-30', periods=120, freq='h', tz='Asia/Shanghai')
    frame = pd.DataFrame(dict(timestamp=dates, pm2_5=50., pm10=60., aqi=80.))
    _, _, data, reasons = rebuild(frame)
    assert 'boundary' in reasons and np.all(data['indices'] >= 24)
    frame.loc[90, 'pm2_5'] = np.nan
    _, _, data, reasons = rebuild(frame)
    assert 'nonfinite' in reasons and not any(66 <= i <= 114 and i != 90 for i in data['indices'])


def test_greedy_tie_overlap_and_insufficient_k():
    times = np.arange(5)*49*HOUR
    groups = np.array(['a', 'a', 'b', 'c', 'd'])
    chosen, _ = greedy(np.ones(5), times, groups, 3)
    assert chosen.tolist() == [0, 2, 3]
    assert (greedy(np.ones(5), times, groups, 5)[0] == -1).all()
    chosen, _ = greedy(np.zeros(3), np.array([0, 48*HOUR, 49*HOUR]), np.array(['a', 'b', 'c']), 2)
    assert chosen.tolist() == [0, 2]


def test_false_positive_denominator_and_all_reject():
    m = metrics(np.array([1, 20, 20, 20]), np.array([20, 2, 20, 20]))
    assert m['fpr'] == 1/3 and m['recall'] == 0
    cdf = np.ones((4, 19))
    assert decision(cdf, 1.000001).tolist() == [20]*4
    m = dict(theta=1.000001, **metrics([1, 20], [20, 20]))
    assert select([m], 1) is None


def test_features_have_no_issue_or_future_concentration():
    dates = pd.date_range('2020-05-01', periods=80, freq='h', tz='Asia/Shanghai')
    frame = pd.DataFrame(dict(timestamp=dates, pm2_5=50., pm10=60., aqi=80.))
    before = rebuild(frame)[2]
    frame.loc[40:, ['pm2_5', 'pm10', 'aqi']] = [400., 500., 600.]
    after = rebuild(frame)[2]
    a = before['features'][before['indices'] == 40]
    b = after['features'][after['indices'] == 40]
    np.testing.assert_array_equal(a, b)
    assert before['labels'][before['indices'] == 40] != after['labels'][after['indices'] == 40]


def test_probability_censored_mass_and_group_weighting():
    labels = np.array([1, 20, 20])
    pmf = np.zeros((3, 20)); pmf[:, 0] = .25; pmf[:, 19] = .75
    cdf = np.cumsum(pmf[:, :19], axis=1)
    assert np.all(cdf[:, -1] == .25) and np.all(pmf.sum(axis=1) == 1)
    p = probability_schema(labels, cdf, np.array(['a', 'b', 'b']))
    assert p['brier_6'] == (0.75**2+2*.25**2)/3
    assert p['group_equal_brier_6'] == (0.75**2+.25**2)/2


def test_supplement_is_not_removed_by_comparison_schema():
    main_comparison = {'probability': {'brier_6': .2, 'group_equal_brier_mean_1_19': .3}}
    retained = detached_metrics(main_comparison)
    main_comparison['probability'] = {'brier_6': .2}
    assert retained['probability']['group_equal_brier_mean_1_19'] == .3
    assert 'group_equal_brier_mean_1_19' not in main_comparison['probability']
