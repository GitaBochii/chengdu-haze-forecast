import json

import numpy as np
import pandas as pd
import pytest

from haze_forecast import paired_interval_score as scorer


def cohort(lo, hi, blocks=None, index=None):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return pd.DataFrame({
        'timestamp': pd.date_range('2018-01-01', periods=len(lo), freq='h', tz='UTC'),
        'continuity_block': blocks if blocks is not None else np.arange(len(lo)),
        'lower_bound_hours': lo,
        'upper_bound_hours': hi,
        'exact_hours': np.where(lo == hi, lo, np.nan),
    }, index=index)


def test_exact_rows_are_direct_paired_error_difference():
    frame = cohort([2, 10, 30], [2, 10, 30], index=[8, 4, 7])
    baseline = np.array([3., 8., 34.])
    candidate = np.array([2., 14., 32.])
    expected = abs(candidate - frame.exact_hours) - abs(baseline - frame.exact_hours)
    result = scorer.paired_error_bounds(frame, baseline, candidate)
    assert result.index.equals(frame.index)
    assert list(result.columns) == ['lower', 'upper']
    np.testing.assert_array_equal(result.lower, expected)
    np.testing.assert_array_equal(result.upper, expected)


@pytest.mark.parametrize(('baseline', 'candidate', 'expected'), [
    (1., 2., (-1., -1.)),
    (12., 14., (2., 2.)),
    (2., 12., (-6., 6.)),
    (12., 2., (-6., 6.)),
    (5., 8., (-3., 3.)),
    (7., 7., (0., 0.)),
])
def test_predictions_on_same_side_crossing_or_equal(baseline, candidate, expected):
    frame = cohort([4], [10])
    result = scorer.paired_error_bounds(frame, [baseline], [candidate])
    assert tuple(result.iloc[0]) == expected


def test_right_censor_has_finite_paired_bounds_without_inf_subtraction():
    frame = cohort([4, 20, 5], [np.inf, np.inf, np.inf])
    with np.errstate(invalid='raise', over='raise'):
        result = scorer.paired_error_bounds(frame, [2., 2., 7.], [12., 12., 7.])
    np.testing.assert_array_equal(result.to_numpy(), [[-10., 6.], [-10., -10.], [0., 0.]])
    summary = scorer.paired_error_summary(frame, [2., 2., 7.], [12., 12., 7.])
    assert summary['right_censored_rows'] == 3
    assert summary['rows'] == 3
    json.dumps(summary, allow_nan=False)


def test_large_finite_values_do_not_overflow_during_pairwise_difference():
    frame = cohort([1.], [np.inf])
    with np.errstate(invalid='raise', over='raise'):
        result = scorer.paired_error_bounds(frame, [0.], [1e308])
    np.testing.assert_array_equal(result.to_numpy(), [[-1e308, 1e308]])


def test_random_finite_and_censored_intervals_match_dense_scan():
    rng = np.random.default_rng(20260912)
    lo = rng.uniform(0.1, 100., size=120)
    hi = lo + rng.uniform(0., 100., size=120)
    hi[::7] = np.inf
    baseline = rng.uniform(0., 250., size=120)
    candidate = rng.uniform(0., 250., size=120)
    result = scorer.paired_error_bounds(cohort(lo, hi), baseline, candidate)
    for i in range(len(lo)):
        stop = hi[i] if np.isfinite(hi[i]) else max(lo[i], baseline[i], candidate[i]) + 1.
        truth = np.linspace(lo[i], stop, 10001)
        direct = abs(candidate[i] - truth) - abs(baseline[i] - truth)
        assert result.lower.iloc[i] == pytest.approx(direct.min(), abs=1e-10)
        assert result.upper.iloc[i] == pytest.approx(direct.max(), abs=1e-10)


def test_summary_preserves_rows_and_uses_process_equal_means():
    frame = cohort([2, 4, 6], [2, 8, 10], blocks=[1, 1, 2], index=[6, 2, 9])
    baseline = np.array([3., 2., 12.])
    candidate = np.array([2., 5., 9.])
    result = scorer.paired_error_bounds(frame, baseline, candidate)
    np.testing.assert_array_equal(result.to_numpy(), [[-1., -1.], [-3., -1.], [-3., -1.]])
    summary = scorer.paired_error_summary(frame, baseline, candidate)
    assert summary['rows'] == 3
    assert summary['blocks'] == 2
    assert summary['prediction_coverage'] == 1.0
    assert summary['all_row_delta_lower'] == pytest.approx(-7 / 3)
    assert summary['all_row_delta_upper'] == -1.
    assert summary['all_block_delta_lower'] == -2.5
    assert summary['all_block_delta_upper'] == -1.
    assert summary['uncertain_row_delta_lower'] == -3.
    assert summary['uncertain_row_delta_upper'] == -1.
    assert summary['uncertain_block_delta_lower'] == -3.
    assert summary['uncertain_block_delta_upper'] == -1.
    assert summary['exact_row_delta'] == -1.
    assert summary['exact_block_delta'] == -1.
    assert summary['uncertain_rows'] == 2
    assert summary['uncertain_blocks'] == 2
    assert summary['exact_rows'] == 1
    json.dumps(summary, allow_nan=False)


def test_empty_subgroups_return_none_not_nan():
    exact = scorer.paired_error_summary(cohort([2], [2]), [1.], [2.])
    uncertain = scorer.paired_error_summary(cohort([2], [4]), [1.], [2.])
    assert exact['uncertain_rows'] == 0
    assert exact['uncertain_row_delta_lower'] is None
    assert exact['uncertain_block_delta_upper'] is None
    assert uncertain['exact_row_delta'] is None
    assert uncertain['exact_block_delta'] is None
    json.dumps(exact, allow_nan=False)
    json.dumps(uncertain, allow_nan=False)


@pytest.mark.parametrize('argument', ['baseline', 'candidate'])
def test_series_index_mismatch_is_rejected_but_arrays_are_positional(argument):
    frame = cohort([2, 4], [2, 8], index=[5, 3])
    aligned = pd.Series([1., 2.], index=frame.index)
    mismatched = pd.Series([1., 2.], index=[3, 5])
    arguments = {'baseline': aligned, 'candidate': aligned}
    arguments[argument] = mismatched
    with pytest.raises(ValueError, match='index'):
        scorer.paired_error_bounds(frame, **arguments)
    with pytest.raises(ValueError, match='index'):
        scorer.paired_error_summary(frame, **arguments)
    expected = scorer.paired_error_bounds(frame, aligned, aligned)
    actual = scorer.paired_error_bounds(frame, aligned.to_numpy(), aligned.to_numpy())
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize('bad_prediction', [[1., np.nan], [1., np.inf], [1., -1.], [1.], [[1.], [2.]]])
@pytest.mark.parametrize('argument', ['baseline', 'candidate'])
def test_invalid_predictions_fail_without_dropping_rows(bad_prediction, argument):
    frame = cohort([2, 4], [2, 8])
    arguments = {'baseline': [1., 2.], 'candidate': [1., 2.]}
    arguments[argument] = bad_prediction
    with pytest.raises(ValueError):
        scorer.paired_error_bounds(frame, **arguments)


@pytest.mark.parametrize('invalid_case', [
    'duplicate_timestamp', 'missing_timestamp', 'missing_block', 'zero_lower',
    'nan_lower', 'nan_upper', 'inverted_bounds', 'wrong_exact', 'empty',
])
def test_existing_scorecard_input_constraints_are_preserved(invalid_case):
    frame = cohort([2, 4], [2, 8])
    if invalid_case == 'duplicate_timestamp':
        frame.loc[1, 'timestamp'] = frame.loc[0, 'timestamp']
    elif invalid_case == 'missing_timestamp':
        frame.loc[1, 'timestamp'] = pd.NaT
    elif invalid_case == 'missing_block':
        frame.loc[1, 'continuity_block'] = np.nan
    elif invalid_case == 'zero_lower':
        frame.loc[1, 'lower_bound_hours'] = 0.
    elif invalid_case == 'nan_lower':
        frame.loc[1, 'lower_bound_hours'] = np.nan
    elif invalid_case == 'nan_upper':
        frame.loc[1, 'upper_bound_hours'] = np.nan
    elif invalid_case == 'inverted_bounds':
        frame.loc[1, 'upper_bound_hours'] = 3.
    elif invalid_case == 'wrong_exact':
        frame.loc[1, 'exact_hours'] = 5.
    elif invalid_case == 'empty':
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        scorer.paired_error_bounds(frame, np.ones(len(frame)), np.ones(len(frame)))
