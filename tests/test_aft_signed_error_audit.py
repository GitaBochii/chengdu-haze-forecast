import numpy as np
import pandas as pd
import pytest

from scripts.audit_aft_signed_errors import summarize


def fixture():
    return pd.DataFrame({
        'exact_hours': [2., 4., 30., np.nan],
        'lower_bound_hours': [2., 4., 30., 40.],
        'upper_bound_hours': [2., 4., 30., 50.],
        'continuity_block': [1, 1, 2, 3],
        'prediction': [5., 5., 20., 20.],
    })


def test_signed_error_and_block_weighting():
    result = summarize(fixture(), 'prediction')
    assert result['exact']['rows'] == 3
    assert result['exact']['row_bias'] == pytest.approx(-2.)
    assert result['exact']['block_bias'] == pytest.approx(-4.)
    assert result['exact']['block_mae'] == pytest.approx(6.)
    assert result['short']['row_bias'] == pytest.approx(2.)
    assert result['tail']['row_bias'] == pytest.approx(-10.)


def test_interval_is_not_exact_mae():
    result = summarize(fixture(), 'prediction')
    assert result['interval']['rows'] == 1
    assert result['interval']['row_distance'] == 20.
    assert result['interval']['below_lower_fraction'] == 1.
    assert 'row_mae' not in result['interval']


def test_empty_subgroup_serializes_as_null():
    result = summarize(fixture().iloc[:2], 'prediction')
    assert result['tail']['row_mae'] is None
    assert result['interval']['row_distance'] is None


@pytest.mark.parametrize('value', [np.nan, np.inf, -1.])
def test_reject_bad_prediction(value):
    frame = fixture()
    frame.loc[0, 'prediction'] = value
    with pytest.raises(ValueError):
        summarize(frame, 'prediction')


def test_reject_inconsistent_exact_label():
    frame = fixture()
    frame.loc[0, 'lower_bound_hours'] = 1.
    with pytest.raises(ValueError):
        summarize(frame, 'prediction')


def test_direction_bounds_preserve_unknown_intervals_and_ties():
    frame = fixture()
    frame.loc[0, 'prediction'] = 2.
    frame.loc[3, 'prediction'] = 45.
    result = summarize(frame, 'prediction')['direction_bounds']
    assert result['definitely_under_fraction'] == .25
    assert result['possibly_under_fraction'] == .5
    assert result['definitely_over_fraction'] == .25
    assert result['possibly_over_fraction'] == .5
    assert result['exact_tie_fraction'] == .25
    assert result['block_definitely_under_fraction'] == pytest.approx(1 / 3)
