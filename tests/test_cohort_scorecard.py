import numpy as np
import pandas as pd
import pytest

from haze_forecast.cohort_scorecard import score_predictions


def rows():
    return pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=3, freq='h', tz='UTC'),
        'continuity_block': [1, 1, 2],
        'lower_bound_hours': [2., 4., 6.],
        'upper_bound_hours': [2., 8., 10.],
        'exact_hours': [2., np.nan, np.nan],
    })


def test_exact_and_interval_bounds_include_every_row():
    result = score_predictions(rows(), np.array([3., 2., 12.]))
    assert result['rows'] == 3
    assert result['exact_row_mae'] == 1
    assert result['all_row_error_lower'] == pytest.approx(5 / 3)
    assert result['all_row_error_upper'] == pytest.approx(13 / 3)
    assert result['all_block_error_lower'] == 1.75
    assert result['tail24_block_mae'] is None


def test_right_censor_does_not_silently_disappear():
    frame = rows()
    frame.loc[2, 'upper_bound_hours'] = np.inf
    result = score_predictions(frame, np.array([3., 2., 12.]))
    assert result['right_censored_rows'] == 1
    assert result['all_row_error_upper'] is None
    assert result['all_block_error_upper'] is None
    assert result['rows'] == 3


@pytest.mark.parametrize('prediction', [[1., np.nan, 3.], [1., 2.], [1., -1., 2.]])
def test_invalid_prediction_rejected_without_dropping_rows(prediction):
    with pytest.raises(ValueError):
        score_predictions(rows(), np.array(prediction))


def test_duplicate_origins_and_inconsistent_exact_label_rejected():
    frame = rows()
    frame.loc[1, 'timestamp'] = frame.loc[0, 'timestamp']
    with pytest.raises(ValueError):
        score_predictions(frame, np.ones(3))
    frame = rows()
    frame.loc[0, 'exact_hours'] = 3.
    with pytest.raises(ValueError):
        score_predictions(frame, np.ones(3))
