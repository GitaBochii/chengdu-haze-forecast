import numpy as np
import pytest

from scripts.diagnose_regional_curves import curve_statistics


def test_first_six_hour_start_not_completion():
    truth = np.full((1, 72), 120.)
    truth[:, 4:10] = 90
    result = curve_statistics(truth, truth)
    assert result['truth_recovery'][0] == 5
    assert result['predicted_recovery'][0] == 5
    assert result['false_clean_hours'][0] == 0


def test_small_concentration_error_can_reverse_recovery():
    truth = np.full((1, 72), 101.)
    prediction = np.full((1, 72), 99.)
    result = curve_statistics(prediction, truth)
    assert result['aqi_mae'][0] == 2
    assert result['truth_recovery'][0] == 68
    assert result['predicted_recovery'][0] == 1
    assert result['false_clean_hours'][0] == 72
    assert result['false_predicted_window'][0]


def test_missing_truth_is_not_clean_or_complete():
    truth = np.full((1, 72), 90.)
    truth[0, 0] = np.nan
    result = curve_statistics(np.full((1, 72), 90.), truth)
    assert result['observed_hours'][0] == 71
    assert np.isnan(result['truth_recovery'][0])
    assert np.isnan(result['false_predicted_window'][0])


def test_nonfinite_predictions_fail():
    with pytest.raises(ValueError):
        curve_statistics(np.full((1, 72), np.nan), np.full((1, 72), 90.))


def test_last_confirmable_start_is_hour_67():
    values = np.full((1, 72), 120.)
    values[:, -6:] = 100
    assert curve_statistics(values, values)['truth_recovery'][0] == 67
    values[:, -6] = 101
    assert curve_statistics(values, values)['truth_recovery'][0] == 68


def test_all_missing_truth_retains_unknown():
    result = curve_statistics(np.full((1, 72), 90.), np.full((1, 72), np.nan))
    assert result['observed_hours'][0] == 0
    assert np.isnan(result['aqi_mae'][0])
    assert np.isnan(result['truth_recovery'][0])
