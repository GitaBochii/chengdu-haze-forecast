import numpy as np
import pytest

from scripts.audit_regional_station_support import finite_mean, fixed_pool_truth


def test_excluded_station_cannot_change_median():
    raw = np.array([[[90., 110., 999.], [80., 100., -100.]]])
    median, count = fixed_pool_truth(raw, np.array([[True, True, False]]))
    np.testing.assert_array_equal(median, [[100., 90.]])
    np.testing.assert_array_equal(count, [[2, 2]])


def test_missing_member_does_not_silently_change_fixed_pool():
    raw = np.array([[[90., np.nan], [90., 110.]]])
    median, count = fixed_pool_truth(raw, np.array([[True, True]]))
    assert np.isnan(median[0, 0])
    assert median[0, 1] == 100
    np.testing.assert_array_equal(count, [[1, 2]])


def test_empty_pool_is_unknown():
    median, count = fixed_pool_truth(np.ones((1, 2, 2)), np.zeros((1, 2), dtype=bool))
    assert np.isnan(median).all()
    assert (count == 0).all()


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        fixed_pool_truth(np.ones((1, 2, 3)), np.ones((1, 2), dtype=bool))


def test_empty_metric_is_unavailable_not_zero():
    assert finite_mean([]) is None
    assert finite_mean([np.nan]) is None
    assert finite_mean([1., np.nan, 3.]) == 2
