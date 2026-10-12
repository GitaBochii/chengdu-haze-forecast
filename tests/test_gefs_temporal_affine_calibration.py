import numpy as np
import pytest

from scripts.benchmark_gefs_temporal_affine_calibration import fit_calibrator, predict


def test_recovers_positive_affine_relation():
    x = np.array([1., 2., 3., 4.])
    params = fit_calibrator(x, 2 + 3 * x, np.ones(4))
    np.testing.assert_allclose(params, [2., 3.], atol=1e-8)
    np.testing.assert_allclose(predict(params, x), 2 + 3 * x)


def test_monotone_with_decreasing_truth():
    params = fit_calibrator(np.arange(1., 5.), np.arange(4., 0., -1.), np.ones(4))
    assert params[1] >= 0
    assert (np.diff(predict(params, np.arange(1., 5.))) >= 0).all()


@pytest.mark.parametrize('weights', [[0, 0], [1, -1], [1, np.nan]])
def test_reject_bad_weights(weights):
    with pytest.raises(ValueError):
        fit_calibrator([1, 2], [2, 3], weights)
