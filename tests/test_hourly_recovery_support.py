import numpy as np
import pytest

from haze_forecast.hourly_recovery_support import enforce_hourly_support


def test_only_eligible_origins_are_projected():
    p = np.array([.2, .2, 1., 200.])
    np.testing.assert_array_equal(enforce_hourly_support(p, [True, False, True, True]), [1, .2, 1, 200])
    np.testing.assert_array_equal(p, [.2, .2, 1, 200])


def test_projection_cannot_increase_error_for_valid_hourly_targets():
    p = np.linspace(0, 200, 1001)
    q = enforce_hourly_support(p, np.ones(len(p), dtype=bool))
    for truth in (1., 2., 24., 200., 1000.):
        assert (np.abs(q-truth) <= np.abs(p-truth)).all()


@pytest.mark.parametrize('p,mask', [([-1], [True]), ([np.nan], [True]),
    ([np.inf], [True]), ([1, 2], [True]), ([1], [1]), ([[1]], [True])])
def test_invalid_inputs_rejected(p, mask):
    with pytest.raises(ValueError):
        enforce_hourly_support(p, mask)
