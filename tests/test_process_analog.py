import numpy as np
import pytest

from scripts.benchmark_process_analog import interval_median


def test_exact_median():
    assert interval_median([1, 3, 9], [1, 3, 9]) == 3


def test_interval_minimizer_not_midpoint_labels():
    lower, upper = np.array([1, 5, 8]), np.array([20, 6, 9])
    p = interval_median(lower, upper)
    loss = lambda x: np.maximum.reduce([lower - x, x - upper, np.zeros(3)]).sum()
    assert loss(p) == min(loss(x) for x in np.linspace(0, 25, 251))
    assert p != np.median((lower + upper) / 2)


@pytest.mark.parametrize('lo,hi', [([], []), ([2], [1]), ([1], [np.inf])])
def test_bad_intervals(lo, hi):
    with pytest.raises(ValueError):
        interval_median(lo, hi)
