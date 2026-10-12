import numpy as np
import pytest
from six_hour_max import maximum_targets, maximum_start, select_buffer


def test_maximum_encodes_exact_six_hour_condition():
    y = np.full((3,24),101.)
    y[0,:6] = 100
    y[1,18:] = 90
    y[2,19:] = 90
    maximum = maximum_targets(y)
    assert maximum.shape == (3,19)
    np.testing.assert_array_equal(maximum_start(maximum), [1,19,20])


def test_one_high_hour_prevents_six_hour_recovery():
    y = np.full((1,24),90.)
    y[0,5] = 150
    maximum = maximum_targets(y)
    assert maximum[0,0] == 150
    assert maximum_start(maximum)[0] == 7
    with pytest.raises(ValueError):
        maximum_start(np.full((1,19),np.nan))


def test_dev_selection_keeps_right_censored_origins():
    y = np.full((4,24),110.)
    y[:2,:6] = 90
    p = maximum_targets(y)
    p[2,0] = 99
    delta, grid = select_buffer(y,p,1.)
    assert delta == 10
    assert len(grid) == 102
    assert grid[delta]['metrics']['confusion'] == dict(tp=2,fn=0,fp=0,tn=2)
