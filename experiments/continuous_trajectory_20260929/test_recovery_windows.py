import numpy as np
import pytest
from recovery_windows import window_labels, decode, choose


def test_observed_six_hour_window_edges():
    y = np.full((3,24),101.)
    y[0,:6] = 100
    y[1,18:] = 99
    y[2,19:] = 90
    labels = window_labels(y)
    assert labels.shape == (3,19)
    np.testing.assert_array_equal(labels.sum(axis=1), [1,1,0])
    assert labels[0,0] and labels[1,18]


def test_probability_decoder_first_window_and_abstention():
    p = np.zeros((3,19))
    p[0,[2,5]] = .8
    p[1,18] = .5
    np.testing.assert_array_equal(decode(p,.5), [3,19,20])
    np.testing.assert_array_equal(decode(p,1.000001), [20,20,20])
    with pytest.raises(ValueError):
        decode(np.full((1,19),np.nan), .5)


def test_threshold_chosen_without_test_and_censored_rows_retained():
    y = np.full((4,24),110.)
    y[:2,:6] = 90
    p = np.zeros((4,19))
    p[:2,0] = .9
    p[2,0] = .4
    threshold, grid = choose(y,p,1.)
    assert threshold == .9
    assert len(grid) == 21
    chosen = next(r for r in grid if r['threshold'] == threshold)
    assert chosen['metrics']['confusion'] == dict(tp=2,fn=0,fp=0,tn=2)
    assert chosen['metrics']['origins'] == 4
