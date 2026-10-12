import numpy as np
from constrained import choose


def test_budget_and_tie_rule():
    best, grid = choose(np.array([1,1,0,0], bool), {'unweighted':np.array([.9,.7,.8,.1])}, 0.)
    assert best['threshold'] == .9
    assert best['recall'] == .5
    assert best['fp'] == 0
    assert len(grid) == 101


def test_silence_is_not_useful():
    best, _ = choose(np.array([1,0], bool), {'unweighted':np.array([.1,.9])}, 0.)
    assert best['tp'] == 0
    assert best['threshold'] == 1.01


def test_identical_candidates_prefer_unweighted():
    p = np.array([.8,.1])
    best, _ = choose(np.array([1,0], bool), {'balanced':p,'unweighted':p}, .1)
    assert best['candidate'] == 'unweighted'
