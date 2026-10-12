import numpy as np
import pytest
from recovery_calibration import calibrate, buffered_prediction, select_candidate


def test_buffer_preserves_curve_and_true_label():
    p = np.full((1, 24), 96.)
    original = p.copy()
    assert buffered_prediction(p, 5)[0, 0] == 101
    np.testing.assert_array_equal(p, original)
    with pytest.raises(ValueError):
        buffered_prediction(p, -1)


def test_selection_is_constrained_then_minimum_error():
    def row(delta, mae, fpr, recall, early):
        return dict(delta=delta, capped_min_T_20_mae_hours=mae,
                    fpr=fpr, recall=recall, premature_start_fraction_all=early)
    rows = [row(0, 1, .2, .9, .3), row(3, 2, .05, .8, .2),
            row(4, 3, 0, .7, .1), row(5, 1, .04, .9, .4)]
    assert select_candidate(rows, .3)['delta'] == 3


def test_development_only_calibration_and_censored_denominator():
    y = np.full((4, 24), 110.)
    y[:2, :6] = 90
    p = np.full((4, 24), 110.)
    p[:2, :6] = 90
    p[2, :6] = 99
    delta, rows = calibrate(y, p)
    assert delta == 10  # Equal performance prefers the most conservative buffer.
    assert len(rows) == 102
    assert rows[delta]['confusion'] == dict(tp=2, fn=0, fp=0, tn=2)


def test_no_negative_population_cannot_prove_fpr():
    with pytest.raises(ValueError):
        calibrate(np.full((2, 24), 90.), np.full((2, 24), 90.))
