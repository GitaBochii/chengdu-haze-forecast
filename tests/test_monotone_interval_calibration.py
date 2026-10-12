import numpy as np
import pytest

from haze_forecast.monotone_interval_calibration import fit_map, predict_map


def test_exact_affine_relation_and_monotonicity():
    x=np.arange(1.,6.)
    model=fit_map(x,2*x,2*x,np.ones(5))
    np.testing.assert_allclose(predict_map(model,x),2*x,atol=1e-8)
    assert (np.diff(predict_map(model,np.linspace(.1,10,100)))>=0).all()


def test_intervals_not_replaced_with_midpoints():
    x=np.arange(1.,6.)
    lo=x; hi=x+10
    model=fit_map(x,lo,hi,np.ones(5))
    p=predict_map(model,x)
    assert ((p>=lo-1e-8)&(p<=hi+1e-8)).all()
    assert model['training_distance']<1e-8


def test_outside_range_uses_identity_slope_and_positive_floor():
    model={'knots':[2.,4.],'values':[3.,7.]}
    np.testing.assert_array_equal(predict_map(model,[.1,1.,5.,10.]),[1.1,2.,8.,13.])


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        fit_map([1,2],[3,4],[2,5],[1,1])
