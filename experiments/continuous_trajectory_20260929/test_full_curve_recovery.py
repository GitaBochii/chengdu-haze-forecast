import numpy as np
import pytest
from full_curve_recovery import recovery_start, recovery_metrics


def test_first_six_hour_window_and_censoring():
    y=np.full((5,24),101.)
    y[0,:6]=100
    y[1,18:24]=100
    y[2,19:24]=100
    y[3,:5]=100
    y[3,6:12]=99
    y[4,:]=99
    np.testing.assert_array_equal(recovery_start(y),[1,19,20,7,1])


def test_censored_rows_and_missed_recoveries_stay_in_denominator():
    y=np.full((4,24),101.)
    p=y.copy()
    y[0,4:10]=90
    y[1,:6]=90
    p[0,2:8]=90
    p[2,:6]=90
    r=recovery_metrics(y,p)
    assert r['confusion']==dict(tp=1,fn=1,fp=1,tn=1)
    assert r['capped_min_T_20_mae_hours']==10
    assert r['premature_start_fraction_all']==.5
    assert r['premature_start_fraction_predicted_recovery']==1
    assert r['actual_right_censored']==2


def test_invalid_curve_is_rejected_not_silently_censored():
    with pytest.raises(ValueError):
        recovery_start(np.zeros((1,23)))
    with pytest.raises(ValueError):
        recovery_start(np.full((1,24),np.nan))
