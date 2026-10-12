import numpy as np
import pytest
from discrete_hazard import risk_rows, hazard_decode


def test_event_and_censored_risk_sets_keep_all_origins():
    origin,lead,event = risk_rows(np.array([1,3,20]))
    assert len(lead)==23
    np.testing.assert_array_equal(np.bincount(origin),[1,3,19])
    np.testing.assert_array_equal(lead[:4],[1,1,2,3])
    np.testing.assert_array_equal(event[:4],[1,0,0,1])
    assert event[4:].sum()==0


def test_cdf_is_coherent_and_capped_median_keeps_censored_mass():
    h = np.zeros((3,19))
    h[0,2] = 1
    h[1,:2] = [.4,.5]
    prediction,cdf = hazard_decode(h,.5)
    np.testing.assert_array_equal(prediction,[3,2,20])
    assert cdf[1,1]==.7
    assert (np.diff(cdf,axis=1)>=0).all()
    assert cdf[2].sum()==0
    with pytest.raises(ValueError):
        hazard_decode(np.full((1,19),np.nan),.5)
