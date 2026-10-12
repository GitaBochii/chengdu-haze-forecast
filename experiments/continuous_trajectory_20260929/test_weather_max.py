import numpy as np
import pandas as pd
from weather_max import weather_history


def test_weather_excludes_issue_hour_and_keeps_missing():
    ts=pd.date_range('2020-01-01',periods=60,freq='h',tz='Asia/Shanghai')
    g=np.arange(300,dtype=float).reshape(60,5)
    g[20,0]=np.nan
    h=weather_history(g,ts,np.array([30]),2019)
    assert h.shape==(1,24,5)
    np.testing.assert_array_equal(h[0,-1],g[29])
    before=h.copy()
    g[30:]=999
    np.testing.assert_array_equal(weather_history(g,ts,np.array([30]),2019),before)
    assert np.isnan(h[0,14,0])


def test_year_boundary_is_masked_not_filled():
    ts=pd.date_range('2019-12-31 12:00',periods=40,freq='h',tz='Asia/Shanghai')
    h=weather_history(np.ones((40,5)),ts,np.array([13]),2019)
    assert np.isnan(h[0,:23]).all()
    assert (h[0,-1]==1).all()
