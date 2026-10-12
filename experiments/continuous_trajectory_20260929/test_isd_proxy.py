import numpy as np
import pandas as pd
from isd_proxy import delayed_weather


def test_delay_is_exact_and_no_wrap_or_boundary_leak():
    ts=pd.date_range('2019-12-27',periods=240,freq='h',tz='Asia/Shanghai')
    grid=np.tile(np.arange(len(ts),dtype=float)[:,None],(1,5))
    x=delayed_weather(grid,ts,np.array([23,150,215]),2019,72)
    assert np.isnan(x[0]).all()
    assert np.isnan(x[1]).all()
    np.testing.assert_array_equal(x[2,:,0],np.arange(120,144))


def test_future_edits_do_not_change_input():
    ts=pd.date_range('2018-01-01',periods=150,freq='h',tz='Asia/Shanghai')
    grid=np.ones((150,5))
    a=delayed_weather(grid,ts,np.array([100]),2019,72)
    grid[29:]=999
    np.testing.assert_array_equal(a,delayed_weather(grid,ts,np.array([100]),2019,72))
