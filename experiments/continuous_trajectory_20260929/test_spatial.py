import numpy as np
import pandas as pd
from spatial_ablation import station_statistics, spatial_features


def test_station_missing_and_fraction():
    ts=pd.date_range('2020-01-01',periods=24,freq='h',tz='Asia/Shanghai')
    rows=pd.DataFrame(dict(timestamp=np.repeat(ts,2),station_id=['a','b']*24))
    for n in ['aqi','pm2_5','pm10','o3_8h']:
        rows[n]=[80.,120.]*24
    s=station_statistics(rows,ts)
    assert s.shape==(24,21)
    assert s[0,0]==90 and s[0,1]==110
    assert s[0,2]==120 and s[0,3]==20
    assert s[0,-1]==.5
    h=spatial_features(s,np.array([23]))
    assert h.shape==(1,252)
    np.testing.assert_array_equal(h[0,105:168],np.zeros(63))


def test_all_missing_not_zero_pollution():
    t=pd.date_range('2020-01-01',periods=1,tz='Asia/Shanghai')
    rows=pd.DataFrame(dict(timestamp=t,station_id=['a'],aqi=[np.nan],pm2_5=[-1.],pm10=[np.nan],o3_8h=[np.nan]))
    s=station_statistics(rows,t)
    assert np.isnan(s[0,-1])
    assert np.isnan(s[0,:16]).all()
    assert (s[0,16:20]==0).all()
