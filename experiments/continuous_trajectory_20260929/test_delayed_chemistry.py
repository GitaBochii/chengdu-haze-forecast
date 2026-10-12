import numpy as np
import pandas as pd
from delayed_chemistry import build
from issue_delay_features import build as baseline


def fixture():
    ts=pd.date_range('2020-12-30',periods=100,freq='h',tz='Asia/Shanghai')
    raw=pd.DataFrame({'timestamp':ts})
    for column in ('pm2_5','pm10','aqi','so2','no2','co','o3_8h'):
        raw[column]=np.arange(100,dtype=float)
    idx=np.array([0,24,48,74])
    return raw,dict(indices=idx,timestamps=ts[idx])


def test_delayed_extra_history_masks_and_baseline_preservation():
    raw,w=fixture()
    x=build(raw,w,2020,3)
    assert x.shape==(4,323) and x.dtype==np.float32
    np.testing.assert_array_equal(x[:,:139],baseline(raw,w,2020,3,True))
    hist=x[:,139:235].reshape(4,24,4)
    assert np.isnan(hist[0]).all()
    assert np.isnan(hist[1,:2]).all()
    np.testing.assert_array_equal(hist[1,2:,0],np.arange(22))
    assert np.isnan(hist[2]).all()
    np.testing.assert_array_equal(hist[3,:,0],np.arange(48,72))
    np.testing.assert_array_equal(x[:,-4:],np.repeat([[0],[22/24],[0],[1]],4,axis=1).astype('float32'))


def test_future_extras_never_affect_features_and_bad_values_stay_missing():
    raw,w=fixture()
    x=build(raw,w,2020,1)
    raw.loc[74:,'so2']=999999
    np.testing.assert_array_equal(x,build(raw,w,2020,1))
    raw.loc[73,'co']=-1
    raw.loc[72,'no2']=np.inf
    h=build(raw,w,2020,1)[-1,139:235].reshape(24,4)
    assert np.isnan(h[-1,2]) and np.isnan(h[-2,1])
