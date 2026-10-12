import numpy as np
import pandas as pd
import pytest
from issue_delay_features import build
from trajectory import features


def fixture():
    ts=pd.date_range('2019-12-30',periods=120,freq='h',tz='Asia/Shanghai')
    raw=pd.DataFrame({'timestamp':ts,'pm2_5':np.arange(120.),'pm10':np.arange(120.)+10,'aqi':np.arange(120.)+20})
    idx=np.array([35,71,95])
    h=raw[['pm2_5','pm10','aqi']].to_numpy()[idx[:,None]+np.arange(-23,1)]
    return raw,dict(indices=idx,timestamps=ts[idx],history=h)


def test_zero_lag_matches_frozen_baseline():
    raw,w=fixture()
    np.testing.assert_array_equal(build(raw,w,2019,0,False),features(w['history'],w['timestamps']))


def test_issue_calendar_and_source_cutoff():
    raw,w=fixture()
    x=build(raw,w,2019,3,True)
    assert x.shape==(3,139) and x.dtype==np.float32
    base=features(w['history'],w['timestamps'])
    np.testing.assert_array_equal(x[:,72:76],base[:,72:76])
    np.testing.assert_array_equal(x[-1,69:72],raw.loc[92,['pm2_5','pm10','aqi']].to_numpy(float))
    changed=raw.copy()
    changed.loc[93:,'aqi']=9999
    np.testing.assert_array_equal(build(changed,w,2019,3,True)[-1],x[-1])


def test_no_cross_partition_or_negative_wrap():
    raw,w=fixture()
    x=build(raw,w,2019,3,False)
    assert np.isnan(x[1,:9]).all()
    assert np.isfinite(x[1,9:]).all()
    w=dict(indices=np.array([23]),timestamps=raw.timestamp.iloc[[23]])
    assert np.isnan(build(raw,w,2019,3,False)[0,:9]).all()


def test_invalid_source_values_stay_missing():
    raw,w=fixture(); raw.loc[92,'aqi']=-1
    assert np.isnan(build(raw,w,2019,3,False)[-1,71])


def test_negative_lag_rejected():
    raw,w=fixture()
    with pytest.raises(ValueError): build(raw,w,2019,-1,False)
