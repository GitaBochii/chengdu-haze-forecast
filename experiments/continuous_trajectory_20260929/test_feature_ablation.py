import numpy as np
import pandas as pd
from feature_ablation import summaries, build_features


def test_summaries_linear_history():
    h=np.arange(24,dtype=float)[None,:,None]
    v=summaries(h)
    assert v.shape==(1,21)
    np.testing.assert_array_equal(v[0,:5],[1,3,6,12,23])
    assert v[0,5]==22
    assert v[0,7]==2
    assert v[0,8]==1


def test_no_future_dependency_and_auxiliary_missing():
    frame=pd.DataFrame({n:np.arange(50,dtype=float) for n in ['so2','no2','co','o3_8h']})
    h=np.ones((1,24,3))
    t=pd.DatetimeIndex(['2020-06-01'],tz='Asia/Shanghai')
    a=build_features(frame,np.array([23]),h,t)
    frame.loc[24:,:]=1e9
    b=build_features(frame,np.array([23]),h,t)
    for name in a:
        np.testing.assert_array_equal(a[name],b[name])
    frame.loc[:23,'co']=-1
    c=build_features(frame,np.array([23]),h,t)
    assert np.isnan(c['chemistry']).any()
    assert np.isfinite(c['trend']).all()
