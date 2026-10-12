import numpy as np
import pandas as pd
from issue_delay_uncertainty import block_ci


def test_constant_paired_difference():
    dates=pd.date_range('2021-01-01',periods=1440,freq='h',tz='Asia/Shanghai')
    out=block_ci(dates,np.full(len(dates),-.5),7,reps=20)
    assert out['delta_mae']==-.5 and out['ci95']==[-.5,-.5]


def test_years_are_separate_and_reproducible():
    dates=pd.DatetimeIndex(['2021-01-01T00:00:00+08:00','2022-01-01T00:00:00+08:00'])
    a=block_ci(dates,np.array([-1.,3.]),30,reps=20)
    assert a['delta_mae']==1. and a['ci95']==[1.,1.]
    assert a==block_ci(dates,np.array([-1.,3.]),30,reps=20)
