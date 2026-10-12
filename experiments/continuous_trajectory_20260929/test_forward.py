import numpy as np
import pandas as pd
from forward import fold_parts, groups, calibration


def test_boundaries_purge_history_and_future():
    ts = pd.DatetimeIndex(['2019-12-30 23:00','2019-12-31 00:00','2020-01-01 22:00','2020-01-01 23:00','2020-12-31 00:00','2021-01-01 23:00'],tz='Asia/Shanghai')
    assert fold_parts(ts,2019).tolist() == ['train','excluded','excluded','dev','excluded','test']


def test_group_edges_and_missing_station_count():
    g=groups(pd.DatetimeIndex(['2021-01-01','2021-04-01','2021-07-01','2021-10-01']),np.array([50,100,150,151]),np.array([6,8,9,np.nan]))
    assert g['season'].tolist() == ['DJF','MAM','JJA','SON']
    assert g['intensity'].tolist() == ['<=50','50-100','100-150','>150']
    assert g['stations'].tolist() == ['<=6','7-8','>=9','missing']


def test_calibration_includes_one_and_empty_bins():
    c=calibration(np.array([0,1]),np.array([0.,1.]))
    assert sum(r['n'] for r in c)==2
    assert c[0]['observed_rate']==0
    assert c[-1]['observed_rate']==1
    assert c[1]['observed_rate'] is None
