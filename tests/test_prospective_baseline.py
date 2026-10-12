import numpy as np
import pandas as pd
import pytest

from haze_forecast.prospective_baseline import history_features, predict, baseline_manifest


def frame():
    return pd.DataFrame({'timestamp': pd.date_range('2026-09-29', periods=24, freq='h', tz='Asia/Shanghai'),
                         'pm2_5': np.arange(24), 'pm10': np.arange(24)+20, 'aqi': np.arange(24)+50})


def test_feature_contract():
    f=frame()
    x=history_features(f, f.timestamp.iloc[-1])
    assert x.shape == (1,76) and x.dtype == np.float32
    np.testing.assert_array_equal(x[0,:72], f[['pm2_5','pm10','aqi']].to_numpy().ravel())


@pytest.mark.parametrize('fault', ['gap','future','missing','negative','naive','duplicate','overflow'])
def test_reject_bad_history(fault):
    f=frame(); origin=f.timestamp.iloc[-1]
    if fault=='gap': f=f.iloc[:-1]
    if fault=='future': origin=origin-pd.Timedelta(hours=1)
    if fault=='missing': f.loc[3,'aqi']=np.nan
    if fault=='negative': f.loc[3,'pm10']=-1
    if fault=='naive': f['timestamp']=f.timestamp.dt.tz_localize(None)
    if fault=='duplicate': f.loc[2,'timestamp']=f.loc[1,'timestamp']
    if fault=='overflow':
        f['aqi']=f.aqi.astype(float)
        f.loc[2,'aqi']=1e100
    with pytest.raises(ValueError): history_features(f,origin)


def test_reject_unbound_model_and_wrong_contract():
    import lightgbm as lgb
    m={'lightgbm_version':lgb.__version__,'files':{},'tasks':{},
       'kind':'wrong','feature_order':['aqi'],'horizon_hours':12}
    with pytest.raises(ValueError,match='contract'):
        predict(m,frame(),frame().timestamp.iloc[-1])


def test_frozen_selection_binding_and_historical_parity():
    from pathlib import Path
    from copy import deepcopy
    root=Path(__file__).resolve().parents[1]
    run=root/'experiments/continuous_trajectory_20260929/probability_runs/20260929T094914341676Z'
    if not (run/'report.json').exists():
        pytest.skip('Local frozen research artifacts unavailable')
    manifest=baseline_manifest(root)
    raw=pd.read_parquet(root/'data/processed/regional_air_models_full_history/city_hourly.parquet')
    for task in ('onset','recovery'):
        meta=pd.read_parquet(run/f'{task}_origins.parquet')
        weight='balanced' if task=='onset' else 'unweighted'
        probabilities=np.load(run/f'{task}_{weight}.npy')
        for row in np.linspace(0,len(meta)-1,8,dtype=int):
            i=int(meta.source_index.iloc[row]); f=raw.iloc[i-23:i+1]
            result=predict(manifest,f,f.timestamp.iloc[-1])
            assert result[task+'_probability']==probabilities[row]
    tampered=deepcopy(manifest)
    tampered['tasks']['onset']['threshold']=0.01
    with pytest.raises(ValueError,match='selection'): predict(tampered,f,f.timestamp.iloc[-1])
    tampered=deepcopy(manifest)
    tampered['tasks']['onset']['model_path']='unbound.txt'
    with pytest.raises(ValueError,match='selection'): predict(tampered,f,f.timestamp.iloc[-1])
