"""Fixed physically motivated forecast features with a matched availability control."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary
from haze_forecast.ventilation_proxy import ventilation_proxy

ROOT = Path(__file__).resolve().parents[1]


def main():
    b = ROOT/'artifacts/task01_recovery_head'
    source = b/'gefs_2020_extension_2023_20260918'
    out = b/'ventilation_proxy_2023_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [source/n for n in ('protocol.json','training.parquet','predictions.parquet','scored.parquet')]
    paths += [b/'gefs_2021_extension_v1/inference_rows.parquet', Path(__file__).resolve(),
              ROOT/'src/haze_forecast/ventilation_proxy.py',ROOT/'ventilation_proxy_plan_20260918.md',
              ROOT/'src/haze_forecast/cohort_scorecard.py',ROOT/'src/haze_forecast/paired_interval_score.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    c=json.loads(paths[0].read_bytes())
    t=pd.read_parquet(paths[1]); v=pd.read_parquet(paths[4]); cohort=pd.read_parquet(paths[3])
    additions=[]
    for frame in (t,v):
        extra=ventilation_proxy(frame)
        independent=[]
        for u,w,h in zip(frame.gefs_mean_wind_u_ms_f024,frame.gefs_mean_wind_v_ms_f024,
                         frame.gefs_control_boundary_layer_height_m_f024):
            speed=math.sqrt(float(u)**2+float(w)**2)
            independent.append([speed,speed*float(h)])
        np.testing.assert_allclose(extra.to_numpy(),independent,rtol=1e-14,equal_nan=True)
        additions.append(extra)
    names=c['features']+list(additions[0].columns)
    out.mkdir()
    (out/'protocol.json').write_text(json.dumps({'input_sha256':hashes,'features':names,
        'params':c['params'],'rounds':500},indent=2),encoding='utf-8')
    prediction=pd.read_parquet(paths[2]).rename(columns={'candidate':'expanded_2020'})
    cohort=cohort.rename(columns={'candidate':'expanded_2020'})
    for arm in ('missing_control','transport'):
        extras=[x.copy() for x in additions]
        if arm=='missing_control':
            extras=[x.where(x.isna(),0.) for x in extras]
        x=pd.concat([t[c['features']],extras[0]],axis=1)
        z=pd.concat([v[c['features']],extras[1]],axis=1)
        dm=xgb.DMatrix(x.astype(float)); dm.set_float_info('label_lower_bound',t.lower_bound_hours)
        dm.set_float_info('label_upper_bound',t.upper_bound_hours)
        model=xgb.train(c['params'],dm,num_boost_round=500)
        model.save_model(out/f'{arm}.ubj')
        prediction[arm]=model.predict(xgb.DMatrix(z.astype(float)))
        restored=xgb.Booster(); restored.load_model(out/f'{arm}.ubj')
        np.testing.assert_array_equal(restored.predict(xgb.DMatrix(z.astype(float))),prediction[arm])
        cohort[arm]=prediction.set_index('timestamp').loc[cohort.timestamp,arm].to_numpy()
    arms=['native','extended_2021','expanded_2020','missing_control','transport']
    scores={n:score_predictions(cohort,cohort[n]) for n in arms}
    paired={n:paired_error_summary(cohort,cohort[n],cohort.transport) for n in arms[:-1]}
    a=scores['transport']
    gates={n:a['exact_row_mae']<scores[n]['exact_row_mae'] and a['exact_block_mae']<scores[n]['exact_block_mae']
           and a['tail24_block_mae']<=scores[n]['tail24_block_mae']
           and paired[n]['all_row_delta_upper']<=0 and paired[n]['all_block_delta_upper']<=0 for n in arms[:-1]}
    for n in arms:
        exact=cohort.loc[cohort.exact_hours.notna()]
        e=(exact[n]-exact.exact_hours).abs()
        np.testing.assert_allclose([e.mean(),e.groupby(exact.continuity_block).mean().mean(),
            e[exact.exact_hours>=24].groupby(exact.loc[exact.exact_hours>=24,'continuity_block']).mean().mean()],
            [scores[n][k] for k in ('exact_row_mae','exact_block_mae','tail24_block_mae')],atol=1e-12)
    prediction.to_parquet(out/'predictions.parquet',index=False)
    cohort.to_parquet(out/'scored.parquet',index=False)
    for p,h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    report={'scores':scores,'paired':paired,'gates':gates,'passed':all(gates.values()),
            'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'metrics':{n:[scores[n][k] for k in ('exact_row_mae','exact_block_mae','tail24_block_mae')] for n in arms},'gates':gates},indent=2))


if __name__=='__main__':
    main()
