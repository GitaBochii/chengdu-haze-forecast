"""One finite historical selection budget; never read 2023+ evaluation predictions."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary

ROOT=Path(__file__).resolve().parents[1]
KEYS=['exact_row_mae','exact_block_mae','tail24_block_mae']


def main():
    b=ROOT/'artifacts/task01_recovery_head'
    out=b/'aft_temporal_search_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths=[b/'gefs_2021_extension_v1/f024_training_rows.parquet',
           b/'grouped_forecast_aft_2023_20260916/protocol.json',Path(__file__).resolve(),
           ROOT/'aft_temporal_search_plan_20260918.md']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    d=pd.read_parquet(paths[0]); frozen=json.loads(paths[1].read_bytes()); names=frozen['features']
    configs=[{'max_depth':a,'min_child_weight':b,'lambda':c} for a,b,c in itertools.product([2,3,4],[8,24],[3.,10.])]
    out.mkdir()
    (out/'protocol.json').write_text(json.dumps({'input_sha256':hashes,'configs':configs,'features':names,
        'base_params':frozen['params'],'rounds':500,'years':[2020,2021,2022]},indent=2),encoding='utf-8')
    scores={str(i):{} for i in range(len(configs))}; paired={}
    baseline_id=next(i for i,c in enumerate(configs) if c=={'max_depth':4,'min_child_weight':8,'lambda':3.})
    for year in (2020,2021,2022):
        cut=pd.Timestamp(f'{year}-01-01',tz='Asia/Shanghai'); end=pd.Timestamp(f'{year+1}-01-01',tz='Asia/Shanghai')
        v=d.loc[d.timestamp.ge(cut)&d.timestamp.lt(end)&d.label_available_at.lt(end)].copy()
        t=d.loc[d.timestamp.lt(cut)&d.label_available_at.lt(cut)&~d.continuity_block.isin(v.continuity_block)]
        assert not set(t.continuity_block)&set(v.continuity_block)
        t[['timestamp','continuity_block','label_available_at']].to_parquet(out/f'{year}_training.parquet',index=False)
        matrix=xgb.DMatrix(t[names].astype(float)); infer=xgb.DMatrix(v[names].astype(float))
        matrix.set_float_info('label_lower_bound',t.lower_bound_hours); matrix.set_float_info('label_upper_bound',t.upper_bound_hours)
        for i,config in enumerate(configs):
            model=xgb.train({**frozen['params'],**config},matrix,num_boost_round=500)
            v[f'candidate_{i}']=model.predict(infer)
            model.save_model(out/f'{year}_{i}.ubj')
            scores[str(i)][str(year)]=score_predictions(v,v[f'candidate_{i}'].to_numpy())
            print(year,i,'completed',flush=True)
        paired[str(year)]={str(i):paired_error_summary(v,v[f'candidate_{baseline_id}'],v[f'candidate_{i}']) for i in range(len(configs))}
        v.to_parquet(out/f'{year}_predictions.parquet',index=False)
    ranking=[]
    for i,config in enumerate(configs):
        ratios=[scores[str(i)][str(y)][k]/scores[str(baseline_id)][str(y)][k] for y in (2020,2021,2022) for k in KEYS]
        ranking.append({'id':i,'config':config,'normalized_mean':float(np.mean(ratios)),'nine_ratios':ratios})
    ranking.sort(key=lambda r:(r['normalized_mean'],r['config']['max_depth'],r['config']['min_child_weight'],r['config']['lambda']))
    for p,h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    report={'scores':scores,'paired':paired,'ranking':ranking,'selected':ranking[0], 'baseline_id':baseline_id,
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(ranking,indent=2),flush=True)


if __name__=='__main__':
    main()
