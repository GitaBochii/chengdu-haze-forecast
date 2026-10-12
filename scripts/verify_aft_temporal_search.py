"""Verify all candidates and the declared ranking without retraining or selecting anew."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[1]


def main():
    b=ROOT/'artifacts/task01_recovery_head/aft_temporal_search_20260918'
    c=json.loads((b/'protocol.json').read_bytes()); r=json.loads((b/'report.json').read_bytes())
    for p,h in c['input_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    for p,h in r['output_sha256'].items():
        assert hashlib.sha256((b/p).read_bytes()).hexdigest()==h
    keys=['exact_row_mae','exact_block_mae','tail24_block_mae']; independently={}; count=0
    for year in c['years']:
        d=pd.read_parquet(b/f'{year}_predictions.parquet'); t=pd.read_parquet(b/f'{year}_training.parquet')
        cut=pd.Timestamp(f'{year}-01-01',tz='Asia/Shanghai')
        assert t.timestamp.lt(cut).all() and t.label_available_at.lt(cut).all()
        assert not set(t.continuity_block)&set(d.continuity_block)
        x=xgb.DMatrix(d[c['features']].astype(float)); independently[str(year)]={}
        for i,config in enumerate(c['configs']):
            m=xgb.Booster(); m.load_model(b/f'{year}_{i}.ubj'); pred=m.predict(x)
            np.testing.assert_array_equal(pred,d[f'candidate_{i}']); count+=len(d)
            e=d.loc[d.exact_hours.notna()].copy(); e['error']=(e[f'candidate_{i}']-e.exact_hours).abs()
            values=[e.error.mean(),e.groupby('continuity_block').error.mean().mean(),e.loc[e.exact_hours>=24].groupby('continuity_block').error.mean().mean()]
            np.testing.assert_allclose(values,[r['scores'][str(i)][str(year)][k] for k in keys],atol=1e-12)
            independently[str(year)][i]=values
    ranks=[]
    for i,config in enumerate(c['configs']):
        ratios=np.array([independently[str(y)][i] for y in c['years']])/np.array([independently[str(y)][r['baseline_id']] for y in c['years']])
        score=float(ratios.mean()); ranks.append((score,config['max_depth'],config['min_child_weight'],config['lambda'],i))
        saved=next(v for v in r['ranking'] if v['id']==i)
        np.testing.assert_allclose(ratios.ravel(),saved['nine_ratios'],atol=1e-12)
        np.testing.assert_allclose(score,saved['normalized_mean'],atol=1e-12)
    assert min(ranks)[-1]==r['selected']['id']
    print(count,'predictions replayed; 108 primary metrics, all 12 rankings, time/process isolation and hashes passed')


if __name__=='__main__':
    main()
