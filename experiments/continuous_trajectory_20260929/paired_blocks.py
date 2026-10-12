"""Paired year-stratified calendar-block resampling of fixed predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from trajectory import sha

HERE=Path(__file__).resolve().parent
BASE=HERE/'forward_runs/20260929T100914338806Z'
NEW=HERE/'offline_weather_runs/20260929T103927867552Z'


def counts(y,p):
    return np.column_stack([y&p,y&~p,~y&p,~y&~p]).astype(int)


def differences(c):
    a,b=c[:4],c[4:]
    return np.array([b[0]/(b[0]+b[1])-a[0]/(a[0]+a[1]),b[2]/(b[2]+b[3])-a[2]/(a[2]+a[3])])


def main():
    base=json.loads((BASE/'report.json').read_text('utf-8'))
    new=json.loads((NEW/'report.json').read_text('utf-8'))
    for root,r in ((BASE,base),(NEW,new)):
        for n,h in r['artifact_sha256'].items():
            assert sha(root/n)==h
    result=dict(scope='Conditional exploratory sensitivity, not post-selection significance or deployment evidence',replicates=1000,seed=20260929,
        hashes={str(p):sha(p) for p in (BASE/'report.json',NEW/'report.json',Path(__file__),HERE/'PAIRED_BLOCK_PROTOCOL.md')},tasks={})
    for task in ('onset','recovery'):
        result['tasks'][task]={}
        for length in (7,30):
            strata=[]
            for year in (2021,2022,2023):
                key=f'{year}_{task}'
                meta=pd.read_parquet(BASE/f'{key}_origins.parquet')
                mask=meta.partition.eq('test').to_numpy()
                y=meta.label.to_numpy(bool)[mask]
                a=base['folds'][key]['chosen']; b=new['folds'][key]['chosen']
                pa=np.load(BASE/f'{key}_{a["candidate"]}.npy')[mask]>=a['threshold']
                pb=np.load(NEW/f'{key}_{b["candidate"]}.npy')[mask]>=b['threshold']
                rows=np.column_stack([counts(y,pa),counts(y,pb)])
                block=(meta.loc[mask,'timestamp'].dt.dayofyear.to_numpy()-1)//length
                strata.append(np.array([rows[block==i].sum(axis=0) for i in np.unique(block)]))
            point=differences(sum(s.sum(axis=0) for s in strata))
            rng=np.random.default_rng(20260929)
            draws=[]
            for _ in range(1000):
                c=sum(s[rng.integers(len(s),size=len(s))].sum(axis=0) for s in strata)
                if all(c[i]+c[i+1]>0 for i in (0,2,4,6)):
                    draws.append(differences(c))
            quantiles=np.quantile(np.array(draws),[.025,.975],axis=0)
            result['tasks'][task][str(length)]=dict(blocks_per_year=[len(s) for s in strata],valid_draws=len(draws),
                recall_delta=float(point[0]),fpr_delta=float(point[1]),recall_delta_interval=quantiles[:,0].tolist(),fpr_delta_interval=quantiles[:,1].tolist())
    (NEW/'paired_block_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n','utf-8')
    print(json.dumps(result['tasks'],indent=2))


if __name__=='__main__':
    main()
