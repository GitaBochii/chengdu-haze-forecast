"""Exploratory paired calendar-block uncertainty, not a fresh blind test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block_ci(dates,differences,blockdays,reps=1000):
    ts=pd.DatetimeIndex(dates)
    delta=np.asarray(differences,float)
    if len(ts)!=len(delta) or not len(delta) or not np.isfinite(delta).all() or blockdays<1 or reps<1:
        raise ValueError('Finite aligned paired losses and positive block settings required')
    table=pd.DataFrame({'year':ts.year,'block':(ts.dayofyear-1)//blockdays,'delta':delta})
    blocks=table.groupby(['year','block']).delta.agg(['sum','count'])
    rng=np.random.default_rng(42)
    numerator=np.zeros(reps); denominator=np.zeros(reps)
    for year in blocks.index.get_level_values(0).unique():
        values=blocks.loc[year].to_numpy(float)
        sampled=rng.integers(0,len(values),size=(reps,len(values)))
        numerator+=values[sampled,0].sum(axis=1)
        denominator+=values[sampled,1].sum(axis=1)
    return {'delta_mae':float(delta.mean()),'ci95':np.quantile(numerator/denominator,[.025,.975]).tolist(),
            'block_days':blockdays,'blocks':len(blocks),'replicates':reps,'seed':42}


def main(run):
    report=json.loads((run/'report.json').read_text('utf-8'))
    if report['status']!='completed_fixed_protocol':
        raise ValueError('Completed experiment required')
    checked={str(run/'report.json'):sha(run/'report.json'),str(Path(__file__).resolve()):sha(__file__)}
    for name,expected in report['artifact_sha256'].items():
        if sha(run/name)!=expected:
            raise ValueError('Artifact changed: '+name)
        checked[str(run/name)]=expected
    dates=[]; losses={arm:[] for arm in ('lag0_base','lag1_base','lag1_trend','lag3_base','lag3_trend')}
    for year in (2021,2022,2023):
        meta=pd.read_parquet(run/f'{year}_origins.parquet'); keep=meta.partition.eq('test').to_numpy()
        dates.extend(meta.timestamp[keep].tolist())
        target=np.load(run/f'{year}_targets.npy')[keep].astype(float)
        for arm in losses:
            prediction=np.load(run/f'{year}_{arm}.npy')[keep].astype(float)
            losses[arm].append(np.abs(prediction-target).mean(axis=1))
    losses={k:np.concatenate(v) for k,v in losses.items()}
    out={'scope':'Exploratory paired mean absolute loss across h1/h6/h24, year-stratified fixed calendar blocks; negative delta favors first arm; not blind or independent events',
         'inputs_sha256':checked,'comparisons':{}}
    for first,second in [('lag1_trend','lag1_base'),('lag3_trend','lag3_base'),('lag1_base','lag0_base'),('lag3_base','lag0_base')]:
        out['comparisons'][first+' - '+second]=[block_ci(dates,losses[first]-losses[second],days) for days in (7,30)]
    if not all(sha(p)==h for p,h in checked.items()):
        raise ValueError('Source changed during analysis')
    with (run/'paired_uncertainty.json').open('x',encoding='utf-8') as stream:
        json.dump(out,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(out['comparisons'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    main(parser.parse_args().run.resolve())
