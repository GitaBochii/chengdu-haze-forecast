"""Persistence and paired uncertainty for the fixed full-curve experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from full_curve_recovery import diagnostics
from event_audit import transition_times
from issue_delay_uncertainty import block_ci,sha

HERE=Path(__file__).resolve().parent
BASE=HERE/'issue_delay_runs/20260929T133825296179Z'


def main(run):
    checked={str(p):sha(p) for p in (Path(__file__).resolve(),HERE/'event_audit.py',
              HERE/'full_curve_recovery.py',HERE/'issue_delay_uncertainty.py')}
    for directory in (run,BASE):
        r=json.loads((directory/'report.json').read_text('utf-8'))
        if r['status']!='completed_fixed_protocol':
            raise ValueError('Completed experiment required')
        checked[str(directory/'report.json')]=sha(directory/'report.json')
        for name,h in r['artifact_sha256'].items():
            if sha(directory/name)!=h:
                raise ValueError('Artifact changed')
            checked[str(directory/name)]=h
    target=[]; persistence=[]; origins=[]; dates=[]; predictions={'lag1_trend':[],'lag1_chemistry':[]}
    for year in (2021,2022,2023):
        meta=pd.read_parquet(run/f'{year}_origins.parquet')
        pd.testing.assert_frame_equal(meta,pd.read_parquet(BASE/f'{year}_origins.parquet'))
        mask=meta.partition.eq('test').to_numpy()
        target.append(np.load(run/f'{year}_targets.npy')[mask])
        persistence.append(np.repeat(np.load(BASE/f'{year}_persistence_lag1.npy')[mask,:1],24,axis=1))
        origins.extend(meta.origin_aqi[mask].tolist()); dates.extend(meta.timestamp[mask].tolist())
        for arm in predictions:
            predictions[arm].append(np.load(run/f'{year}_{arm}.npy')[mask])
    target=np.concatenate(target); persistence=np.concatenate(persistence); origins=np.asarray(origins)
    predictions={k:np.concatenate(v) for k,v in predictions.items()}
    dirty=origins>100; dates=pd.DatetimeIndex(dates)
    times={k:transition_times(v[dirty])[1] for k,v in predictions.items()}
    truth=transition_times(target[dirty])[1]
    dtime=np.abs(times['lag1_chemistry']-truth)-np.abs(times['lag1_trend']-truth)
    daqi=(np.abs(predictions['lag1_chemistry'].astype(float)-target)-np.abs(predictions['lag1_trend'].astype(float)-target)).mean(axis=1)
    result={'scope':'Exploratory paired chemistry minus trend; cap20 is censored score, not true full recovery duration; no blind/prospective claim',
            'sha256':checked,'persistence':diagnostics(target,persistence,origins),
            'capped_recovery_mae_delta':[block_ci(dates[dirty],dtime,days) for days in (7,30)],
            'full_curve_aqi_mae_delta':[block_ci(dates,daqi,days) for days in (7,30)]}
    if not all(sha(p)==h for p,h in checked.items()):
        raise ValueError('Dependency changed')
    with (run/'followup_analysis.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    main(parser.parse_args().run.resolve())
