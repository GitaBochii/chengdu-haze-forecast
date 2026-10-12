"""Matched historical chemistry-vs-trend error differences at fixed delays."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from issue_delay_uncertainty import block_ci,sha

HERE=Path(__file__).resolve().parent
BASE=HERE/'issue_delay_runs/20260929T133825296179Z'


def main(run):
    checked={str(Path(__file__).resolve()):sha(__file__),
             str(HERE/'issue_delay_uncertainty.py'):sha(HERE/'issue_delay_uncertainty.py')}
    for directory in (BASE,run):
        report=json.loads((directory/'report.json').read_text('utf-8'))
        if report['status']!='completed_fixed_protocol':
            raise ValueError('Completed experiment required')
        checked[str(directory/'report.json')]=sha(directory/'report.json')
        for name,expected in report['artifact_sha256'].items():
            if sha(directory/name)!=expected:
                raise ValueError('Artifact changed: '+name)
            checked[str(directory/name)]=expected
    out={'scope':'Exploratory paired chemistry minus trend MAE in AQI units; same delay, all test origins, equal weight h1/h6/h24; not blind or prospective',
         'input_sha256':checked,'comparisons':{}}
    for lag in (1,3):
        dates=[];differences=[]
        for year in (2021,2022,2023):
            old=pd.read_parquet(BASE/f'{year}_origins.parquet')
            new=pd.read_parquet(run/f'{year}_origins.parquet')
            pd.testing.assert_frame_equal(old,new)
            target=np.load(BASE/f'{year}_targets.npy')
            np.testing.assert_array_equal(target,np.load(run/f'{year}_targets.npy'))
            mask=old.partition.eq('test').to_numpy()
            target=target[mask].astype(float)
            before=np.load(BASE/f'{year}_lag{lag}_trend.npy')[mask].astype(float)
            after=np.load(run/f'{year}_lag{lag}_chemistry.npy')[mask].astype(float)
            dates.extend(old.timestamp[mask].tolist())
            differences.append((np.abs(after-target)-np.abs(before-target)).mean(axis=1))
        delta=np.concatenate(differences)
        out['comparisons'][f'lag{lag}']=[block_ci(dates,delta,days) for days in (7,30)]
    if not all(sha(p)==h for p,h in checked.items()):
        raise ValueError('Inputs changed')
    with (run/'paired_uncertainty.json').open('x',encoding='utf-8') as stream:
        json.dump(out,stream,indent=2,allow_nan=False)
    print(json.dumps(out['comparisons'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    main(parser.parse_args().run.resolve())
