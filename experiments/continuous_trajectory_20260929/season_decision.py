"""Season-aware decisions selected entirely on each fold's development year."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from constrained import choose
from probability import binary_metrics
from feature_ablation import pooled
from trajectory import sha

HERE=Path(__file__).resolve().parent
BASE=HERE/'forward_runs/20260929T100914338806Z'
FEATURE=HERE/'feature_runs/20260929T102202428649Z'
SPATIAL=HERE/'spatial_runs/20260929T102734925552Z'


def sufficient(y):
    return bool(y.sum()>=20 and (~y).sum()>=50)


def decisions_score(y,pred):
    m=binary_metrics(y,pred.astype(float),.5)
    m.pop('brier')
    m.pop('average_precision')
    return m


def main():
    identities={}
    for root in (BASE,FEATURE,SPATIAL):
        report=json.loads((root/'report.json').read_text('utf-8'))
        identities[str(root/'report.json')]=sha(root/'report.json')
        for n,h in report['artifact_sha256'].items():
            assert sha(root/n)==h
    for n in ('season_decision.py','SEASON_PROTOCOL.md','constrained.py','probability.py','feature_ablation.py','trajectory.py','event_audit.py'):
        identities[str(HERE/n)]=sha(HERE/n)
    baseline=json.loads((BASE/'report.json').read_text('utf-8'))
    output=HERE/'season_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    out=dict(status='completed',hashes=identities,strategies={},deployed=False)
    for strategy in ('baseline_season','full_season'):
        folds={}
        for key,old in baseline['folds'].items():
            meta=pd.read_parquet(BASE/f'{key}_origins.parquet')
            dev=meta.partition.eq('dev').to_numpy()
            test=~dev
            y=meta.label.to_numpy(bool)
            candidates={}
            for arm in ('baseline','trend','chemistry','spatial'):
                if strategy=='baseline_season' and arm!='baseline':
                    continue
                for weight in ('unweighted','balanced'):
                    path=(BASE/f'{key}_{weight}.npy' if arm=='baseline' else SPATIAL/f'{key}_{weight}.npy' if arm=='spatial' else FEATURE/f'{key}_{arm}_{weight}.npy')
                    candidates[f'{arm}_{weight}']=np.load(path)
            global_best,_=choose(y[dev],{n:p[dev] for n,p in candidates.items()},old['budget'])
            decisions=np.zeros(len(y),bool)
            choices={}
            for season in ('DJF','MAM','JJA','SON'):
                group=meta.season.eq(season).to_numpy()
                m=dev&group
                fallback=not sufficient(y[m])
                if fallback:
                    best=global_best
                    grid=[]
                else:
                    best,grid=choose(y[m],{n:p[m] for n,p in candidates.items()},old['budget'])
                decisions[group]=candidates[best['candidate']][group]>=best['threshold']
                mt=test&group
                choices[season]=dict(fallback=fallback,dev_positive=int(y[m].sum()),dev_negative=int((~y[m]).sum()),
                    chosen=best,development_grid=grid,test=decisions_score(y[mt],decisions[mt]) if mt.any() else None)
            np.save(output/f'{strategy}_{key}_decisions.npy',decisions,allow_pickle=False)
            folds[key]=dict(global_choice=global_best,seasons=choices,test=decisions_score(y[test],decisions[test]))
        totals={}
        for task,budget in [('onset',.1),('recovery',.05)]:
            keys=[k for k in folds if k.endswith('_'+task)]
            b=pooled([baseline['folds'][k]['diagnostics']['test']['overall'] for k in keys])
            new=pooled([folds[k]['test'] for k in keys])
            years=sum(folds[k]['test']['recall']>baseline['folds'][k]['diagnostics']['test']['overall']['recall'] for k in keys)
            totals[task]=dict(baseline=b,new=new,years_recall_improved=years,
                meaningful_gate=bool(new['recall']>=b['recall']+.05 and new['false_positive_rate']<=min(budget,b['false_positive_rate']) and years>=2))
        out['strategies'][strategy]=dict(folds=folds,pooled=totals)
        print(json.dumps(dict(strategy=strategy,pooled=totals),indent=2),flush=True)
    assert all(sha(Path(p))==h for p,h in identities.items())
    out['artifact_sha256']={p.name:sha(p) for p in output.iterdir()}
    (output/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n','utf-8')
    print(str(output),flush=True)


if __name__=='__main__':
    main()
