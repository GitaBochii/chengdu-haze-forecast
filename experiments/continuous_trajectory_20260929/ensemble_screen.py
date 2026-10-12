"""Fixed score combinations; all choices are made on development rows."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from season_decision import BASE,FEATURE,SPATIAL
from constrained import choose
from forward import diagnostic
from feature_ablation import pooled
from trajectory import sha


def main():
    here=Path(__file__).resolve().parent
    hashes={str(p):sha(p) for p in [Path(__file__),here/'ENSEMBLE_PROTOCOL.md',*[r/'report.json' for r in (BASE,FEATURE,SPATIAL)]]}
    old=json.loads((BASE/'report.json').read_text('utf-8'))
    for root in (BASE,FEATURE,SPATIAL):
        rr=json.loads((root/'report.json').read_text('utf-8'))
        for n,h in rr['artifact_sha256'].items():
            assert sha(root/n)==h
    output=here/'ensemble_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    out=dict(status='completed',hashes=hashes,folds={},pooled={},deployed=False)
    for key,b in old['folds'].items():
        meta=pd.read_parquet(BASE/f'{key}_origins.parquet')
        dev=meta.partition.eq('dev').to_numpy()
        candidates={}
        for weight in ('unweighted','balanced'):
            stack=np.stack([np.load(BASE/f'{key}_{weight}.npy'),np.load(FEATURE/f'{key}_chemistry_{weight}.npy'),np.load(SPATIAL/f'{key}_{weight}.npy')])
            candidates['mean_'+weight]=stack.mean(axis=0)
            candidates['median_'+weight]=np.median(stack,axis=0)
        for n,p in candidates.items():
            np.save(output/f'{key}_{n}.npy',p,allow_pickle=False)
        best,grid=choose(meta.label.to_numpy()[dev],{n:p[dev] for n,p in candidates.items()},b['budget'])
        out['folds'][key]=dict(chosen=best,development_grid=grid,diagnostics={})
        for part in ('dev','test'):
            m=meta.partition.eq(part).to_numpy()
            out['folds'][key]['diagnostics'][part]=diagnostic(meta.loc[m],candidates[best['candidate']][m],best['threshold'],b['train_positive']/b['train_n'])
    for task,budget in [('onset',.1),('recovery',.05)]:
        keys=[k for k in out['folds'] if k.endswith('_'+task)]
        b=pooled([old['folds'][k]['diagnostics']['test']['overall'] for k in keys])
        new=pooled([out['folds'][k]['diagnostics']['test']['overall'] for k in keys])
        years=sum(out['folds'][k]['diagnostics']['test']['overall']['recall']>old['folds'][k]['diagnostics']['test']['overall']['recall'] for k in keys)
        out['pooled'][task]=dict(baseline=b,new=new,years_recall_improved=years,
            meaningful_gate=bool(new['recall']>=b['recall']+.05 and new['false_positive_rate']<=min(budget,b['false_positive_rate']) and years>=2))
    assert all(sha(Path(p))==h for p,h in hashes.items())
    out['artifact_sha256']={p.name:sha(p) for p in output.iterdir()}
    (output/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps(out['pooled'],indent=2));print(str(output))


if __name__=='__main__':
    main()
