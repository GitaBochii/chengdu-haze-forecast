"""Fixed exponential training-age weighting, not test-year threshold fitting."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from trajectory import windows,SOURCE,SOURCE_SHA,sha,save_booster,load_booster
from feature_ablation import build_features,pooled,BASE
from forward import fold_parts,diagnostic
from event_audit import transition_times
from constrained import choose

HERE=Path(__file__).resolve().parent


def recency_weights(ts,end,half_life):
    age=(end-pd.DatetimeIndex(ts)).total_seconds().to_numpy()/86400
    if (age<0).any() or half_life<=0:
        raise ValueError('Past-only positive half-life required')
    return 2.**(-age/(365.25*half_life))


def main():
    assert sha(SOURCE)==SOURCE_SHA
    deps=[SOURCE,BASE/'report.json',*[HERE/n for n in ('recency.py','RECENCY_PROTOCOL.md','feature_ablation.py','forward.py','trajectory.py','constrained.py','probability.py','event_audit.py')]]
    hashes={str(p):sha(p) for p in deps}
    old=json.loads((BASE/'report.json').read_text('utf-8'))
    for n,h in old['artifact_sha256'].items():
        assert sha(BASE/n)==h
    raw=pd.read_parquet(SOURCE)
    w=windows(raw)
    x=build_features(raw,w['indices'],w['history'],w['timestamps'])['chemistry']
    on,rec=transition_times(w['target'][:,:,2])
    output=HERE/'recency_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    report=dict(status='completed',source_sha256=hashes,folds={},pooled={},deployed=False)
    for cutoff in (2019,2020,2021):
        parts=fold_parts(w['timestamps'],cutoff)
        for task,eligible,label,budget in [('onset',~w['polluted'],on<25,.1),('recovery',w['polluted'],rec<20,.05)]:
            key=f'{cutoff+2}_{task}'
            train=eligible&(parts=='train')
            ev=eligible&np.isin(parts,['dev','test'])
            meta=pd.read_parquet(BASE/f'{key}_origins.parquet')
            np.testing.assert_array_equal(meta.source_index,w['indices'][ev])
            np.testing.assert_array_equal(meta.label,label[ev])
            meta.to_parquet(output/f'{key}_origins.parquet',index=False)
            candidates={}
            for half in (2,4):
                weights=recency_weights(w['timestamps'][train],pd.Timestamp(f'{cutoff+1}-01-01',tz='Asia/Shanghai'),half)
                for name,cw in [('unweighted',None),('balanced','balanced')]:
                    candidate=f'half{half}_{name}'
                    model=lgb.LGBMClassifier(objective='binary',class_weight=cw,n_estimators=300,learning_rate=.04,
                        num_leaves=24,min_child_samples=30,random_state=42,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
                    model.fit(x[train],label[train],sample_weight=weights)
                    p=model.predict_proba(x[ev])[:,1]
                    path=output/f'{key}_{candidate}.txt'
                    save_booster(model.booster_,path)
                    np.testing.assert_array_equal(p,load_booster(path).predict(x[ev],num_threads=4))
                    np.save(output/f'{key}_{candidate}.npy',p,allow_pickle=False)
                    candidates[candidate]=p
            dev=meta.partition.eq('dev').to_numpy()
            best,grid=choose(label[ev][dev],{n:p[dev] for n,p in candidates.items()},budget)
            record=dict(train_n=int(train.sum()),train_positive=int(label[train].sum()),budget=budget,chosen=best,development_grid=grid,diagnostics={})
            for part in ('dev','test'):
                m=meta.partition.eq(part).to_numpy()
                record['diagnostics'][part]=diagnostic(meta.loc[m],candidates[best['candidate']][m],best['threshold'],float(label[train].mean()))
            report['folds'][key]=record
            print(json.dumps(dict(fold=key,chosen=best['candidate'],threshold=best['threshold'],test=record['diagnostics']['test']['overall'])),flush=True)
    for task,budget in [('onset',.1),('recovery',.05)]:
        keys=[k for k in report['folds'] if k.endswith('_'+task)]
        b=pooled([old['folds'][k]['diagnostics']['test']['overall'] for k in keys])
        new=pooled([report['folds'][k]['diagnostics']['test']['overall'] for k in keys])
        years=sum(report['folds'][k]['diagnostics']['test']['overall']['recall']>old['folds'][k]['diagnostics']['test']['overall']['recall'] for k in keys)
        report['pooled'][task]=dict(baseline=b,new=new,years_recall_improved=years,
            meaningful_gate=bool(new['recall']>=b['recall']+.05 and new['false_positive_rate']<=min(budget,b['false_positive_rate']) and years>=2))
    assert all(sha(Path(p))==h for p,h in hashes.items())
    report['artifact_sha256']={p.name:sha(p) for p in output.iterdir()}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps(report['pooled'],indent=2));print(str(output))


if __name__=='__main__':
    main()
