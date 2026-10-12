"""History-only feature ablation on the exact frozen forward populations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from trajectory import windows, features, SOURCE, SOURCE_SHA, sha, save_booster, load_booster
from forward import fold_parts, diagnostic
from constrained import choose
from event_audit import transition_times

HERE=Path(__file__).resolve().parent
BASE=HERE/'forward_runs/20260929T100914338806Z'


def summaries(history):
    h=np.asarray(history,float)
    output=[h[:,-1,:]-h[:,-1-k,:] for k in (1,3,6,12,23)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for width in (3,6,12,24):
            a=h[:,-width:,:]
            avg=np.nanmean(a,axis=1)
            output.extend([avg,np.nanstd(a,axis=1),np.nanmax(a,axis=1)-np.nanmin(a,axis=1),h[:,-1,:]-avg])
    return np.concatenate(output,axis=1).astype('float32')


def build_features(raw,indices,history,timestamps):
    baseline=features(history,timestamps)
    trend=np.concatenate([baseline,summaries(history)],axis=1)
    extra=raw[['so2','no2','co','o3_8h']].to_numpy(float)
    extra[~np.isfinite(extra)|(extra<0)]=np.nan
    h=extra[indices[:,None]+np.arange(-23,1)]
    chemistry=np.concatenate([trend,h.reshape(len(h),-1),summaries(h),np.isfinite(h).mean(axis=1)],axis=1).astype('float32')
    return dict(baseline=baseline,trend=trend,chemistry=chemistry)


def pooled(records):
    tp,fn,fp,tn=[sum(r[k] for r in records) for k in ('tp','fn','fp','tn')]
    return dict(tp=tp,fn=fn,fp=fp,tn=tn,recall=tp/(tp+fn),false_positive_rate=fp/(fp+tn),
                precision=tp/(tp+fp) if tp+fp else None,f1=2*tp/(2*tp+fp+fn))


def main(seed=42):
    assert sha(SOURCE)==SOURCE_SHA
    dependencies=[SOURCE,BASE/'report.json',*[HERE/n for n in ['FEATURE_PROTOCOL.md','feature_ablation.py','forward.py','trajectory.py','probability.py','constrained.py','event_audit.py']]]
    identities={str(p):sha(p) for p in dependencies}
    old=json.loads((BASE/'report.json').read_text('utf-8'))
    for n,h in old['artifact_sha256'].items():
        assert sha(BASE/n)==h
    raw=pd.read_parquet(SOURCE)
    w=windows(raw)
    arms=build_features(raw,w['indices'],w['history'],w['timestamps'])
    on,rec=transition_times(w['target'][:,:,2])
    output=HERE/'feature_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    report=dict(seed=seed,status='running',source_sha256=identities,feature_dimensions={n:x.shape[1] for n,x in arms.items()},folds={},deployed=False)
    for cutoff in (2019,2020,2021):
        parts=fold_parts(w['timestamps'],cutoff)
        for task,eligible,label,budget in [('onset',~w['polluted'],on<25,.1),('recovery',w['polluted'],rec<20,.05)]:
            key=f'{cutoff+2}_{task}'
            train=eligible&(parts=='train')
            ev=eligible&np.isin(parts,['dev','test'])
            meta=pd.read_parquet(BASE/f'{key}_origins.parquet')
            assert np.array_equal(meta.source_index,w['indices'][ev])
            assert np.array_equal(meta.label,label[ev])
            meta.to_parquet(output/f'{key}_origins.parquet',index=False)
            dev=meta.partition.eq('dev').to_numpy()
            record=dict(budget=budget,train_n=int(train.sum()),arms={})
            candidates={}
            for arm,x in arms.items():
                probabilities={}
                for name,weight in [('unweighted',None),('balanced','balanced')]:
                    if arm=='baseline' and seed==42:
                        p=np.load(BASE/f'{key}_{name}.npy')
                    else:
                        model=lgb.LGBMClassifier(objective='binary',class_weight=weight,n_estimators=300,learning_rate=.04,
                            num_leaves=24,min_child_samples=30,random_state=seed,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
                        model.fit(x[train],label[train])
                        p=model.predict_proba(x[ev])[:,1]
                        path=output/f'{key}_{arm}_{name}.txt'
                        save_booster(model.booster_,path)
                        np.testing.assert_array_equal(p,load_booster(path).predict(x[ev],num_threads=4))
                    np.save(output/f'{key}_{arm}_{name}.npy',p,allow_pickle=False)
                    probabilities[name]=p
                    candidates[f'{arm}_{name}']=p
                best,grid=choose(label[ev][dev],{n:p[dev] for n,p in probabilities.items()},budget)
                arm_record=dict(chosen=best,development_grid=grid,diagnostics={})
                for part in ('dev','test'):
                    m=meta.partition.eq(part).to_numpy()
                    arm_record['diagnostics'][part]=diagnostic(meta.loc[m],probabilities[best['candidate']][m],best['threshold'],float(label[train].mean()))
                record['arms'][arm]=arm_record
            best,grid=choose(label[ev][dev],{n:p[dev] for n,p in candidates.items()},budget)
            m=meta.partition.eq('test').to_numpy()
            record['selected']=dict(chosen=best,development_grid=grid,
                diagnostics=diagnostic(meta.loc[m],candidates[best['candidate']][m],best['threshold'],float(label[train].mean())))
            report['folds'][key]=record
            print(json.dumps(dict(fold=key,arms={n:r['diagnostics']['test']['overall'] for n,r in record['arms'].items()},selected=best['candidate'])),flush=True)
    report['pooled']={}
    for task,budget in [('onset',.1),('recovery',.05)]:
        rr=[v for k,v in report['folds'].items() if k.endswith('_'+task)]
        totals={n:pooled([r['arms'][n]['diagnostics']['test']['overall'] for r in rr]) for n in arms}
        totals['selected']=pooled([r['selected']['diagnostics']['overall'] for r in rr])
        for n,v in totals.items():
            years=sum((r['selected']['diagnostics']['overall'] if n=='selected' else r['arms'][n]['diagnostics']['test']['overall'])['recall']>r['arms']['baseline']['diagnostics']['test']['overall']['recall'] for r in rr)
            v['years_recall_improved']=years
            v['meaningful_gate']=bool(v['recall']>=totals['baseline']['recall']+.05 and v['false_positive_rate']<=min(budget,totals['baseline']['false_positive_rate']) and years>=2)
        report['pooled'][task]=totals
    assert all(sha(Path(p))==h for p,h in identities.items())
    report['artifact_sha256']={p.name:sha(p) for p in output.iterdir()}
    report['status']='completed'
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps(report['pooled'],indent=2),flush=True)
    print(str(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=42)
    main(parser.parse_args().seed)
