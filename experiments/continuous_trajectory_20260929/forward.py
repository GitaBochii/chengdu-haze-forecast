"""Purged historical expanding-window diagnosis; no final-year tuning."""
from datetime import datetime, timezone
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from trajectory import windows, features, SOURCE, SOURCE_SHA, sha, save_booster, load_booster
from probability import binary_metrics
from constrained import choose
from event_audit import transition_times

HERE=Path(__file__).resolve().parent


def fold_parts(ts, cutoff):
    ts=pd.DatetimeIndex(ts)
    lo=(ts-pd.Timedelta(hours=23)).year
    hi=(ts+pd.Timedelta(hours=24)).year
    return np.where(hi<=cutoff,'train',np.where((lo==cutoff+1)&(hi==cutoff+1),'dev',
                    np.where((lo==cutoff+2)&(hi==cutoff+2),'test','excluded')))


def groups(ts,aqi,stations):
    months=pd.DatetimeIndex(ts).month
    return dict(season=np.array(['DJF','MAM','JJA','SON'])[(months%12)//3],
                intensity=np.select([aqi<=50,aqi<=100,aqi<=150],['<=50','50-100','100-150'],default='>150'),
                stations=np.select([~np.isfinite(stations),stations<=6,stations<=8],['missing','<=6','7-8'],default='>=9'))


def calibration(y,p):
    result=[]
    for i in range(5):
        m=(p>=i/5)&((p<(i+1)/5) if i<4 else (p<=1))
        result.append(dict(lower=i/5,upper=(i+1)/5,n=int(m.sum()),
                      mean_probability=float(p[m].mean()) if m.any() else None,
                      observed_rate=float(y[m].mean()) if m.any() else None))
    return result


def diagnostic(meta,p,threshold,prevalence):
    y=meta.label.to_numpy(bool)
    result=dict(overall=binary_metrics(y,p,threshold),calibration=calibration(y,p),
                constant_train_brier=float(np.mean((prevalence-y)**2)),groups={})
    for name,levels in [('season',['DJF','MAM','JJA','SON']),('intensity',['<=50','50-100','100-150','>150']),('stations',['<=6','7-8','>=9','missing'])]:
        result['groups'][name]={}
        for level in levels:
            m=meta[name].eq(level).to_numpy()
            result['groups'][name][level]=binary_metrics(y[m],p[m],threshold) if m.any() else None
    return result


def main():
    assert sha(SOURCE)==SOURCE_SHA
    deps=[SOURCE,*[HERE/n for n in ['forward.py','FORWARD_PROTOCOL.md','trajectory.py','probability.py','constrained.py','event_audit.py']]]
    hashes={str(p):sha(p) for p in deps}
    raw=pd.read_parquet(SOURCE)
    w=windows(raw)
    x=features(w['history'],w['timestamps'])
    on,rec=transition_times(w['target'][:,:,2])
    group=groups(w['timestamps'],w['history'][:,-1,2],raw.aqi_station_count.to_numpy()[w['indices']])
    output=HERE/'forward_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    report=dict(status='running',source_sha256=hashes,folds={},deployed=False)
    for cutoff in (2019,2020,2021):
        parts=fold_parts(w['timestamps'],cutoff)
        for task,eligible,label,budget in [('onset',~w['polluted'],on<25,.1),('recovery',w['polluted'],rec<20,.05)]:
            key=f'{cutoff+2}_{task}'
            train=eligible&(parts=='train')
            ev=eligible&np.isin(parts,['dev','test'])
            meta=pd.DataFrame(dict(source_index=w['indices'][ev],timestamp=w['timestamps'][ev],partition=parts[ev],label=label[ev],
                origin_aqi=w['history'][ev,-1,2],**{k:v[ev] for k,v in group.items()}))
            meta.to_parquet(output/f'{key}_origins.parquet',index=False)
            probabilities={}
            for name,weight in [('unweighted',None),('balanced','balanced')]:
                model=lgb.LGBMClassifier(objective='binary',class_weight=weight,n_estimators=300,learning_rate=.04,
                    num_leaves=24,min_child_samples=30,random_state=42,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
                model.fit(x[train],label[train])
                p=model.predict_proba(x[ev])[:,1]
                path=output/f'{key}_{name}.txt'
                save_booster(model.booster_,path)
                np.testing.assert_array_equal(p,load_booster(path).predict(x[ev],num_threads=4))
                np.save(output/f'{key}_{name}.npy',p,allow_pickle=False)
                probabilities[name]=p
            dev=meta.partition.eq('dev').to_numpy()
            best,grid=choose(meta.label.to_numpy()[dev],{n:p[dev] for n,p in probabilities.items()},budget)
            prevalence=float(label[train].mean())
            record=dict(train_end_year=cutoff,train_n=int(train.sum()),train_positive=int(label[train].sum()),
                        budget=budget,chosen=best,development_grid=grid,diagnostics={})
            for part in ('dev','test'):
                m=meta.partition.eq(part).to_numpy()
                record['diagnostics'][part]=diagnostic(meta.loc[m],probabilities[best['candidate']][m],best['threshold'],prevalence)
            report['folds'][key]=record
            print(json.dumps(dict(fold=key,chosen=best['candidate'],threshold=best['threshold'],test=record['diagnostics']['test']['overall'])),flush=True)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    report['artifact_sha256']={p.name:sha(p) for p in output.iterdir()}
    report['status']='completed'
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n','utf-8')
    print(str(output),flush=True)


if __name__=='__main__':
    main()
