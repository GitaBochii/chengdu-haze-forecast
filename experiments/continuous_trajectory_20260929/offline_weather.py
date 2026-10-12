"""Offline ERA5 reference, explicitly not verified issue-time weather."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from trajectory import windows,SOURCE,SOURCE_SHA,sha,save_booster,load_booster
from feature_ablation import build_features,summaries,pooled,BASE
from forward import fold_parts,diagnostic
from event_audit import transition_times
from constrained import choose

HERE=Path(__file__).resolve().parent
WEATHER=SOURCE.with_name('weather_proxy.parquet')


def weather_features(raw,indices):
    a=raw[['temperature_c','relative_humidity_pct','wind_u_ms','wind_v_ms']].to_numpy(float)
    a[~np.isfinite(a)]=np.nan
    a=np.column_stack([a,np.hypot(a[:,2],a[:,3])])
    h=a[indices[:,None]+np.arange(-23,1)]
    return np.concatenate([h.reshape(len(h),-1),summaries(h)],axis=1).astype('float32')


def main():
    assert sha(SOURCE)==SOURCE_SHA
    deps=[SOURCE,WEATHER,BASE/'report.json',*[HERE/n for n in ('offline_weather.py','OFFLINE_WEATHER_PROTOCOL.md','feature_ablation.py','forward.py','trajectory.py','constrained.py','probability.py','event_audit.py')]]
    hashes={str(p):sha(p) for p in deps}
    old=json.loads((BASE/'report.json').read_text('utf-8'))
    for n,h in old['artifact_sha256'].items():
        assert sha(BASE/n)==h
    raw=pd.read_parquet(SOURCE)
    weather=pd.read_parquet(WEATHER)
    pd.testing.assert_series_equal(raw.timestamp,weather.timestamp)
    w=windows(raw)
    x=np.concatenate([build_features(raw,w['indices'],w['history'],w['timestamps'])['chemistry'],weather_features(weather,w['indices'])],axis=1)
    on,rec=transition_times(w['target'][:,:,2])
    output=HERE/'offline_weather_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    out=dict(status='completed',scope='OFFLINE ERA5 REFERENCE ONLY; not verified issue-time weather',source_sha256=hashes,feature_dimension=x.shape[1],folds={},pooled={},deployed=False)
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
            for name,weight in [('unweighted',None),('balanced','balanced')]:
                model=lgb.LGBMClassifier(objective='binary',class_weight=weight,n_estimators=300,learning_rate=.04,
                    num_leaves=24,min_child_samples=30,random_state=42,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
                model.fit(x[train],label[train])
                p=model.predict_proba(x[ev])[:,1]
                path=output/f'{key}_{name}.txt'
                save_booster(model.booster_,path)
                np.testing.assert_array_equal(p,load_booster(path).predict(x[ev],num_threads=4))
                np.save(output/f'{key}_{name}.npy',p,allow_pickle=False)
                candidates[name]=p
            dev=meta.partition.eq('dev').to_numpy()
            best,grid=choose(label[ev][dev],{n:p[dev] for n,p in candidates.items()},budget)
            record=dict(train_n=int(train.sum()),train_positive=int(label[train].sum()),budget=budget,chosen=best,development_grid=grid,diagnostics={})
            for part in ('dev','test'):
                m=meta.partition.eq(part).to_numpy()
                record['diagnostics'][part]=diagnostic(meta.loc[m],candidates[best['candidate']][m],best['threshold'],float(label[train].mean()))
            out['folds'][key]=record
            print(json.dumps(dict(fold=key,chosen=best['candidate'],threshold=best['threshold'],test=record['diagnostics']['test']['overall'])),flush=True)
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
