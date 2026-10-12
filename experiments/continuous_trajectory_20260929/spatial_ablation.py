"""Historical station-spread information without changing evaluation populations."""
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from trajectory import windows,SOURCE,SOURCE_SHA,sha,save_booster,load_booster
from forward import fold_parts,diagnostic
from feature_ablation import build_features,pooled,BASE
from constrained import choose
from event_audit import transition_times

HERE=Path(__file__).resolve().parent
STATIONS=SOURCE.with_name('station_hourly.parquet')


def station_statistics(stations,timestamps):
    if stations.duplicated(['timestamp','station_id']).any():
        raise ValueError('Duplicate station-hour')
    arrays=[]
    counts=[]
    fraction=None
    for name in ('aqi','pm2_5','pm10','o3_8h'):
        a=stations.pivot(index='timestamp',columns='station_id',values=name).reindex(timestamps).to_numpy(float)
        a[~np.isfinite(a)|(a<0)]=np.nan
        count=np.isfinite(a).sum(axis=1)
        counts.append(count)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            arrays.extend([np.nanquantile(a,.25,axis=1),np.nanquantile(a,.75,axis=1),np.nanmax(a,axis=1),np.nanstd(a,axis=1)])
        if name=='aqi':
            fraction=np.divide((a>100).sum(axis=1),count,out=np.full(len(a),np.nan),where=count>0)
    return np.stack([*arrays,*counts,fraction],axis=1).astype('float32')


def spatial_features(stats,indices):
    h=stats[indices[:,None]+np.arange(-23,1)]
    result=[h[:,-1-k,:] for k in (0,3,6,12,23)]
    result.extend([h[:,-1,:]-h[:,-1-k,:] for k in (3,6,12)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        result.extend([np.nanmean(h[:,-k:,:],axis=1) for k in (3,6,12,24)])
    return np.concatenate(result,axis=1)


def main():
    assert sha(SOURCE)==SOURCE_SHA
    deps=[SOURCE,STATIONS,BASE/'report.json',*[HERE/n for n in ('spatial_ablation.py','SPATIAL_PROTOCOL.md','feature_ablation.py','trajectory.py','forward.py','constrained.py','probability.py','event_audit.py')]]
    hashes={str(p):sha(p) for p in deps}
    old=json.loads((BASE/'report.json').read_text('utf-8'))
    for n,h in old['artifact_sha256'].items():
        assert sha(BASE/n)==h
    raw=pd.read_parquet(SOURCE)
    stations=pd.read_parquet(STATIONS)
    for name in ('aqi','pm2_5','pm10','o3_8h'):
        med=stations.groupby('timestamp')[name].median().reindex(raw.timestamp).to_numpy()
        np.testing.assert_allclose(med,raw[name].to_numpy(),equal_nan=True)
    w=windows(raw)
    x=np.concatenate([build_features(raw,w['indices'],w['history'],w['timestamps'])['chemistry'],
        spatial_features(station_statistics(stations,raw.timestamp),w['indices'])],axis=1)
    on,rec=transition_times(w['target'][:,:,2])
    output=HERE/'spatial_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    report=dict(status='running',source_sha256=hashes,feature_dimension=x.shape[1],folds={},deployed=False)
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
            record=dict(train_n=int(train.sum()),train_positive=int(label[train].sum()),budget=budget,chosen=best,development_grid=grid,diagnostics={})
            for part in ('dev','test'):
                m=meta.partition.eq(part).to_numpy()
                record['diagnostics'][part]=diagnostic(meta.loc[m],probabilities[best['candidate']][m],best['threshold'],float(label[train].mean()))
            report['folds'][key]=record
            print(json.dumps(dict(fold=key,chosen=best['candidate'],threshold=best['threshold'],test=record['diagnostics']['test']['overall'])),flush=True)
    report['pooled']={}
    for task,budget in [('onset',.1),('recovery',.05)]:
        keys=[k for k in report['folds'] if k.endswith('_'+task)]
        baseline=pooled([old['folds'][k]['diagnostics']['test']['overall'] for k in keys])
        new=pooled([report['folds'][k]['diagnostics']['test']['overall'] for k in keys])
        years=sum(report['folds'][k]['diagnostics']['test']['overall']['recall']>old['folds'][k]['diagnostics']['test']['overall']['recall'] for k in keys)
        report['pooled'][task]=dict(baseline=baseline,spatial=new,years_recall_improved=years,
            meaningful_gate=bool(new['recall']>=baseline['recall']+.05 and new['false_positive_rate']<=min(budget,baseline['false_positive_rate']) and years>=2))
    assert all(sha(Path(p))==h for p,h in hashes.items())
    report['artifact_sha256']={p.name:sha(p) for p in output.iterdir()}
    report['status']='completed'
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps(report['pooled'],indent=2),flush=True)
    print(str(output),flush=True)


if __name__=='__main__':
    main()
