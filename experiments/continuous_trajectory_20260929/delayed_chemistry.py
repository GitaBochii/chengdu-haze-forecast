"""Fixed delayed additional-pollutant ablation, with frozen paired controls."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd

from trajectory import SOURCE, SOURCE_SHA, windows, sha, save_booster, load_booster
from forward import fold_parts
from feature_ablation import summaries
from issue_delay_features import build as baseline_features
from issue_delay import LEADS, diagnostics

HERE=Path(__file__).resolve().parent
BASE=HERE/'issue_delay_runs/20260929T133825296179Z'
CHANNELS=('so2','no2','co','o3_8h')


def build(raw,w,cutoff,lag):
    base=baseline_features(raw,w,cutoff,lag,True)
    ts=pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    idx=np.asarray(w['indices'],int)
    pos=idx[:,None]+np.arange(-23,1)-lag
    safe=pos.clip(0,len(ts)-1)
    years=ts.year.to_numpy()
    valid=(pos>=0)&(pos<len(ts))
    valid &= np.where(years[idx,None]<=cutoff,years[safe]<=cutoff,
                      years[safe]==years[idx,None])
    values=raw[list(CHANNELS)].to_numpy(float)
    values[~np.isfinite(values)|(values<0)]=np.nan
    history=values[safe].copy()
    history[~valid]=np.nan
    return np.column_stack([base,history.reshape(len(idx),96),summaries(history),
                            np.isfinite(history).mean(axis=1)]).astype('float32')


def compare(target,prediction,control,origin):
    result={}
    for group,mask in [('all',np.ones(len(target),bool)),('clean_origin',origin<=100),
                       ('polluted_origin',origin>100)]:
        candidate=np.abs(prediction[mask].astype(float)-target[mask])
        reference=np.abs(control[mask].astype(float)-target[mask])
        delta=candidate-reference
        result[group]=dict(origins=int(mask.sum()),mae_difference=float(delta.mean()),
            relative_mae_reduction=float(-delta.mean()/reference.mean()),
            leads={f'h{lead}':dict(mae_difference=float(delta[:,j].mean()),
                relative_mae_reduction=float(-delta[:,j].mean()/reference[:,j].mean()))
                   for j,lead in enumerate(LEADS)})
    return result


def main():
    started=time.perf_counter()
    old=json.loads((BASE/'report.json').read_text('utf-8'))
    if old['status']!='completed_fixed_protocol' or sha(SOURCE)!=SOURCE_SHA:
        raise ValueError('Baseline or source invalid')
    checked={**old['source_sha256'],**{str(BASE/n):h for n,h in old['artifact_sha256'].items()}}
    for path,digest in checked.items():
        if sha(Path(path))!=digest:
            raise ValueError(f'Frozen baseline hash mismatch: {path}')
    dependencies=[BASE/'report.json',*[HERE/name for name in ('delayed_chemistry.py',
        'DELAYED_CHEMISTRY_PROTOCOL.md','test_delayed_chemistry.py')]]
    hashes={**checked,**{str(path):sha(path) for path in dependencies}}
    raw=pd.read_parquet(SOURCE)
    w=windows(raw)
    y=w['target'][:,np.array(LEADS)-1,2]
    prepared={}
    # Validate all folds before any model is fitted.
    for cutoff in (2019,2020,2021):
        key=str(cutoff+2)
        parts=fold_parts(w['timestamps'],cutoff)
        train,ev=parts=='train',np.isin(parts,['dev','test'])
        meta=pd.DataFrame(dict(source_index=w['indices'][ev],timestamp=w['timestamps'][ev],
            partition=parts[ev],origin_aqi=w['history'][ev,-1,2]))
        pd.testing.assert_frame_equal(meta,pd.read_parquet(BASE/f'{key}_origins.parquet'))
        np.testing.assert_array_equal(y[ev],np.load(BASE/f'{key}_targets.npy'))
        if int(train.sum())!=old['folds'][key]['train_n']:
            raise ValueError('Training population changed')
        prepared[key]=(cutoff,train,ev,meta)
    output=HERE/'delayed_chemistry_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',source_sha256=hashes,folds={},pooled_test={},paired_comparisons={},
        models={},feature_dimensions=323,channels=list(CHANNELS),leads=list(LEADS),units='AQI',
        target='available-station-median AQI',historical_stress_test=True,blind_test=False,
        prospective=False,deployed=False,baseline_run=str(BASE),eligibility=w['eligibility'],
        versions=dict(lightgbm=lgb.__version__,numpy=np.__version__,pandas=pd.__version__))
    pooled={}
    def save():
        tmp=output/'report.tmp'
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        tmp.replace(output/'report.json')
    save()
    print(json.dumps(dict(run_dir=str(output))),flush=True)
    try:
        for key,(cutoff,train,ev,meta) in prepared.items():
            target=y[ev]
            meta.to_parquet(output/f'{key}_origins.parquet',index=False)
            np.save(output/f'{key}_targets.npy',target,allow_pickle=False)
            report['folds'][key]=dict(train_end_year=cutoff,train_n=int(train.sum()),arms={})
            for lag in (1,3):
                name=f'lag{lag}_chemistry'
                x=build(raw,w,cutoff,lag)
                if x.shape!=(len(y),323) or x.dtype!=np.float32 or np.isinf(x).any():
                    raise ValueError('Invalid features')
                prediction=np.empty_like(target)
                for j,lead in enumerate(LEADS):
                    model=lgb.LGBMRegressor(objective='regression_l1',n_estimators=300,
                        learning_rate=.04,num_leaves=24,min_child_samples=30,random_state=42,
                        n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
                    model.fit(x[train],y[train,j])
                    p=model.predict(x[ev]).clip(min=0).astype('float32')
                    path=output/f'{key}_{name}_h{lead}.txt'
                    save_booster(model.booster_,path)
                    restored=load_booster(path).predict(x[ev],num_threads=4).clip(min=0).astype('float32')
                    np.testing.assert_array_equal(p,restored)
                    if not np.isfinite(p).all():
                        raise ValueError('Nonfinite predictions')
                    prediction[:,j]=p
                    report['models'][path.name]=sha(path)
                    model.booster_.free_dataset()
                    del model
                np.save(output/f'{key}_{name}.npy',prediction,allow_pickle=False)
                record={}
                controls={kind:np.load(BASE/f'{key}_lag{lag}_{kind}.npy') for kind in ('base','trend')}
                for partition in ('dev','test'):
                    mask=meta.partition.eq(partition).to_numpy()
                    origin=meta.origin_aqi.to_numpy()[mask]
                    history=x[ev][mask,139:235].reshape(-1,24,4)
                    record[partition]=dict(diagnostics=diagnostics(target[mask],prediction[mask],origin),
                        extra_coverage=dict(zip(CHANNELS,np.isfinite(history).mean(axis=(0,1)).tolist())),
                        comparisons={kind:compare(target[mask],prediction[mask],control[mask],origin)
                                     for kind,control in controls.items()})
                    if partition=='test':
                        pooled.setdefault(name,[]).append((target[mask],prediction[mask],origin,
                            controls['base'][mask],controls['trend'][mask]))
                record['train_extra_coverage']=dict(zip(CHANNELS,np.isfinite(
                    x[train,139:235].reshape(-1,24,4)).mean(axis=(0,1)).tolist()))
                report['folds'][key]['arms'][name]=record
                save()
                print(json.dumps(dict(fold=key,arm=name,test=record['test'])),flush=True)
                del x
        for name,items in pooled.items():
            yy,pp,oo,bb,tt=[np.concatenate([item[j] for item in items]) for j in range(5)]
            report['pooled_test'][name]=diagnostics(yy,pp,oo)
            report['paired_comparisons'][name]={kind:compare(yy,pp,control,oo)
                                                for kind,control in [('base',bb),('trend',tt)]}
        if any(sha(Path(path))!=digest for path,digest in hashes.items()):
            raise ValueError('Frozen dependency changed during run')
        report.update(status='completed_fixed_protocol',elapsed_seconds=time.perf_counter()-started)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),
                      elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        report['artifact_sha256']={p.name:sha(p) for p in output.iterdir()
                                  if p.suffix in ('.npy','.txt','.parquet')}
        save()
    print(json.dumps(dict(run_dir=str(output),status=report['status'],
                         paired_comparisons=report['paired_comparisons'])),flush=True)


if __name__=='__main__':
    main()
