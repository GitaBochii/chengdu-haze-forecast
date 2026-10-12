"""Frozen lag-one hourly trajectories, with censored AQI recovery diagnostics."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd

from trajectory import SOURCE, SOURCE_SHA, windows, sha, save_booster, load_booster
from forward import fold_parts
from issue_delay_features import build as trend_build
from delayed_chemistry import build as chemistry_build

HERE=Path(__file__).resolve().parent
BASE=HERE/'issue_delay_runs/20260929T133825296179Z'
CHEM=HERE/'delayed_chemistry_runs/20260929T134650568191Z'
ARMS={'lag1_trend':(BASE,139),'lag1_chemistry':(CHEM,323)}


def recovery_start(curves):
    a=np.asarray(curves)
    if a.ndim!=2 or a.shape[1]!=24 or not np.isfinite(a).all():
        raise ValueError('Recovery requires complete finite 24-hour curves')
    six=np.lib.stride_tricks.sliding_window_view(a<=100,6,axis=1).all(axis=2)
    return np.where(six.any(axis=1),six.argmax(axis=1)+1,20)


def recovery_metrics(target,prediction):
    t,p=recovery_start(target),recovery_start(prediction)
    yes,hat=t<20,p<20
    tp,fn,fp,tn=[int(v.sum()) for v in (yes&hat,yes&~hat,~yes&hat,~yes&~hat)]
    n=len(t)
    early=(p<t)&hat
    ratio=lambda a,b:float(a/b) if b else None
    return dict(origins=n,actual_right_censored=int((~yes).sum()),
        predicted_no_recovery=int((~hat).sum()),
        capped_min_T_20_mae_hours=float(np.abs(p-t).mean()) if n else None,
        confusion=dict(tp=tp,fn=fn,fp=fp,tn=tn),precision=ratio(tp,tp+fp),
        recall=ratio(tp,tp+fn),fpr=ratio(fp,fp+tn),
        premature_start_count=int(early.sum()),premature_start_fraction_all=ratio(early.sum(),n),
        premature_start_fraction_predicted_recovery=ratio(early.sum(),hat.sum()))


def diagnostics(target,prediction,origin):
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions')
    out={}
    for name,mask in [('all',np.ones(len(origin),bool)),('clean_origin',origin<=100),
                      ('polluted_origin',origin>100)]:
        e=prediction[mask].astype(float)-target[mask].astype(float)
        out[name]=dict(origins=int(mask.sum()),mae=float(np.abs(e).mean()) if len(e) else None,
            lead_mae=np.abs(e).mean(axis=0).tolist() if len(e) else None)
    out['aqi_recovery']=recovery_metrics(target[origin>100],prediction[origin>100])
    return out


def main():
    start=time.perf_counter()
    if sha(SOURCE)!=SOURCE_SHA:
        raise ValueError('Source mismatch')
    hashes={}
    old={}
    for base in (BASE,CHEM):
        r=json.loads((base/'report.json').read_text('utf-8'))
        if r['status']!='completed_fixed_protocol':
            raise ValueError('Reference incomplete')
        checked={**r['source_sha256'],**{str(base/n):h for n,h in r['artifact_sha256'].items()}}
        for p,h in checked.items():
            if sha(Path(p))!=h:
                raise ValueError(f'Reference mismatch: {p}')
        hashes.update(checked)
        hashes[str(base/'report.json')]=sha(base/'report.json')
        old[base]=r
    for name in ('full_curve_recovery.py','FULL_CURVE_PROTOCOL.md','test_full_curve_recovery.py'):
        hashes[str(HERE/name)]=sha(HERE/name)
    raw=pd.read_parquet(SOURCE)
    w=windows(raw)
    y=w['target'][:,:,2]
    prepared={}
    for cutoff in (2019,2020,2021):
        key=str(cutoff+2)
        parts=fold_parts(w['timestamps'],cutoff)
        train,ev=parts=='train',np.isin(parts,['dev','test'])
        meta=pd.DataFrame(dict(source_index=w['indices'][ev],timestamp=w['timestamps'][ev],
            partition=parts[ev],origin_aqi=w['history'][ev,-1,2]))
        for base in (BASE,CHEM):
            pd.testing.assert_frame_equal(meta,pd.read_parquet(base/f'{key}_origins.parquet'))
            np.testing.assert_array_equal(y[ev][:,[0,5,23]],np.load(base/f'{key}_targets.npy'))
            if int(train.sum())!=old[base]['folds'][key]['train_n']:
                raise ValueError('Training population mismatch')
        prepared[key]=(cutoff,train,ev,meta)
    output=HERE/'full_curve_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',source_sha256=hashes,folds={},pooled_test={},models={},
        reused_models={},leads=list(range(1,25)),lag=1,feature_dimensions={n:d for n,(_,d) in ARMS.items()},
        target='available-station-median AQI',recovery='first six-hour AQI<=100 start in 1..19; otherwise T>19',
        historical_stress_test=True,blind_test=False,prospective=False,deployed=False,
        eligibility=w['eligibility'],versions=dict(lightgbm=lgb.__version__,numpy=np.__version__,pandas=pd.__version__))
    pooled={n:[] for n in ARMS}
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
            report['folds'][key]=dict(train_end_year=cutoff,train_n=int(train.sum()),
                train_clean=int((train&~w['polluted']).sum()),train_polluted=int((train&w['polluted']).sum()),arms={})
            for name,(base,dim) in ARMS.items():
                x=trend_build(raw,w,cutoff,1,True) if name=='lag1_trend' else chemistry_build(raw,w,cutoff,1)
                if x.shape!=(len(y),dim) or x.dtype!=np.float32 or np.isinf(x).any():
                    raise ValueError('Invalid features')
                pred=np.empty_like(target)
                control=np.load(base/f'{key}_{name}.npy')
                for j in range(24):
                    lead=j+1
                    path=output/f'{key}_{name}_h{lead}.txt'
                    if lead in (1,6,24):
                        original=base/path.name
                        shutil.copyfile(original,path)
                        p=load_booster(path).predict(x[ev],num_threads=4).clip(min=0).astype('float32')
                        np.testing.assert_array_equal(p,control[:,(1,6,24).index(lead)])
                        report['reused_models'][path.name]=dict(path=str(original),sha256=sha(original))
                    else:
                        model=lgb.LGBMRegressor(objective='regression_l1',n_estimators=300,
                            learning_rate=.04,num_leaves=24,min_child_samples=30,random_state=42,
                            n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
                        model.fit(x[train],y[train,j])
                        p=model.predict(x[ev]).clip(min=0).astype('float32')
                        save_booster(model.booster_,path)
                        model.booster_.free_dataset()
                        del model
                    restored=load_booster(path).predict(x[ev],num_threads=4).clip(min=0).astype('float32')
                    np.testing.assert_array_equal(p,restored)
                    pred[:,j]=p
                    report['models'][path.name]=sha(path)
                    print(json.dumps(dict(fold=key,arm=name,lead=lead)),flush=True)
                np.save(output/f'{key}_{name}.npy',pred,allow_pickle=False)
                record={}
                for partition in ('dev','test'):
                    m=meta.partition.eq(partition).to_numpy()
                    origin=meta.origin_aqi.to_numpy()[m]
                    record[partition]=diagnostics(target[m],pred[m],origin)
                    if partition=='test':
                        pooled[name].append((target[m],pred[m],origin))
                report['folds'][key]['arms'][name]=record
                save()
                del x
        for name,items in pooled.items():
            yy,pp,oo=[np.concatenate([v[j] for v in items]) for j in range(3)]
            report['pooled_test'][name]=diagnostics(yy,pp,oo)
        if any(sha(Path(p))!=h for p,h in hashes.items()):
            raise ValueError('Frozen dependency changed')
        report.update(status='completed_fixed_protocol',elapsed_seconds=time.perf_counter()-start)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start)
        raise
    finally:
        report['artifact_sha256']={p.name:sha(p) for p in output.iterdir() if p.suffix in ('.npy','.txt','.parquet')}
        save()
    print(json.dumps(dict(run_dir=str(output),pooled=report['pooled_test'])),flush=True)


if __name__=='__main__':
    main()
