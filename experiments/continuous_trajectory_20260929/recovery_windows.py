"""Separate all-condition six-hour recovery classifiers; dev-only decisions."""
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
from delayed_chemistry import build
from full_curve_recovery import recovery_start, recovery_metrics

HERE = Path(__file__).resolve().parent
BASE = HERE / 'full_curve_runs/20260930T033524196780Z'
BUFFER = HERE / 'recovery_calibration_runs/20260930T121425927928Z'
THRESHOLDS = [i / 20 for i in range(1,21)] + [1.000001]


def window_labels(target):
    a = np.asarray(target)
    if a.ndim != 2 or a.shape[1] != 24 or not np.isfinite(a).all():
        raise ValueError('Complete finite 24-hour observations required')
    return np.lib.stride_tricks.sliding_window_view(a <= 100,6,axis=1).all(axis=2)


def decode(probability, threshold):
    p = np.asarray(probability)
    if (p.ndim != 2 or p.shape[1] != 19 or not np.isfinite(p).all()
            or ((p < 0) | (p > 1)).any() or threshold not in THRESHOLDS):
        raise ValueError('Valid nineteen-window probabilities and frozen threshold required')
    yes = p >= threshold
    return np.where(yes.any(axis=1),yes.argmax(axis=1)+1,20)


def start_metrics(truth, prediction):
    t,p = np.asarray(truth),np.asarray(prediction)
    yes,hat = t<20,p<20
    tp,fn,fp,tn = [int(v.sum()) for v in (yes&hat,yes&~hat,~yes&hat,~yes&~hat)]
    early = int(((p<t)&hat).sum())
    ratio = lambda a,b: float(a/b) if b else None
    return dict(origins=len(t), actual_right_censored=int((~yes).sum()),
        predicted_no_recovery=int((~hat).sum()),
        capped_min_T_20_mae_hours=float(np.abs(p-t).mean()) if len(t) else None,
        confusion=dict(tp=tp,fn=fn,fp=fp,tn=tn), precision=ratio(tp,tp+fp),
        recall=ratio(tp,tp+fn), fpr=ratio(fp,fp+tn), premature_start_count=early,
        premature_start_fraction_all=ratio(early,len(t)),
        premature_start_fraction_predicted_recovery=ratio(early,int(hat.sum())))


def choose(target, probability, raw_early):
    truth = recovery_start(target)
    if not len(truth) or not (truth==20).any():
        raise ValueError('Development requires actual nonrecoveries')
    grid = []
    for threshold in THRESHOLDS:
        m = start_metrics(truth,decode(probability,threshold))
        grid.append(dict(threshold=threshold, metrics=m,
            feasible=m['fpr']<=.05 and m['premature_start_fraction_all']<=raw_early))
    valid = [r for r in grid if r['feasible']]
    best = min(valid,key=lambda r:(r['metrics']['capped_min_T_20_mae_hours'],
        -(r['metrics']['recall'] or 0), r['metrics']['fpr'],-r['threshold']))
    return best['threshold'], grid


def main():
    started = time.perf_counter()
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Frozen source changed')
    ref = json.loads((BASE/'report.json').read_text('utf-8'))
    if json.loads((BASE/'independent_audit.json').read_text('utf-8'))['status'] != 'passed':
        raise ValueError('Reference must be independently verified')
    selected = json.loads((BUFFER/'selected.json').read_text('utf-8'))
    inputs = {str(SOURCE):SOURCE_SHA}
    for n in ('trajectory.py','forward.py','delayed_chemistry.py','issue_delay_features.py',
              'feature_ablation.py','full_curve_recovery.py'):
        inputs[str(HERE/n)] = sha(HERE/n)
    for base in (BASE,BUFFER):
        for n in ('report.json','independent_audit.json'):
            inputs[str(base/n)] = sha(base/n)
    inputs[str(BUFFER/'selected.json')] = sha(BUFFER/'selected.json')
    for n,h in ref['artifact_sha256'].items():
        if n.endswith(('.npy','.parquet')):
            inputs[str(BASE/n)] = h
    for p,h in inputs.items():
        if sha(Path(p)) != h:
            raise ValueError(f'Frozen dependency changed: {p}')
    raw = pd.read_parquet(SOURCE)
    w = windows(raw)
    y = w['target'][:,:,2]
    labels = window_labels(y)
    output = HERE/'recovery_window_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    report = dict(status='running',source_run=str(BASE),buffer_run=str(BUFFER),
        input_sha256=inputs,code_sha256=sha(Path(__file__)),
        protocol_sha256=sha(HERE/'RECOVERY_WINDOWS_PROTOCOL.md'),
        lag=1,feature_dimension=323,thresholds=THRESHOLDS,folds={},models={},
        pooled_test={},historical_stress_test=True,blind_test=False,prospective=False,
        deployed=False,concentration_predictions_changed=False)
    pooled = {k:[] for k in ('truth','direct','raw','buffered')}
    def save():
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),'utf-8')
    save()
    print(output,flush=True)
    try:
        for cutoff in (2019,2020,2021):
            year = str(cutoff+2)
            parts = fold_parts(w['timestamps'],cutoff)
            train,ev = parts=='train',np.isin(parts,['dev','test'])
            meta = pd.DataFrame(dict(source_index=w['indices'][ev],timestamp=w['timestamps'][ev],
                partition=parts[ev],origin_aqi=w['history'][ev,-1,2]))
            pd.testing.assert_frame_equal(meta,pd.read_parquet(BASE/f'{year}_origins.parquet'))
            target = y[ev]
            np.testing.assert_array_equal(target,np.load(BASE/f'{year}_targets.npy'))
            x = build(raw,w,cutoff,1)
            if x.shape != (len(y),323):
                raise ValueError('Feature dimension changed')
            probability = np.empty((int(ev.sum()),19),dtype=float)
            for j in range(19):
                model = lgb.LGBMClassifier(objective='binary',n_estimators=300,learning_rate=.04,
                    num_leaves=24,min_child_samples=30,random_state=42,n_jobs=4,
                    verbosity=-1,deterministic=True,force_col_wise=True)
                model.fit(x[train],labels[train,j])
                p = model.predict_proba(x[ev])[:,1]
                path = output/f'{year}_window{j+1}.txt'
                save_booster(model.booster_,path)
                np.testing.assert_array_equal(p,load_booster(path).predict(x[ev],num_threads=4))
                probability[:,j] = p
                report['models'][path.name] = sha(path)
                print(json.dumps(dict(year=year,window=j+1)),flush=True)
            meta.to_parquet(output/f'{year}_origins.parquet',index=False)
            np.save(output/f'{year}_targets.npy',target,allow_pickle=False)
            np.save(output/f'{year}_probabilities.npy',probability,allow_pickle=False)
            control = np.load(BASE/f'{year}_lag1_chemistry.npy')
            buffered = control.astype(float)+selected[year]['lag1_chemistry']
            polluted = meta.origin_aqi.to_numpy()>100
            dev = (meta.partition.to_numpy()=='dev')&polluted
            test = (meta.partition.to_numpy()=='test')&polluted
            raw_dev = recovery_metrics(target[dev],control[dev])
            threshold,grid = choose(target[dev],probability[dev],raw_dev['premature_start_fraction_all'])
            truth,direct = recovery_start(target),decode(probability,threshold)
            report['folds'][year] = dict(train_end_year=cutoff,train_n=int(train.sum()),
                train_clean=int((train&~w['polluted']).sum()),train_polluted=int((train&w['polluted']).sum()),
                threshold=threshold,development_grid=grid,
                dev=start_metrics(truth[dev],direct[dev]),test=start_metrics(truth[test],direct[test]),
                raw_test=recovery_metrics(target[test],control[test]),
                buffered_test=recovery_metrics(target[test],buffered[test]))
            for name,values in dict(truth=truth,direct=direct,raw=recovery_start(control),
                                     buffered=recovery_start(buffered)).items():
                pooled[name].append(values[test])
            save()
            del x
        a = {k:np.concatenate(v) for k,v in pooled.items()}
        report['pooled_test'] = {k:start_metrics(a['truth'],a[k]) for k in ('direct','raw','buffered')}
        if any(sha(Path(p)) != h for p,h in inputs.items()):
            raise ValueError('Frozen input changed during experiment')
        report.update(status='completed_fixed_protocol',elapsed_seconds=time.perf_counter()-started)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        report['artifact_sha256'] = {p.name:sha(p) for p in output.iterdir() if p.suffix in ('.txt','.npy','.parquet')}
        save()
    print(json.dumps(report['pooled_test'],indent=2),flush=True)


if __name__ == '__main__':
    main()
