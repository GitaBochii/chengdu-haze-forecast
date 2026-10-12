"""One fixed history-regime weight contrast; all training dates retained."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from trajectory import SOURCE, SOURCE_SHA, windows, sha, save_booster, load_booster
from forward import fold_parts
from delayed_chemistry import build
from full_curve_recovery import recovery_start, recovery_metrics
from recovery_windows import window_labels, decode, choose, start_metrics, THRESHOLDS, BASE, BUFFER

HERE = Path(__file__).resolve().parent
UNIFORM = HERE/'recovery_window_runs/20260930T122334448606Z'


def regime_weights(last_observed_aqi):
    a = np.asarray(last_observed_aqi,dtype=float)
    if a.ndim != 1 or not len(a):
        raise ValueError('Nonempty one-dimensional training history required')
    valid = np.isfinite(a)&(a>=0)
    clean,polluted = valid&(a<=100),valid&(a>100)
    nc,npoll,nv = int(clean.sum()),int(polluted.sum()),int(valid.sum())
    if not nc or not npoll:
        raise ValueError('Both observed training regimes must exist')
    weight = np.ones(len(a),dtype=float)
    weight[clean],weight[polluted] = nv/(2*nc),nv/(2*npoll)
    return weight,dict(clean_count=nc,polluted_count=npoll,finite_count=nv,
        invalid_count=int((~valid).sum()),clean_weight_mass=float(weight[clean].sum()),
        polluted_weight_mass=float(weight[polluted].sum()),mean_weight=float(weight.mean()),
        clean_weight=float(nv/(2*nc)),polluted_weight=float(nv/(2*npoll)))


def qualifying_diagnostics(target, probability, predicted, origin):
    labels = window_labels(target)
    truth = recovery_start(target)
    out = {}
    for name,mask in (('all',np.ones(len(target),bool)),('clean_origin',origin<=100),
                      ('polluted_origin',origin>100)):
        event = truth[mask]<20
        scores = probability[mask].max(axis=1)
        out[name] = dict(origins=int(mask.sum()),
            window_brier=float(np.square(probability[mask]-labels[mask]).mean()),
            any_window_auc=float(roc_auc_score(event,scores)) if len(np.unique(event))==2 else None,
            qualifying_window_metrics=start_metrics(truth[mask],predicted[mask]))
    return out


def main():
    started = time.perf_counter()
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Frozen source changed')
    ref = json.loads((BASE/'report.json').read_text('utf-8'))
    uniform = json.loads((UNIFORM/'report.json').read_text('utf-8'))
    for base in (BASE,BUFFER,UNIFORM):
        audit = json.loads((base/'independent_audit.json').read_text('utf-8'))
        if audit['status'] != 'passed':
            raise ValueError(f'Unverified control: {base}')
        if 'report_sha256' in audit and sha(base/'report.json') != audit['report_sha256']:
            raise ValueError(f'Control report changed: {base}')
    selected = json.loads((BUFFER/'selected.json').read_text('utf-8'))
    inputs = {str(SOURCE):SOURCE_SHA}
    for n in ('trajectory.py','forward.py','delayed_chemistry.py','issue_delay_features.py',
              'feature_ablation.py','full_curve_recovery.py','recovery_windows.py'):
        inputs[str(HERE/n)] = sha(HERE/n)
    for base in (BASE,BUFFER,UNIFORM):
        for n in ('report.json','independent_audit.json'):
            inputs[str(base/n)] = sha(base/n)
    inputs[str(BUFFER/'selected.json')] = sha(BUFFER/'selected.json')
    for base,record in ((BASE,ref),(UNIFORM,uniform)):
        for n,h in record['artifact_sha256'].items():
            if n.endswith(('.npy','.parquet')):
                inputs[str(base/n)] = h
    for p,h in inputs.items():
        if sha(Path(p)) != h:
            raise ValueError(f'Frozen dependency changed: {p}')
    raw = pd.read_parquet(SOURCE)
    w = windows(raw)
    y = w['target'][:,:,2]
    labels = window_labels(y)
    output = HERE/'weighted_recovery_window_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    report = dict(status='running',source_run=str(BASE),buffer_run=str(BUFFER),uniform_run=str(UNIFORM),
        input_sha256=inputs,code_sha256=sha(Path(__file__)),
        protocol_sha256=sha(HERE/'WEIGHTED_RECOVERY_WINDOWS_PROTOCOL.md'),
        lag=1,feature_dimension=323,thresholds=THRESHOLDS,folds={},models={},pooled_test={},
        historical_stress_test=True,blind_test=False,prospective=False,deployed=False,
        concentration_predictions_changed=False)
    pooled = {k:[] for k in ('truth','weighted','uniform','raw','buffered')}
    qual = {k:[] for k in ('target','probability','uniform_probability','predicted','uniform_predicted','origin')}
    def save():
        tmp = output/'report.tmp'
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False),'utf-8')
        tmp.replace(output/'report.json')
    save()
    print(output,flush=True)
    try:
        for cutoff in (2019,2020,2021):
            year = str(cutoff+2)
            parts = fold_parts(w['timestamps'],cutoff)
            train,ev = parts=='train',np.isin(parts,['dev','test'])
            meta = pd.DataFrame(dict(source_index=w['indices'][ev],timestamp=w['timestamps'][ev],
                partition=parts[ev],origin_aqi=w['history'][ev,-1,2]))
            pd.testing.assert_frame_equal(meta,pd.read_parquet(UNIFORM/f'{year}_origins.parquet'))
            target = y[ev]
            np.testing.assert_array_equal(target,np.load(UNIFORM/f'{year}_targets.npy'))
            x = build(raw,w,cutoff,1)
            if x.shape != (len(y),323):
                raise ValueError('Feature dimension changed')
            weights,info = regime_weights(x[train,71])
            pd.DataFrame(dict(source_index=w['indices'][train],last_observed_aqi=x[train,71],
                weight=weights)).to_parquet(output/f'{year}_train_weights.parquet',index=False)
            np.save(output/f'{year}_train_labels.npy',labels[train],allow_pickle=False)
            probability = np.empty((int(ev.sum()),19),dtype=float)
            for j in range(19):
                model = lgb.LGBMClassifier(objective='binary',n_estimators=300,learning_rate=.04,
                    num_leaves=24,min_child_samples=30,random_state=42,n_jobs=4,
                    verbosity=-1,deterministic=True,force_col_wise=True)
                model.fit(x[train],labels[train,j],sample_weight=weights)
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
            up = np.load(UNIFORM/f'{year}_probabilities.npy')
            ustart = decode(up,uniform['folds'][year]['threshold'])
            origin = meta.origin_aqi.to_numpy()
            polluted = origin>100
            dev = (meta.partition.to_numpy()=='dev')&polluted
            test = (meta.partition.to_numpy()=='test')&polluted
            raw_dev = recovery_metrics(target[dev],control[dev])
            threshold,grid = choose(target[dev],probability[dev],raw_dev['premature_start_fraction_all'])
            truth,predicted = recovery_start(target),decode(probability,threshold)
            fold = dict(train_end_year=cutoff,train_n=int(train.sum()),
                train_clean=int((train&~w['polluted']).sum()),train_polluted=int((train&w['polluted']).sum()),
                weights=info,threshold=threshold,development_grid=grid,
                dev=start_metrics(truth[dev],predicted[dev]),test=start_metrics(truth[test],predicted[test]),
                uniform_test=start_metrics(truth[test],ustart[test]),
                raw_test=recovery_metrics(target[test],control[test]),
                buffered_test=recovery_metrics(target[test],buffered[test]),qualifying_diagnostics={})
            for part in ('dev','test'):
                k = meta.partition.to_numpy()==part
                fold['qualifying_diagnostics'][part] = {
                    'weighted':qualifying_diagnostics(target[k],probability[k],predicted[k],origin[k]),
                    'uniform':qualifying_diagnostics(target[k],up[k],ustart[k],origin[k])}
            report['folds'][year] = fold
            for name,values in dict(truth=truth,weighted=predicted,uniform=ustart,
                    raw=recovery_start(control),buffered=recovery_start(buffered)).items():
                pooled[name].append(values[test])
            k = meta.partition.to_numpy()=='test'
            for name,values in dict(target=target,probability=probability,uniform_probability=up,
                    predicted=predicted,uniform_predicted=ustart,origin=origin).items():
                qual[name].append(values[k])
            save()
            del x
        a = {k:np.concatenate(v) for k,v in pooled.items()}
        report['pooled_test'] = {k:start_metrics(a['truth'],a[k]) for k in ('weighted','uniform','raw','buffered')}
        q = {k:np.concatenate(v) for k,v in qual.items()}
        report['pooled_qualifying_diagnostics'] = {
            'weighted':qualifying_diagnostics(q['target'],q['probability'],q['predicted'],q['origin']),
            'uniform':qualifying_diagnostics(q['target'],q['uniform_probability'],q['uniform_predicted'],q['origin'])}
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
