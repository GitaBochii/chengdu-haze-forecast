"""Direct continuous six-hour maxima rather than pointwise curve medians."""
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
from recovery_windows import start_metrics, BASE, BUFFER

HERE = Path(__file__).resolve().parent


def maximum_targets(target):
    a = np.asarray(target)
    if a.ndim != 2 or a.shape[1] != 24 or not np.isfinite(a).all() or (a<0).any():
        raise ValueError('Complete finite nonnegative 24-hour targets required')
    return np.lib.stride_tricks.sliding_window_view(a,6,axis=1).max(axis=2)


def maximum_start(prediction, delta=0):
    p = np.asarray(prediction)
    if p.ndim != 2 or p.shape[1] != 19 or not np.isfinite(p).all() or (p<0).any():
        raise ValueError('Finite nineteen-window maxima required')
    if not isinstance(delta,(int,np.integer)) or not 0<=delta<=101:
        raise ValueError('Frozen integer buffer required')
    eligible = p.astype(float)+delta<=100
    return np.where(eligible.any(axis=1),eligible.argmax(axis=1)+1,20)


def select_buffer(target, prediction, raw_early):
    truth = recovery_start(target)
    if not len(truth) or not (truth==20).any():
        raise ValueError('Development needs actual nonrecoveries')
    grid = []
    for delta in range(102):
        m = start_metrics(truth,maximum_start(prediction,delta))
        grid.append(dict(delta=delta,metrics=m,
            feasible=m['fpr']<=.05 and m['premature_start_fraction_all']<=raw_early))
    best = min([r for r in grid if r['feasible']],key=lambda r:(
        r['metrics']['capped_min_T_20_mae_hours'],-(r['metrics']['recall'] or 0),
        r['metrics']['fpr'],-r['delta']))
    return best['delta'],grid


def max_diagnostics(target, prediction, curve, origin):
    truth,derived = maximum_targets(target),maximum_targets(curve)
    return {name:dict(origins=int(mask.sum()),
        mae=float(np.abs(prediction[mask].astype(float)-truth[mask]).mean()),
        curve_derived_mae=float(np.abs(derived[mask].astype(float)-truth[mask]).mean()))
        for name,mask in [('all',np.ones(len(target),bool)),('clean_origin',origin<=100),
                          ('polluted_origin',origin>100)]}


def main():
    started = time.perf_counter()
    if sha(SOURCE)!=SOURCE_SHA:
        raise ValueError('Frozen source changed')
    ref = json.loads((BASE/'report.json').read_text('utf-8'))
    selected = json.loads((BUFFER/'selected.json').read_text('utf-8'))
    inputs = {str(SOURCE):SOURCE_SHA}
    for n in ('trajectory.py','forward.py','delayed_chemistry.py','issue_delay_features.py',
              'feature_ablation.py','full_curve_recovery.py','recovery_windows.py'):
        inputs[str(HERE/n)] = sha(HERE/n)
    for base in (BASE,BUFFER):
        audit = json.loads((base/'independent_audit.json').read_text('utf-8'))
        if audit['status']!='passed' or sha(base/'report.json')!=audit['report_sha256']:
            raise ValueError('Control is not frozen and verified')
        for n in ('report.json','independent_audit.json'):
            inputs[str(base/n)] = sha(base/n)
    inputs[str(BUFFER/'selected.json')] = sha(BUFFER/'selected.json')
    for n,h in ref['artifact_sha256'].items():
        if n.endswith(('.npy','.parquet')):
            inputs[str(BASE/n)] = h
    for p,h in inputs.items():
        if sha(Path(p))!=h:
            raise ValueError(f'Frozen dependency changed: {p}')
    raw = pd.read_parquet(SOURCE)
    w = windows(raw)
    y = w['target'][:,:,2]
    labels = maximum_targets(y)
    output = HERE/'six_hour_max_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    report = dict(status='running',source_run=str(BASE),buffer_run=str(BUFFER),
        input_sha256=inputs,code_sha256=sha(Path(__file__)),
        protocol_sha256=sha(HERE/'SIX_HOUR_MAX_PROTOCOL.md'),lag=1,feature_dimension=323,
        folds={},models={},pooled_test={},historical_stress_test=True,blind_test=False,
        prospective=False,deployed=False,concentration_predictions_changed=False)
    pooled = {k:[] for k in ('truth','maximum','raw','buffered')}
    diagnosis = {k:[] for k in ('target','prediction','curve','origin')}
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
            pd.testing.assert_frame_equal(meta,pd.read_parquet(BASE/f'{year}_origins.parquet'))
            target = y[ev]
            np.testing.assert_array_equal(target,np.load(BASE/f'{year}_targets.npy'))
            x = build(raw,w,cutoff,1)
            if x.shape!=(len(y),323):
                raise ValueError('Feature dimension changed')
            pred = np.empty((int(ev.sum()),19),dtype='float32')
            np.save(output/f'{year}_train_maximum_targets.npy',labels[train],allow_pickle=False)
            for j in range(19):
                model = lgb.LGBMRegressor(objective='regression_l1',n_estimators=300,learning_rate=.04,
                    num_leaves=24,min_child_samples=30,random_state=42,n_jobs=4,
                    verbosity=-1,deterministic=True,force_col_wise=True)
                model.fit(x[train],labels[train,j])
                p = model.predict(x[ev]).clip(min=0).astype('float32')
                path = output/f'{year}_window{j+1}.txt'
                save_booster(model.booster_,path)
                np.testing.assert_array_equal(p,load_booster(path).predict(x[ev],num_threads=4).clip(min=0).astype('float32'))
                pred[:,j] = p
                report['models'][path.name] = sha(path)
                print(json.dumps(dict(year=year,window=j+1)),flush=True)
            meta.to_parquet(output/f'{year}_origins.parquet',index=False)
            for name,values in dict(targets=target,maximum_targets=labels[ev],maximum_predictions=pred).items():
                np.save(output/f'{year}_{name}.npy',values,allow_pickle=False)
            curve = np.load(BASE/f'{year}_lag1_chemistry.npy')
            buffered = curve.astype(float)+selected[year]['lag1_chemistry']
            origin = meta.origin_aqi.to_numpy()
            dev = (meta.partition.to_numpy()=='dev')&(origin>100)
            test = (meta.partition.to_numpy()=='test')&(origin>100)
            raw_dev = recovery_metrics(target[dev],curve[dev])
            delta,grid = select_buffer(target[dev],pred[dev],raw_dev['premature_start_fraction_all'])
            truth,predicted = recovery_start(target),maximum_start(pred,delta)
            report['folds'][year] = dict(train_end_year=cutoff,train_n=int(train.sum()),
                train_clean=int((train&~w['polluted']).sum()),train_polluted=int((train&w['polluted']).sum()),
                delta=delta,development_grid=grid,dev=start_metrics(truth[dev],predicted[dev]),
                test=start_metrics(truth[test],predicted[test]),raw_test=recovery_metrics(target[test],curve[test]),
                buffered_test=recovery_metrics(target[test],buffered[test]),max_diagnostics={})
            for part in ('dev','test'):
                k = meta.partition.to_numpy()==part
                report['folds'][year]['max_diagnostics'][part] = max_diagnostics(target[k],pred[k],curve[k],origin[k])
            for name,values in dict(truth=truth,maximum=predicted,raw=recovery_start(curve),
                                     buffered=recovery_start(buffered)).items():
                pooled[name].append(values[test])
            k = meta.partition.to_numpy()=='test'
            for name,values in dict(target=target,prediction=pred,curve=curve,origin=origin).items():
                diagnosis[name].append(values[k])
            save()
            del x
        a = {k:np.concatenate(v) for k,v in pooled.items()}
        report['pooled_test'] = {k:start_metrics(a['truth'],a[k]) for k in ('maximum','raw','buffered')}
        q = {k:np.concatenate(v) for k,v in diagnosis.items()}
        report['pooled_max_diagnostics'] = max_diagnostics(q['target'],q['prediction'],q['curve'],q['origin'])
        c,b = report['pooled_test']['maximum'],report['pooled_test']['buffered']
        report['clear_improvement_gate'] = dict(
            timing_at_least_10pct_better=c['capped_min_T_20_mae_hours']<=.9*b['capped_min_T_20_mae_hours'],
            recall_at_least_5pp_better=c['recall']>=b['recall']+.05,pooled_fpr_at_most_5pct=c['fpr']<=.05,
            all_years_fpr_at_most_5pct=all(f['test']['fpr']<=.05 for f in report['folds'].values()),
            no_year_timing_worse=all(f['test']['capped_min_T_20_mae_hours']<=f['buffered_test']['capped_min_T_20_mae_hours'] for f in report['folds'].values()))
        report['clear_improvement_gate']['passed'] = all(report['clear_improvement_gate'].values())
        if any(sha(Path(p))!=h for p,h in inputs.items()):
            raise ValueError('Frozen input changed during experiment')
        report.update(status='completed_fixed_protocol',elapsed_seconds=time.perf_counter()-started)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        report['artifact_sha256'] = {p.name:sha(p) for p in output.iterdir() if p.suffix in ('.txt','.npy','.parquet')}
        save()
    print(json.dumps(dict(metrics=report['pooled_test'],gate=report['clear_improvement_gate']),indent=2),flush=True)


if __name__=='__main__':
    main()
