"""Shared person-period classifier for discrete AQI recovery survival."""
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
THRESHOLDS = tuple(i / 20 for i in range(1, 21)) + (1.000001,)
PARAMS = dict(objective='binary', n_estimators=300, learning_rate=.04,
    num_leaves=24, min_child_samples=30, random_state=42, n_jobs=4,
    verbosity=-1, deterministic=True, force_col_wise=True)


def risk_rows(true_start):
    t = np.asarray(true_start)
    if t.ndim != 1 or not np.isfinite(t).all() or ((t < 1) | (t > 20) | (t != np.floor(t))).any():
        raise ValueError('Integer starts1..19 or censored sentinel20 required')
    lengths = np.minimum(t, 19).astype(int)
    origins = np.repeat(np.arange(len(t)), lengths)
    offsets = np.cumsum(lengths) - lengths
    leads = np.arange(int(lengths.sum())) - np.repeat(offsets, lengths) + 1
    events = (leads == t[origins]) & (t[origins] < 20)
    return origins, leads, events


def hazard_decode(hazards, threshold):
    h = np.asarray(hazards, dtype=float)
    if h.ndim != 2 or h.shape[1] != 19 or not np.isfinite(h).all() or ((h < 0) | (h > 1)).any():
        raise ValueError('Finite nineteen hazards within[0,1] required')
    if not np.isfinite(threshold) or not 0 < threshold <= 1.000001:
        raise ValueError('Positive valid recovery probability threshold required')
    cdf = 1 - np.cumprod(1 - h, axis=1)
    # Retain nonrecovery mass when minimizing absolute error of min(T,20).
    reached = cdf >= .5
    median = np.where(reached.any(axis=1), reached.argmax(axis=1) + 1, 20)
    return np.where(cdf[:, -1] >= threshold, median, 20), cdf


def select_threshold(truth, hazards, raw_early):
    truth = np.asarray(truth)
    if not len(truth) or not (truth == 20).any():
        raise ValueError('Development requires observed nonrecoveries')
    grid = []
    for threshold in THRESHOLDS:
        pred, _ = hazard_decode(hazards, threshold)
        m = start_metrics(truth, pred)
        grid.append(dict(threshold=threshold, metrics=m,
            feasible=m['fpr'] <= .05 and m['premature_start_fraction_all'] <= raw_early))
    best = min([r for r in grid if r['feasible']], key=lambda r: (
        r['metrics']['capped_min_T_20_mae_hours'], -(r['metrics']['recall'] or 0),
        r['metrics']['fpr'], -r['threshold']))
    return best['threshold'], grid


def main():
    started = time.perf_counter()
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Frozen source changed')
    ref = json.loads((BASE / 'report.json').read_text('utf-8'))
    selected = json.loads((BUFFER / 'selected.json').read_text('utf-8'))
    inputs = {str(SOURCE): SOURCE_SHA}
    for n in ('trajectory.py', 'forward.py', 'delayed_chemistry.py', 'issue_delay_features.py',
              'feature_ablation.py', 'full_curve_recovery.py', 'recovery_windows.py'):
        inputs[str(HERE / n)] = sha(HERE / n)
    for base in (BASE, BUFFER):
        audit = json.loads((base / 'independent_audit.json').read_text('utf-8'))
        if audit['status'] != 'passed' or sha(base / 'report.json') != audit['report_sha256']:
            raise ValueError('Control is not frozen and verified')
        for n in ('report.json', 'independent_audit.json'):
            inputs[str(base / n)] = sha(base / n)
    inputs[str(BUFFER / 'selected.json')] = sha(BUFFER / 'selected.json')
    for n, h in ref['artifact_sha256'].items():
        if n.endswith(('.npy', '.parquet')):
            inputs[str(BASE / n)] = h
    for p, h in inputs.items():
        if sha(Path(p)) != h:
            raise ValueError(f'Frozen dependency changed: {p}')
    raw = pd.read_parquet(SOURCE)
    w = windows(raw)
    y = w['target'][:, :, 2]
    starts = recovery_start(y)
    output = HERE / 'discrete_hazard_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    report = dict(status='running', source_run=str(BASE), buffer_run=str(BUFFER),
        input_sha256=inputs, code_sha256=sha(Path(__file__)),
        protocol_sha256=sha(HERE / 'DISCRETE_HAZARD_PROTOCOL.md'),
        params=PARAMS, thresholds=THRESHOLDS, lightgbm_version=lgb.__version__,
        lag=1, feature_dimension=324, base_feature_dimension=323,
        folds={}, models={}, pooled_test={}, historical_stress_test=True,
        blind_test=False, prospective=False, deployed=False,
        concentration_predictions_changed=False)
    pooled = {k: [] for k in ('truth', 'hazard', 'raw', 'buffered')}

    def save():
        tmp = output / 'report.tmp'
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False), 'utf-8')
        tmp.replace(output / 'report.json')

    save()
    print(output, flush=True)
    try:
        for cutoff in (2019, 2020, 2021):
            year = str(cutoff + 2)
            parts = fold_parts(w['timestamps'], cutoff)
            train, ev = parts == 'train', np.isin(parts, ['dev', 'test'])
            meta = pd.DataFrame(dict(source_index=w['indices'][ev], timestamp=w['timestamps'][ev],
                partition=parts[ev], origin_aqi=w['history'][ev, -1, 2]))
            pd.testing.assert_frame_equal(meta, pd.read_parquet(BASE / f'{year}_origins.parquet'))
            target = y[ev]
            np.testing.assert_array_equal(target, np.load(BASE / f'{year}_targets.npy'))
            x = build(raw, w, cutoff, 1)
            if x.shape != (len(y), 323):
                raise ValueError('Feature dimension changed')
            row, lead, event = risk_rows(starts[train])
            train_ids = w['indices'][train]
            pd.DataFrame(dict(source_index=train_ids, timestamp=w['timestamps'][train],
                true_start=starts[train])).to_parquet(output / f'{year}_train_origins.parquet', index=False)
            pd.DataFrame(dict(source_index=train_ids[row], lead=lead, event=event)).to_parquet(
                output / f'{year}_train_risk.parquet', index=False)
            meta.to_parquet(output / f'{year}_origins.parquet', index=False)
            np.save(output / f'{year}_targets.npy', target, allow_pickle=False)
            train_x = np.column_stack([x[train][row], lead]).astype('float32')
            model = lgb.LGBMClassifier(**PARAMS)
            model.fit(train_x, event.astype('int8'))
            del train_x
            path = output / f'{year}_hazard.txt'
            save_booster(model.booster_, path)
            reloaded = load_booster(path)
            hazards = np.empty((int(ev.sum()), 19), dtype='float64')
            for k in range(1, 20):
                ev_x = np.column_stack([x[ev], np.full(int(ev.sum()), k)]).astype('float32')
                p = model.predict_proba(ev_x)[:, 1]
                np.testing.assert_array_equal(p, reloaded.predict(ev_x, num_threads=4))
                hazards[:, k - 1] = p
            del ev_x
            report['models'][path.name] = sha(path)
            np.save(output / f'{year}_hazards.npy', hazards, allow_pickle=False)
            origin = meta.origin_aqi.to_numpy()
            dev = (meta.partition.to_numpy() == 'dev') & (origin > 100)
            test = (meta.partition.to_numpy() == 'test') & (origin > 100)
            curve = np.load(BASE / f'{year}_lag1_chemistry.npy')
            buffered = curve.astype(float) + selected[year]['lag1_chemistry']
            truth = recovery_start(target)
            raw_dev = recovery_metrics(target[dev], curve[dev])
            threshold, grid = select_threshold(truth[dev], hazards[dev], raw_dev['premature_start_fraction_all'])
            predicted, _ = hazard_decode(hazards, threshold)
            report['folds'][year] = dict(train_end_year=cutoff, train_n=int(train.sum()),
                train_clean=int((train & ~w['polluted']).sum()), train_polluted=int((train & w['polluted']).sum()),
                train_risk_n=len(row), threshold=threshold, development_grid=grid,
                dev=start_metrics(truth[dev], predicted[dev]), test=start_metrics(truth[test], predicted[test]),
                raw_test=recovery_metrics(target[test], curve[test]),
                buffered_test=recovery_metrics(target[test], buffered[test]))
            for name, values in dict(truth=truth, hazard=predicted, raw=recovery_start(curve),
                                     buffered=recovery_start(buffered)).items():
                pooled[name].append(values[test])
            save()
            print(json.dumps(dict(year=year, threshold=threshold, metrics=report['folds'][year]['test'])), flush=True)
            del x
        a = {k: np.concatenate(v) for k, v in pooled.items()}
        report['pooled_test'] = {k: start_metrics(a['truth'], a[k]) for k in ('hazard', 'raw', 'buffered')}
        c, b = report['pooled_test']['hazard'], report['pooled_test']['buffered']
        gate = dict(timing_at_least_10pct_better=c['capped_min_T_20_mae_hours'] <= .9 * b['capped_min_T_20_mae_hours'],
            recall_at_least_5pp_better=c['recall'] >= b['recall'] + .05, pooled_fpr_at_most_5pct=c['fpr'] <= .05,
            all_years_fpr_at_most_5pct=all(f['test']['fpr'] <= .05 for f in report['folds'].values()),
            no_year_timing_worse=all(f['test']['capped_min_T_20_mae_hours'] <= f['buffered_test']['capped_min_T_20_mae_hours'] for f in report['folds'].values()))
        gate['passed'] = all(gate.values())
        report['clear_improvement_gate'] = gate
        if any(sha(Path(p)) != h for p, h in inputs.items()):
            raise ValueError('Frozen input changed during experiment')
        if sha(Path(__file__)) != report['code_sha256'] or sha(HERE / 'DISCRETE_HAZARD_PROTOCOL.md') != report['protocol_sha256']:
            raise ValueError('Protocol or implementation changed during experiment')
        report.update(status='completed_fixed_protocol', elapsed_seconds=time.perf_counter() - started)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['artifact_sha256'] = {p.name: sha(p) for p in output.iterdir() if p.suffix in ('.txt', '.npy', '.parquet')}
        save()
    print(json.dumps(dict(metrics=report['pooled_test'], gate=report['clear_improvement_gate']), indent=2), flush=True)


if __name__ == '__main__':
    main()
