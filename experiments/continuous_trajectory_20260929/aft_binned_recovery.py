"""Interval-censored AFT for integer six-hour AQI recovery starts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
import xgboost as xgb
from trajectory import SOURCE, SOURCE_SHA, windows, sha
from forward import fold_parts
from delayed_chemistry import build
from full_curve_recovery import recovery_start, recovery_metrics
from recovery_windows import start_metrics, BASE, BUFFER

HERE = Path(__file__).resolve().parent
SCALES = (.35, .7, 1.2)
THRESHOLDS = tuple(i / 20 for i in range(1, 21)) + (1.000001,)
PARAMS = dict(objective='survival:aft', eval_metric='aft-nloglik',
    aft_loss_distribution='normal', eta=.05, max_depth=4, min_child_weight=8,
    subsample=.9, colsample_bytree=.8, reg_lambda=3., seed=42,
    tree_method='hist', nthread=4, verbosity=0)
ROUNDS = 500


def binned_bounds(start):
    t = np.asarray(start)
    if t.ndim != 1 or not np.isfinite(t).all() or ((t < 1) | (t > 20) | (t != np.floor(t))).any():
        raise ValueError('Integer recovery starts 1..19 or censoring sentinel20 required')
    lower = np.where(t == 1, 0., t.astype(float) - .5)
    upper = np.where(t == 20, np.inf, t.astype(float) + .5)
    return lower, upper


def survival_decode(margin, sigma, threshold):
    mu = np.asarray(margin, dtype=float)
    if mu.ndim != 1 or not np.isfinite(mu).all():
        raise ValueError('Finite one-dimensional log-location margins required')
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(threshold) or not 0 <= threshold <= 1.000001:
        raise ValueError('Positive finite noise scale and valid probability threshold required')
    p = ndtr((np.log(19.5) - mu) / sigma)
    log_time = mu + sigma * ndtri(np.clip(.5 * p, 1e-12, 1 - 1e-12))
    # Only the inverse-CDF time calculation is stabilized; gating retains raw CDF.
    continuous = np.exp(np.clip(log_time, -745., np.log(19.5)))
    integer = np.clip(np.floor(continuous + .5), 1, 19).astype(int)
    return np.where(p >= threshold, integer, 20), p


def select_candidate(truth, margins, raw_early):
    truth = np.asarray(truth)
    if not len(truth) or not (truth == 20).any():
        raise ValueError('Development requires observed nonrecoveries')
    grid = []
    for sigma in SCALES:
        for threshold in THRESHOLDS:
            pred, _ = survival_decode(margins[sigma], sigma, threshold)
            m = start_metrics(truth, pred)
            grid.append(dict(sigma=sigma, threshold=threshold, metrics=m,
                feasible=m['fpr'] <= .05 and m['premature_start_fraction_all'] <= raw_early))
    feasible = [r for r in grid if r['feasible']]
    best = min(feasible, key=lambda r: (
        r['metrics']['capped_min_T_20_mae_hours'], -(r['metrics']['recall'] or 0),
        r['metrics']['fpr'], -r['threshold'], -r['sigma']))
    return best['sigma'], best['threshold'], grid


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
    output = HERE / 'aft_binned_recovery_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    report = dict(status='running', source_run=str(BASE), buffer_run=str(BUFFER),
        input_sha256=inputs, code_sha256=sha(Path(__file__)),
        protocol_sha256=sha(HERE / 'AFT_BINNED_RECOVERY_PROTOCOL.md'),
        params=PARAMS, num_boost_round=ROUNDS, scales=SCALES, thresholds=THRESHOLDS,
        xgboost_version=xgb.__version__, lag=1, feature_dimension=323,
        folds={}, models={}, pooled_test={}, historical_stress_test=True,
        blind_test=False, prospective=False, deployed=False,
        concentration_predictions_changed=False)
    pooled = {k: [] for k in ('truth', 'aft', 'raw', 'buffered')}

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
            lo, hi = binned_bounds(starts[train])
            np.save(output / f'{year}_train_bounds.npy', np.column_stack([lo, hi]), allow_pickle=False)
            pd.DataFrame(dict(source_index=w['indices'][train], timestamp=w['timestamps'][train])).to_parquet(
                output / f'{year}_train_origins.parquet', index=False)
            meta.to_parquet(output / f'{year}_origins.parquet', index=False)
            np.save(output / f'{year}_targets.npy', target, allow_pickle=False)
            dtrain = xgb.DMatrix(x[train].astype('float32'), label_lower_bound=lo, label_upper_bound=hi)
            deval = xgb.DMatrix(x[ev].astype('float32'))
            margins = {}
            for sigma in SCALES:
                params = dict(PARAMS, aft_loss_distribution_scale=sigma)
                model = xgb.train(params, dtrain, num_boost_round=ROUNDS)
                margin = model.predict(deval, output_margin=True).astype('float32')
                np.testing.assert_allclose(model.predict(deval), np.exp(margin), rtol=2e-6, atol=1e-6)
                if not np.isfinite(margin).all():
                    raise ValueError('Nonfinite AFT margins')
                path = output / f'{year}_sigma{sigma}.ubj'
                path.write_bytes(model.save_raw(raw_format='ubj'))
                reloaded = xgb.Booster()
                reloaded.load_model(bytearray(path.read_bytes()))
                np.testing.assert_array_equal(margin, reloaded.predict(deval, output_margin=True).astype('float32'))
                np.save(output / f'{year}_sigma{sigma}_margins.npy', margin, allow_pickle=False)
                margins[sigma] = margin
                report['models'][path.name] = sha(path)
                print(json.dumps(dict(year=year, sigma=sigma)), flush=True)
                save()
            origin = meta.origin_aqi.to_numpy()
            dev = (meta.partition.to_numpy() == 'dev') & (origin > 100)
            test = (meta.partition.to_numpy() == 'test') & (origin > 100)
            curve = np.load(BASE / f'{year}_lag1_chemistry.npy')
            buffered = curve.astype(float) + selected[year]['lag1_chemistry']
            truth = recovery_start(target)
            raw_dev = recovery_metrics(target[dev], curve[dev])
            sigma, threshold, grid = select_candidate(truth[dev], {s: m[dev] for s, m in margins.items()},
                raw_dev['premature_start_fraction_all'])
            predicted, _ = survival_decode(margins[sigma], sigma, threshold)
            report['folds'][year] = dict(train_end_year=cutoff, train_n=int(train.sum()),
                train_clean=int((train & ~w['polluted']).sum()), train_polluted=int((train & w['polluted']).sum()),
                sigma=sigma, threshold=threshold, development_grid=grid,
                dev=start_metrics(truth[dev], predicted[dev]), test=start_metrics(truth[test], predicted[test]),
                raw_test=recovery_metrics(target[test], curve[test]),
                buffered_test=recovery_metrics(target[test], buffered[test]))
            for name, values in dict(truth=truth, aft=predicted, raw=recovery_start(curve),
                                     buffered=recovery_start(buffered)).items():
                pooled[name].append(values[test])
            save()
            del x, dtrain, deval
        a = {k: np.concatenate(v) for k, v in pooled.items()}
        report['pooled_test'] = {k: start_metrics(a['truth'], a[k]) for k in ('aft', 'raw', 'buffered')}
        c, b = report['pooled_test']['aft'], report['pooled_test']['buffered']
        gate = dict(timing_at_least_10pct_better=c['capped_min_T_20_mae_hours'] <= .9 * b['capped_min_T_20_mae_hours'],
            recall_at_least_5pp_better=c['recall'] >= b['recall'] + .05, pooled_fpr_at_most_5pct=c['fpr'] <= .05,
            all_years_fpr_at_most_5pct=all(f['test']['fpr'] <= .05 for f in report['folds'].values()),
            no_year_timing_worse=all(f['test']['capped_min_T_20_mae_hours'] <= f['buffered_test']['capped_min_T_20_mae_hours'] for f in report['folds'].values()))
        gate['passed'] = all(gate.values())
        report['clear_improvement_gate'] = gate
        if any(sha(Path(p)) != h for p, h in inputs.items()):
            raise ValueError('Frozen input changed during experiment')
        if sha(Path(__file__)) != report['code_sha256'] or sha(HERE / 'AFT_BINNED_RECOVERY_PROTOCOL.md') != report['protocol_sha256']:
            raise ValueError('Protocol or implementation changed during experiment')
        report.update(status='completed_fixed_protocol', elapsed_seconds=time.perf_counter() - started)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['artifact_sha256'] = {p.name: sha(p) for p in output.iterdir() if p.suffix in ('.ubj', '.npy', '.parquet')}
        save()
    print(json.dumps(dict(metrics=report['pooled_test'], gate=report['clear_improvement_gate']), indent=2), flush=True)


if __name__ == '__main__':
    main()
