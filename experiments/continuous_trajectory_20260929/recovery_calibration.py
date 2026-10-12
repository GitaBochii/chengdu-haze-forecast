"""Development-only conservative recovery decisions for frozen AQI curves."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from full_curve_recovery import recovery_start, recovery_metrics

HERE = Path(__file__).resolve().parent
SOURCE_RUN = HERE / 'full_curve_runs/20260930T033524196780Z'
ARMS = ('lag1_trend', 'lag1_chemistry')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def buffered_prediction(prediction, delta):
    if not isinstance(delta, (int, np.integer)) or not 0 <= delta <= 101:
        raise ValueError('Buffer must be an integer in [0,101]')
    return np.asarray(prediction, dtype=float) + delta


def feasible(metrics, raw_early):
    return (metrics['fpr'] is not None and metrics['fpr'] <= .05
            and metrics['premature_start_fraction_all'] <= raw_early)


def select_candidate(rows, raw_early):
    candidates = [r for r in rows if feasible(r, raw_early)]
    if not candidates:
        raise ValueError('No candidate establishes the development risk budget')
    return min(candidates, key=lambda r: (
        r['capped_min_T_20_mae_hours'],
        -(r['recall'] if r['recall'] is not None else 0), r['fpr'], -r['delta']))


def calibrate(target, prediction):
    if len(target) == 0 or recovery_metrics(target, prediction)['actual_right_censored'] == 0:
        raise ValueError('Development requires an actual nonrecovery population')
    raw = recovery_metrics(target, prediction)
    rows = [dict(delta=d, **recovery_metrics(target, buffered_prediction(prediction, d)))
            for d in range(102)]
    return select_candidate(rows, raw['premature_start_fraction_all'])['delta'], rows


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')


def main():
    ref = json.loads((SOURCE_RUN / 'report.json').read_text('utf-8'))
    audit = json.loads((SOURCE_RUN / 'independent_audit.json').read_text('utf-8'))
    if ref['status'] != 'completed_fixed_protocol' or audit['status'] != 'passed':
        raise ValueError('Frozen source is not complete and verified')
    inputs = {str(SOURCE_RUN / n): h for n, h in ref['artifact_sha256'].items()
              if n.endswith(('.npy', '.parquet'))}
    inputs[str(SOURCE_RUN / 'report.json')] = sha(SOURCE_RUN / 'report.json')
    inputs[str(SOURCE_RUN / 'independent_audit.json')] = sha(SOURCE_RUN / 'independent_audit.json')
    inputs[str(HERE / 'full_curve_recovery.py')] = sha(HERE / 'full_curve_recovery.py')
    for p, h in inputs.items():
        if sha(p) != h:
            raise ValueError(f'Frozen artifact changed: {p}')
    run = HERE / 'recovery_calibration_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True)
    report = dict(status='completed_development_calibration', source_run=str(SOURCE_RUN),
        input_sha256=inputs, protocol_sha256=sha(HERE / 'RECOVERY_CALIBRATION_PROTOCOL.md'),
        code_sha256=sha(__file__), folds={}, pooled_test={},
        historical_stress_test=True, blind_test=False, prospective=False, deployed=False,
        concentration_predictions_changed=False, fpr_budget=.05, grid=list(range(102)))
    selected, grid = {}, {}
    pooled = {arm: {'y': [], 'raw': [], 'calibrated': []} for arm in ARMS}
    for year in ('2021', '2022', '2023'):
        meta = pd.read_parquet(SOURCE_RUN / f'{year}_origins.parquet')
        y = np.load(SOURCE_RUN / f'{year}_targets.npy')
        report['folds'][year], selected[year], grid[year] = {}, {}, {}
        dev = (meta['partition'].to_numpy() == 'dev') & (meta['origin_aqi'].to_numpy() > 100)
        test = (meta['partition'].to_numpy() == 'test') & (meta['origin_aqi'].to_numpy() > 100)
        for arm in ARMS:
            p = np.load(SOURCE_RUN / f'{year}_{arm}.npy')
            delta, rows = calibrate(y[dev], p[dev])
            adjusted = buffered_prediction(p, delta)
            raw_dev = recovery_metrics(y[dev], p[dev])
            selected[year][arm] = delta
            grid[year][arm] = [dict(delta=r['delta'], metrics={k: v for k, v in r.items() if k != 'delta'},
                feasible=feasible(r, raw_dev['premature_start_fraction_all'])) for r in rows]
            report['folds'][year][arm] = dict(delta=delta, raw_dev=raw_dev,
                dev=recovery_metrics(y[dev], adjusted[dev]),
                raw_test=recovery_metrics(y[test], p[test]), test=recovery_metrics(y[test], adjusted[test]))
            starts = meta.copy()
            starts['true_start'] = recovery_start(y)
            starts['raw_start'] = recovery_start(p)
            starts['calibrated_start'] = recovery_start(adjusted)
            starts.to_parquet(run / f'{year}_{arm}_starts.parquet', index=False)
            pooled[arm]['y'].append(y[test])
            pooled[arm]['raw'].append(p[test])
            pooled[arm]['calibrated'].append(adjusted[test])
    for arm, arrays in pooled.items():
        a = {k: np.concatenate(v) for k, v in arrays.items()}
        report['pooled_test'][arm] = dict(raw=recovery_metrics(a['y'], a['raw']),
            calibrated=recovery_metrics(a['y'], a['calibrated']))
    write_json(run / 'selected.json', selected)
    write_json(run / 'grid.json', grid)
    report['artifact_sha256'] = {p.name: sha(p) for p in run.iterdir()}
    write_json(run / 'report.json', report)
    print(run)
    print(json.dumps(report['pooled_test'], indent=2))


if __name__ == '__main__':
    main()
