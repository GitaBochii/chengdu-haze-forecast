"""Independently verify saved nested-selection probabilities, errors and gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.benchmark_recovery_head import load_experiment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_scores(frame):
    selected = frame.loc[frame.actual_hours.notna() & frame.label_verified_in_scoring_window].copy()
    selected['error'] = np.abs(np.minimum(selected.actual_hours, 68) - np.minimum(selected.predicted_hours, 68))
    short, tail = selected.loc[selected.actual_hours < 68], selected.loc[selected.actual_hours >= 68]
    return {'row_mae': float(selected.error.mean()),
            'event_equal_mae': float(selected.groupby('event_id').error.mean().mean()),
            'short_event_equal_mae': float(short.groupby('event_id').error.mean().mean()),
            'tail_event_equal_mae': float(tail.groupby('event_id').error.mean().mean()),
            'tail_events': [int(x) for x in sorted(tail.event_id.unique())],
            'tail_rows': len(tail), 'rows': len(selected), 'events': int(selected.event_id.nunique())}, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = json.loads((args.run / 'report.json').read_text('utf-8'))
    assert report['decision'] != 'error'
    for folder, mapping in ((root / report['configuration']['data'], report['inputs_before']),
                             (root / 'artifacts/cnemc_2025_deployment_candidate', report['deployment_before'])):
        for name, expected in mapping.items():
            assert sha(folder / name) == expected
    assert sha(args.run / 'rule_selection.json') == report['rule_selection_sha256']
    results, events = {}, []
    for val_year, fold in report['folds'].items():
        folder = args.run / f'validation_{val_year}'
        trace = json.loads((folder / 'training_trace.json').read_text('utf-8'))
        assert len(trace) == report['configuration']['epochs'] == fold['epochs_ran']
        for name, expected in fold['frozen'].items():
            assert sha(folder / name) == expected
        for rule in ('legacy', 'balanced'):
            chosen = min(trace, key=lambda row: tuple(row['selection_keys'][rule]))
            assert chosen['epoch'] == fold['selections'][rule]['epoch']
            assert fold['selections'][rule]['candidate_epochs'] == list(range(1, len(trace) + 1))
        panel, _, _, _, partitions = load_experiment(SimpleNamespace(**fold['configuration']))
        np.testing.assert_array_equal(panel['scaler'].mean, fold['scaler']['mean'])
        np.testing.assert_array_equal(panel['scaler'].scale, fold['scaler']['scale'])
        assert panel['nodes'] == fold['nodes']
        last_train = np.concatenate([partitions[y] for y in range(2014, int(val_year))])[-1]
        assert panel['timestamps'][last_train + 72] < pd.Timestamp(f'{val_year}-01-01', tz='Asia/Shanghai')
        results[val_year] = {}
        for year, saved in fold['years'].items():
            pair, checked = [], {}
            for rule in ('legacy', 'balanced'):
                frame = pd.read_parquet(folder / f'{rule}_{year}_recovery.parquet')
                pair.append(frame)
                assert not frame.timestamp.duplicated().any()
                np.testing.assert_array_equal(pd.DatetimeIndex(frame.timestamp).asi8,
                                               panel['timestamps'][partitions[int(year)]].asi8)
                with np.load(folder / f'{rule}_{year}_distribution.npz') as arrays:
                    np.testing.assert_array_equal(arrays['origin_ns'], pd.DatetimeIndex(frame.timestamp).asi8)
                    h = 1 / (1 + np.exp(-arrays['logits'].astype(float)))
                    survival = np.cumprod(1 - h, axis=1)
                    prior = np.concatenate([np.ones((len(h), 1)), survival[:, :-1]], axis=1)
                    probability = np.concatenate([prior * h, survival[:, -1:]], axis=1)
                    np.testing.assert_allclose(probability, arrays['probability'], atol=2e-7, rtol=2e-5)
                    predicted = np.array([np.searchsorted(row.cumsum(), .5) + 1 for row in probability])
                    np.testing.assert_array_equal(predicted, frame.predicted_hours)
                metrics, selected = independent_scores(frame)
                for metric in ('row_mae', 'event_equal_mae', 'short_event_equal_mae', 'tail_event_equal_mae'):
                    np.testing.assert_allclose(metrics[metric], saved[rule][metric], atol=1e-12)
                brier = np.mean((selected.probability_recovered_by_67h - (selected.actual_hours < 68))**2)
                checked[rule] = {**metrics, 'brier_67h': float(brier),
                                 'tail_false_recovery': int(((selected.actual_hours >= 68) & (selected.predicted_hours < 68)).sum())}
                grouped = selected.groupby('event_id').error.agg(['mean', 'size']).reset_index()
                grouped['validation_year'], grouped['scoring_year'], grouped['rule'] = val_year, year, rule
                events.append(grouped)
            for key in ('timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window'):
                pd.testing.assert_series_equal(pair[0][key], pair[1][key])
            results[val_year][year] = checked
    development = [results['2022']['2023'], results['2023']['2024']]
    means = {rule: {metric: float(np.mean([f[rule][metric] for f in development]))
                   for metric in ('row_mae', 'event_equal_mae', 'tail_event_equal_mae')}
             for rule in ('legacy', 'balanced')}
    tail_events = set().union(*[set(f['legacy']['tail_events']) for f in development])
    accepted = (len(tail_events) >= 3
                and means['balanced']['row_mae'] <= means['legacy']['row_mae']
                and means['balanced']['event_equal_mae'] <= means['legacy']['event_equal_mae']
                and means['balanced']['tail_event_equal_mae'] < means['legacy']['tail_event_equal_mae'])
    assert accepted == report['development_gate']['accepted'] == report['final_ran']
    if not accepted:
        assert not (args.run / 'validation_2024').exists()
    summary = {'verification': 'PASS', 'folds': results, 'annual_development_means': means,
               'independent_tail_events': len(tail_events), 'gate_accepted': bool(accepted),
               'final_ran': report['final_ran']}
    (args.run / 'independent_verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pd.concat(events).to_csv(args.run / 'independent_event_errors.csv', index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
