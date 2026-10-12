"""Independently verify the two development folds of the auxiliary ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.benchmark_recovery_head import RecoveryWindows, duration_records, load_experiment

METRICS = ('row_mae', 'event_equal_mae', 'tail_event_equal_mae')
ARMS = ('auxiliary_0_1', 'auxiliary_0')


def read_json(path):
    return json.loads(path.read_text('utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(folder, mapping):
    assert mapping, f'empty hash manifest: {folder}'
    for name, expected in mapping.items():
        assert sha(folder / name) == expected, f'hash mismatch: {folder / name}'


def probabilities(logits):
    assert logits.ndim == 2 and np.isfinite(logits).all()
    z = logits.astype(np.float64)
    log_h = -np.logaddexp(0., -z)
    log_s = np.cumsum(-np.logaddexp(0., z), axis=1)
    previous = np.concatenate([np.zeros((len(z), 1)), log_s[:, :-1]], axis=1)
    return np.concatenate([np.exp(previous + log_h), np.exp(log_s[:, -1:])], axis=1)


def independent_scores(frame):
    selected = frame.loc[np.isfinite(frame.actual_hours) & frame.label_verified_in_scoring_window].copy()
    assert len(selected) and np.isfinite(selected.predicted_hours).all()
    assert selected.event_id.notna().all()
    selected['error'] = abs(np.minimum(selected.actual_hours, 68) - np.minimum(selected.predicted_hours, 68))
    short = selected.loc[selected.actual_hours < 68]
    tail = selected.loc[selected.actual_hours >= 68]
    return {'row_mae': float(selected.error.mean()),
            'event_equal_mae': float(selected.groupby('event_id').error.mean().mean()),
            'short_event_equal_mae': float(short.groupby('event_id').error.mean().mean()) if len(short) else None,
            'tail_event_equal_mae': float(tail.groupby('event_id').error.mean().mean()) if len(tail) else None,
            'rows': len(selected), 'events': int(selected.event_id.nunique()),
            'tail_rows': len(tail), 'tail_events': [int(x) for x in sorted(tail.event_id.unique())]}


def independent_gate(folds):
    for fold in folds:
        assert fold[ARMS[0]]['tail_events'] == fold[ARMS[1]]['tail_events']
    supported = len(folds) == 2 and all(f[ARMS[0]]['tail_events'] for f in folds)
    events = set().union(*(set(f[ARMS[0]]['tail_events']) for f in folds))
    finite = all(f[arm][metric] is not None and np.isfinite(f[arm][metric])
                 for f in folds for arm in ARMS for metric in METRICS)
    means = {arm: {metric: float(np.mean([f[arm][metric] for f in folds]))
                   if finite else None for metric in METRICS} for arm in ARMS}
    old, new = means[ARMS[0]], means[ARMS[1]]
    accepted = bool(supported and len(events) >= 3 and finite
                    and new['row_mae'] <= old['row_mae']
                    and new['event_equal_mae'] <= old['event_equal_mae']
                    and new['tail_event_equal_mae'] < old['tail_event_equal_mae'])
    return {'accepted': accepted, 'annual_means': means, 'independent_tail_events': len(events)}


def check_selection(folder, fold):
    trace = read_json(folder / 'training_trace.json')
    assert len(trace) == fold['epochs_ran'] == fold['configuration']['epochs'] == 12
    assert [x['epoch'] for x in trace] == list(range(1, 13))
    for row in trace:
        key = [row['validation']['event_equal_mae'], row['validation']['row_mae']]
        assert np.isfinite(key).all()
        np.testing.assert_array_equal(key, row['selection_keys']['legacy'])
    chosen = min(trace, key=lambda x: (x['validation']['event_equal_mae'], x['validation']['row_mae']))
    selection = read_json(folder / 'legacy_selection.json')
    assert selection == fold['selections']['legacy']
    assert selection['epoch'] == chosen['epoch']
    assert selection['candidate_epochs'] == list(range(1, 13))
    np.testing.assert_array_equal(selection['validation_key'], chosen['selection_keys']['legacy'])
    assert read_json(folder / 'frozen_before_evaluation.json') == fold['frozen']
    check_hashes(folder, fold['frozen'])


def check_predictions(folder, year, fold, panel, city, labels, eligible, partitions):
    frame = pd.read_parquet(folder / f'legacy_{year}_recovery.parquet')
    assert not frame.timestamp.duplicated().any()
    origins = partitions[year]
    np.testing.assert_array_equal(pd.DatetimeIndex(frame.timestamp).asi8, panel['timestamps'][origins].asi8)
    dataset = RecoveryWindows(panel, origins, 24, 72, city.aqi.to_numpy(), eligible)
    with np.load(folder / f'legacy_{year}_distribution.npz') as arrays:
        np.testing.assert_array_equal(arrays['origin_ns'], panel['timestamps'][origins].asi8)
        for key, value in (('target', dataset.hazard), ('risk_mask', dataset.risk_mask), ('eligible', dataset.eligible)):
            np.testing.assert_array_equal(arrays[key], value)
        assert arrays['logits'].shape == (len(origins), 67)
        prob = probabilities(arrays['logits'])
        np.testing.assert_allclose(prob, arrays['probability'], atol=2e-7, rtol=2e-5)
        np.testing.assert_allclose(prob.sum(axis=1), 1., atol=1e-12)
        medians = np.array([min(67, np.searchsorted(row.cumsum(), .5)) + 1 for row in prob])
        np.testing.assert_array_equal(medians, frame.predicted_hours)
        np.testing.assert_allclose(1 - prob[:, -1], frame.probability_recovered_by_67h, atol=2e-7)
    fresh = duration_records(origins, panel, labels, city.aqi.to_numpy(), medians, 72)
    identity = ['timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window']
    pd.testing.assert_frame_equal(frame[identity], fresh[identity])
    scores = independent_scores(frame)
    saved = fold['years'][str(year)]['legacy']
    for metric in (*METRICS, 'short_event_equal_mae'):
        np.testing.assert_allclose(scores[metric], saved[metric], atol=1e-12)
    assert scores['tail_events'] == saved['tail_events']
    return frame[identity], scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root / args.run
    report = read_json(run / 'report.json')
    assert report['decision'] not in ('running', 'error')
    assert set(report['folds']) == {'2022', '2023'}
    assert not (run / 'validation_2024').exists()
    control = root / report['control']
    check_hashes(control, report['control_sha256'])
    check_hashes(root / report['configuration']['data'], report['inputs_before'])
    check_hashes(root / 'artifacts/cnemc_2025_deployment_candidate', report['deployment_before'])
    results = {}
    for validation_year in (2022, 2023):
        year = validation_year + 1
        candidate = report['folds'][str(validation_year)]
        base_folder = control / f'validation_{validation_year}'
        baseline = read_json(base_folder / 'report.json')
        candidate_folder = run / f'validation_{validation_year}'
        assert read_json(candidate_folder / 'report.json') == candidate
        assert candidate['configuration']['auxiliary_weight'] == 0
        assert baseline['configuration']['auxiliary_weight'] == .1
        for key in ('seed', 'learning_rate', 'batch_size', 'epochs', 'data', 'history', 'horizon', 'validation_year'):
            assert baseline['configuration'][key] == candidate['configuration'][key], key
        assert candidate['validation_year'] == validation_year
        assert candidate['test_years'] == baseline['test_years'] == [year]
        for key in ('nodes', 'feature_names', 'coordinates_lon_lat', 'scaler', 'training_windows_by_year',
                    'partitions', 'training_label_audit', 'validation_label_audit'):
            assert candidate[key] == baseline[key], key
        panel, city, labels, eligible, partitions = load_experiment(SimpleNamespace(**candidate['configuration']))
        np.testing.assert_array_equal(panel['scaler'].mean, candidate['scaler']['mean'])
        np.testing.assert_array_equal(panel['scaler'].scale, candidate['scaler']['scale'])
        assert panel['nodes'] == candidate['nodes']
        train_origins = np.concatenate([partitions[y] for y in range(2014, validation_year)])
        assert panel['timestamps'][train_origins.max() + 72] < pd.Timestamp(f'{validation_year}-01-01', tz='Asia/Shanghai')
        identities, result = [], {}
        for arm, folder, fold in ((ARMS[0], base_folder, baseline), (ARMS[1], candidate_folder, candidate)):
            check_selection(folder, fold)
            identity, scores = check_predictions(folder, year, fold, panel, city, labels, eligible, partitions)
            identities.append(identity)
            result[arm] = scores
        pd.testing.assert_frame_equal(*identities)
        results[str(validation_year)] = result
    gate = independent_gate(list(results.values()))
    assert gate['accepted'] == report['gate']['accepted']
    for arm in ARMS:
        for metric in METRICS:
            np.testing.assert_allclose(gate['annual_means'][arm][metric], report['gate']['annual_means'][arm][metric], atol=1e-12)
    check_hashes(control, report['control_sha256'])
    summary = {'verification': 'PASS', 'folds': results, 'gate': gate,
               'limitations': ['checkpoint_probability_recomputed_without_model_forward',
                               'same_initialization_verified_by_seed_and_configuration_not_saved_initial_weights'],
               'inputs_unchanged': True, 'control_unchanged': True, 'deployment_unchanged': True}
    (run / 'independent_verification.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'verification': 'PASS', 'gate': gate}, indent=2))


if __name__ == '__main__':
    main()
