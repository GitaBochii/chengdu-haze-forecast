"""Verify formal artifacts, with optional selected-checkpoint forward replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from scripts.benchmark_recovery_head import ROOT, duration_records, load_experiment
from scripts.benchmark_regional_air_models import hashes
from scripts.train_recovery_weather_formal import (
    ARMS,
    FormalWeatherWindows,
    actual_input_hashes,
    collected_frame,
    fit_weather,
    fixed_orders,
    load_ready_manifest,
    sha,
    validation_outputs,
)
from scripts.verify_recovery_auxiliary import independent_scores, probabilities, read_json


def verify_trace(trace, selection, epochs, windows, informative):
    assert len(trace) == epochs and [r['epoch'] for r in trace] == list(range(1, epochs + 1))
    for row in trace:
        assert row['training_windows'] == windows
        assert row['informative_survival_origins'] == informative
        assert row['concentration_elements'] > 0
        assert np.isfinite(row['training_survival_nll']) and np.isfinite(row['training_concentration_l1'])
        key = (row['validation_2023']['event_equal_mae'], row['validation_2023']['row_mae'])
        assert np.isfinite(key).all()
        np.testing.assert_array_equal(key, row['selection_key'])
    selected = min(trace, key=lambda r: (r['validation_2023']['event_equal_mae'], r['validation_2023']['row_mae']))
    assert selection['epoch'] == selected['epoch']
    assert selection['candidate_epochs'] == list(range(1, epochs + 1))
    assert selection['validation_year'] == 2023
    assert selection['rule'] == 'event_equal_mae_then_row_mae_earliest_tie'
    np.testing.assert_array_equal(selection['validation_key'], selected['selection_key'])
    return selected


def tensor_digest(state, names=None):
    digest = hashlib.sha256()
    for name in state if names is None else names:
        value = state[name].detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}:'.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--replay-device', choices=('cuda', 'cpu'),
                        help='Replay each selected checkpoint on validation inputs without optimizer updates')
    parsed = parser.parse_args()
    run = ROOT / parsed.run
    report = read_json(run / 'report.json')
    assert report['status'] == 'completed_train_validation_only' and set(report['arms']) == set(ARMS)
    args = SimpleNamespace(**report['configuration'])
    replay_device = torch.device(parsed.replay_device) if parsed.replay_device else None
    if replay_device is not None:
        assert replay_device.type != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable for replay'
        torch.set_num_threads(2)
    for key, expected in {'history': 24, 'horizon': 72, 'validation_year': 2023, 'epochs': 12,
                          'seed': 42, 'batch_size': 256, 'learning_rate': .001, 'auxiliary_weight': .1}.items():
        assert getattr(args, key) == expected
    data_hashes = actual_input_hashes(ROOT / args.data)
    assert data_hashes == report['inputs_before']
    gate_path = ROOT / args.gate
    assert sha(gate_path) == report['gate_sha256']
    gate, plan, records = load_ready_manifest(gate_path, ROOT / args.manifest, ROOT / args.collection / 'results', data_hashes)
    assert gate['manifest_sha256'] == report['manifest_sha256'] and gate['results_sha256'] == report['results_sha256']
    frame, fields = collected_frame(plan, records)
    assert fields == report['field_sha256']
    assert hashes(ROOT / 'artifacts/cnemc_2025_deployment_candidate') == report['deployment_before']
    times = pd.DatetimeIndex(pd.to_datetime(plan.issue_utc.unique(), utc=True)).sort_values()
    values, mask, scaler = fit_weather(frame, times)
    np.testing.assert_array_equal(scaler.mean, report['weather_scaler']['mean'])
    np.testing.assert_array_equal(scaler.scale, report['weather_scaler']['scale'])
    assert report['weather_scaler']['fit_year'] == 2022
    assert sha(run / 'weather_inputs.npz') == report['weather_inputs_sha256']
    with np.load(run / 'weather_inputs.npz', allow_pickle=False) as saved:
        for key, expected in (('origin_ns', times.asi8), ('normalized', values), ('mask', mask),
                              ('mean', scaler.mean), ('scale', scaler.scale),
                              ('training_rows', times.tz_convert('Asia/Shanghai').year == 2022)):
            np.testing.assert_array_equal(saved[key], expected)
    panel, city, labels, eligible, partitions = load_experiment(args)
    np.testing.assert_array_equal(panel['scaler'].mean, report['history_scaler']['mean'])
    np.testing.assert_array_equal(panel['scaler'].scale, report['history_scaler']['scale'])
    assert panel['nodes'] == report['nodes'] and panel['feature_names'] == report['feature_names']
    train_origins = np.concatenate([partitions[y] for y in range(2014, 2023)])
    assert panel['timestamps'][train_origins.max() + 72] < pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    assert {str(y): len(partitions[y]) for y in range(2014, 2023)} == report['training_windows_by_year']
    weather_origins = panel['timestamps'].get_indexer(times)
    for year in (2022, 2023):
        np.testing.assert_array_equal(weather_origins[times.tz_convert('Asia/Shanghai').year == year],
                                      partitions[year][eligible[partitions[year]]])
    mapping = {int(origin): (torch.from_numpy(values[i]), torch.from_numpy(mask[i])) for i, origin in enumerate(weather_origins)}
    training = FormalWeatherWindows(panel, train_origins, city.aqi.to_numpy(), eligible, mapping)
    validation = FormalWeatherWindows(panel, partitions[2023], city.aqi.to_numpy(), eligible, mapping)
    assert training.audit() == report['training_label_audit'] and validation.audit() == report['validation_label_audit']
    all_origins = np.concatenate([train_origins, partitions[2023]])
    expected_windows = pd.DataFrame({'timestamp': panel['timestamps'][all_origins], 'origin': all_origins,
                                    'partition': ['train'] * len(training) + ['validation'] * len(validation),
                                    'survival_eligible': eligible[all_origins],
                                    'future_weather_available': np.isin(all_origins, weather_origins)})
    pd.testing.assert_frame_equal(pd.read_parquet(run / 'windows.parquet'), expected_windows)
    orders = np.load(run / 'batch_orders.npy', allow_pickle=False)
    assert sha(run / 'batch_orders.npy') == report['batch_orders_sha256']
    np.testing.assert_array_equal(orders, fixed_orders(len(training), 12, 42))
    for order in orders:
        np.testing.assert_array_equal(np.sort(order), np.arange(len(training)))
    initial = {arm: torch.load(run / arm / 'initial_state.pt', map_location='cpu', weights_only=True) for arm in ARMS}
    base_keys = list(initial['base'])
    for arm in ARMS:
        assert tensor_digest(initial[arm]) == report['initial_state_sha256'][arm]
        assert tensor_digest(initial[arm], base_keys) == report['initial_base_state_sha256'][arm]
        for name in base_keys:
            torch.testing.assert_close(initial[arm][name], initial['base'][name], rtol=0, atol=0)
    assert list(initial['masked_values']) == list(initial['weather'])
    for name in initial['weather']:
        torch.testing.assert_close(initial['masked_values'][name], initial['weather'][name], rtol=0, atol=0)
    results, identities = {}, []
    for arm, arm_report in report['arms'].items():
        folder = run / arm
        assert read_json(folder / 'report.json') == arm_report
        assert sha(folder / 'initial_state.pt') == arm_report['initial_state_sha256']
        assert sha(folder / 'selected_state.pt') == arm_report['checkpoint_sha256']
        assert sha(folder / 'selection.json') == arm_report['selection_sha256']
        selected_state = torch.load(folder / 'selected_state.pt', map_location='cpu', weights_only=True)
        assert list(selected_state) == list(initial[arm])
        assert all(torch.isfinite(value).all() for value in selected_state.values())
        assert all(value.shape == initial[arm][key].shape for key, value in selected_state.items())
        assert arm_report['epochs_ran'] == 12
        selection = read_json(folder / 'selection.json')
        assert selection == arm_report['selection']
        chosen = verify_trace(read_json(folder / 'training_trace.json'), selection, 12, len(training),
                              training.audit()['informative_eligible'])
        saved_frame = pd.read_parquet(folder / 'validation_2023_recovery.parquet')
        with np.load(folder / 'validation_2023_distribution.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['origin_ns'], panel['timestamps'][validation.origins].asi8)
            for key, expected in (('target', validation.hazard), ('risk_mask', validation.risk_mask), ('eligible', validation.eligible)):
                np.testing.assert_array_equal(saved[key], expected)
            assert saved['logits'].shape == (len(validation), 67)
            assert np.isfinite(saved['logits']).all() and np.isfinite(saved['probability']).all()
            prob = probabilities(saved['logits'])
            assert ((saved['probability'] >= 0) & (saved['probability'] <= 1)).all()
            np.testing.assert_allclose(saved['probability'].sum(1), 1., atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(prob, saved['probability'], atol=2e-7, rtol=2e-5)
            predicted = np.array([min(67, np.searchsorted(row.cumsum(), .5)) + 1 for row in prob])
            np.testing.assert_array_equal(predicted, saved_frame.predicted_hours)
            np.testing.assert_allclose(1 - prob[:, -1], saved_frame.probability_recovered_by_67h, atol=2e-7)
            saved_logits = saved['logits'].copy()
        fresh = duration_records(validation.origins, panel, labels, city.aqi.to_numpy(), predicted, 72)
        identity = ['timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window']
        pd.testing.assert_frame_equal(saved_frame[identity], fresh[identity])
        identities.append(saved_frame[identity])
        scores = independent_scores(saved_frame)
        for key in ('row_mae', 'event_equal_mae', 'short_event_equal_mae', 'tail_event_equal_mae'):
            np.testing.assert_allclose(scores[key], arm_report['validation_2023'][key], atol=1e-12)
            np.testing.assert_allclose(scores[key], chosen['validation_2023'][key], atol=1e-12)
        assert scores['tail_events'] == arm_report['validation_2023']['tail_events']
        results[arm] = {'selected_epoch': selection['epoch'], 'validation_2023': scores}
        if replay_device is not None:
            model = ARMS[arm](len(panel['feature_names']), panel['coords'],
                              panel['scaler'].mean[-2:], panel['scaler'].scale[-2:]).to(replay_device)
            model.load_state_dict(selected_state)
            model.eval()
            before_replay = tensor_digest(model.state_dict())
            with torch.no_grad():
                replay_frame, replay_logits, _, _ = validation_outputs(
                    model, validation, panel, labels, city.aqi.to_numpy(), args, replay_device)
            assert tensor_digest(model.state_dict()) == before_replay
            np.testing.assert_allclose(replay_logits.numpy(), saved_logits, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(replay_frame.predicted_hours, saved_frame.predicted_hours)
            pd.testing.assert_frame_equal(replay_frame[identity], saved_frame[identity])
            results[arm]['replay_verified'] = True
            results[arm]['replay_max_abs_logit_difference'] = float(np.max(abs(replay_logits.numpy() - saved_logits)))
            del model
        else:
            results[arm]['replay_verified'] = False
    for identity in identities[1:]:
        pd.testing.assert_frame_equal(identity, identities[0])
    assert sha(gate_path) == report['gate_sha256'] and actual_input_hashes(ROOT / args.data) == data_hashes
    assert hashes(ROOT / 'artifacts/cnemc_2025_deployment_candidate') == report['deployment_before']
    assert sha(ROOT / args.manifest) == report['manifest_sha256']
    result_folder = ROOT / args.collection / 'results'
    assert {p.name for p in result_folder.glob('*.json')} == set(report['results_sha256'])
    assert all(sha(result_folder / name) == digest for name, digest in report['results_sha256'].items())
    assert all(sha(ROOT / name) == digest for name, digest in fields.items())
    coverage = expected_windows.groupby('partition').future_weather_available.agg(['sum', 'count'])
    coverage_report = {partition: {'windows': int(row['count']), 'with_weather': int(row['sum']),
                                    'without_weather': int(row['count'] - row['sum']),
                                    'missing_weather_fraction': float(1 - row['sum'] / row['count'])}
                       for partition, row in coverage.iterrows()}
    limitations = ['2023_is_selection_validation_not_independent_test',
                   'saved_batch_schedule_verified_not_optimizer_step_replay']
    if replay_device is None:
        limitations.append('artifact_verification_without_checkpoint_forward_replay')
    summary = {'verification': 'PASS', 'arms': results, 'windows': coverage_report,
               'shared_initial_weights_identical': True, 'fixed_epoch_orders_verified': True,
               'replay_device': parsed.replay_device, 'checkpoint_replay_verified': replay_device is not None,
               'limitations': limitations}
    (run / 'independent_verification.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
