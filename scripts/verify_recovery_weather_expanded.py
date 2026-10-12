"""Independent scoring and mandatory checkpoint replay for expanded weather runs."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from haze_forecast.recovery_weather_inputs import WEATHER_FEATURES
from scripts.benchmark_recovery_head import ROOT, duration_records
from scripts.benchmark_regional_air_models import hashes, write_json
from scripts.train_recovery_weather_expanded import load_sources, prepare_experiment
from scripts.train_recovery_weather_formal import (
    ARMS,
    actual_input_hashes,
    fixed_orders,
    sha,
    validation_outputs,
)
from scripts.verify_recovery_auxiliary import independent_scores, probabilities, read_json
from scripts.verify_recovery_weather_formal import tensor_digest, verify_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--replay-device', choices=('cuda', 'cpu'), default='cuda')
    parsed = parser.parse_args()
    run = ROOT / parsed.run
    report_path = run / 'report.json'
    report_hash = sha(report_path)
    report = read_json(report_path)
    assert report['status'] == 'completed_train_validation_only' and set(report['arms']) == set(ARMS)
    args = SimpleNamespace(**report['configuration'])
    for key, expected in {'history': 24, 'horizon': 72, 'validation_year': 2023, 'epochs': 12,
                          'seed': 42, 'batch_size': 256, 'learning_rate': .001, 'auxiliary_weight': .1}.items():
        assert getattr(args, key) == expected
    device = torch.device(parsed.replay_device)
    assert device.type != 'cuda' or torch.cuda.is_available()
    torch.set_num_threads(2)
    for group in ('source_code_sha256', 'reference_v1_sha256'):
        assert report[group]
        for path, digest in report[group].items():
            assert sha(ROOT / path) == digest, path
    data_hashes = actual_input_hashes(ROOT / args.data)
    assert data_hashes == report['inputs_before']
    sources, frame, field_hashes = load_sources(data_hashes)
    assert sources == report['sources'] and field_hashes == report['field_sha256']
    prepared = prepare_experiment(args, frame)
    panel, city, labels, eligible, partitions, times, weather, mask, scaler, training, validation = prepared
    assert {str(year): len(partitions[year]) for year in range(2014, 2023)} == report['training_windows_by_year']
    assert len(training) == 12337 and len(validation) == 1427 and len(times) == 386
    train_rows = np.isin(times.tz_convert('Asia/Shanghai').year, [2020, 2021, 2022])
    assert int(train_rows.sum()) == 279 and int((~train_rows).sum()) == 107
    raw_frame = frame.copy()
    raw_frame['issue_utc'] = pd.to_datetime(raw_frame.issue_utc, utc=True)
    raw_frame['valid_utc'] = pd.to_datetime(raw_frame.valid_utc, utc=True)
    groups = {issue: group.sort_values('valid_utc') for issue, group in raw_frame.groupby('issue_utc')}
    raw = np.stack([groups[issue][list(WEATHER_FEATURES)].to_numpy(dtype=float) for issue in times])
    assert raw.shape == (386, 24, 4) and np.isfinite(raw).all() and mask.all()
    train_values = raw[train_rows].reshape(-1, 4)
    independent_mean = train_values.mean(axis=0)
    independent_scale = train_values.std(axis=0)
    independent_scale = np.where(independent_scale > 0, independent_scale, 1.)
    np.testing.assert_allclose(scaler.mean, independent_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaler.scale, independent_scale, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weather, ((raw - independent_mean) / independent_scale).astype(np.float32),
                               rtol=1e-6, atol=1e-6)
    assert report['weather_scaler']['fit_years'] == [2020, 2021, 2022]
    for key in ('mean', 'scale'):
        np.testing.assert_array_equal(getattr(scaler, key), report['weather_scaler'][key])
        np.testing.assert_array_equal(getattr(panel['scaler'], key), report['history_scaler'][key])
    assert panel['nodes'] == report['nodes'] and panel['feature_names'] == report['feature_names']
    assert training.audit() == report['training_label_audit']
    assert validation.audit() == report['validation_label_audit']
    assert sha(run / 'training_targets.npz') == report['training_targets_sha256']
    with np.load(run / 'training_targets.npz', allow_pickle=False) as saved:
        for key, value in [('origin_ns', panel['timestamps'][training.origins].asi8),
                           ('target', training.hazard), ('risk_mask', training.risk_mask),
                           ('eligible', training.eligible)]:
            np.testing.assert_array_equal(saved[key], value)
    assert sha(run / 'weather_inputs.npz') == report['weather_inputs_sha256']
    with np.load(run / 'weather_inputs.npz', allow_pickle=False) as saved:
        for key, value in [('origin_ns', times.asi8), ('normalized', weather), ('mask', mask),
                           ('mean', scaler.mean), ('scale', scaler.scale), ('training_rows', train_rows)]:
            np.testing.assert_array_equal(saved[key], value)
    old = ROOT / 'artifacts/task01_recovery_head/future_weather_formal_v1'
    origins = np.concatenate([training.origins, validation.origins])
    windows = pd.DataFrame({'timestamp': panel['timestamps'][origins], 'origin': origins,
                           'partition': ['train'] * len(training) + ['validation'] * len(validation),
                           'survival_eligible': eligible[origins],
                           'future_weather_available': panel['timestamps'][origins].isin(times)})
    pd.testing.assert_frame_equal(pd.read_parquet(run / 'windows.parquet'), windows)
    assert sha(run / 'windows.parquet') == report['windows_sha256']
    identity_columns = list(windows.columns[:-1])
    pd.testing.assert_frame_equal(windows[identity_columns], pd.read_parquet(old / 'windows.parquet')[identity_columns])
    orders = np.load(run / 'batch_orders.npy', allow_pickle=False)
    assert sha(run / 'batch_orders.npy') == report['batch_orders_sha256']
    np.testing.assert_array_equal(orders, fixed_orders(len(training), 12, 42))
    np.testing.assert_array_equal(orders, np.load(old / 'batch_orders.npy', allow_pickle=False))
    initial = {arm: torch.load(run / arm / 'initial_state.pt', map_location='cpu', weights_only=True) for arm in ARMS}
    base_keys = list(initial['base'])
    results = {}
    for arm in ARMS:
        folder = run / arm
        item = report['arms'][arm]
        assert read_json(folder / 'report.json') == item
        assert tensor_digest(initial[arm]) == report['initial_state_sha256'][arm]
        assert tensor_digest(initial[arm], base_keys) == report['initial_base_state_sha256'][arm]
        old_initial = torch.load(old / arm / 'initial_state.pt', map_location='cpu', weights_only=True)
        assert list(old_initial) == list(initial[arm])
        for key in initial[arm]:
            torch.testing.assert_close(initial[arm][key], old_initial[key], rtol=0, atol=0)
        for key in base_keys:
            torch.testing.assert_close(initial[arm][key], initial['base'][key], rtol=0, atol=0)
        assert sha(folder / 'initial_state.pt') == item['initial_state_sha256']
        assert sha(folder / 'selected_state.pt') == item['checkpoint_sha256']
        assert sha(folder / 'selection.json') == item['selection_sha256']
        assert item['epochs_ran'] == 12
        selection = read_json(folder / 'selection.json')
        assert selection == item['selection']
        chosen = verify_trace(read_json(folder / 'training_trace.json'), selection, 12, len(training),
                              training.audit()['informative_eligible'])
        saved_frame = pd.read_parquet(folder / 'validation_2023_recovery.parquet')
        with np.load(folder / 'validation_2023_distribution.npz', allow_pickle=False) as saved:
            logits = saved['logits'].copy()
            assert logits.shape == (len(validation), 67) and np.isfinite(logits).all()
            prob = probabilities(logits)
            np.testing.assert_allclose(prob, saved['probability'], atol=2e-7, rtol=2e-5)
            assert np.isfinite(saved['probability']).all()
            assert ((saved['probability'] >= 0) & (saved['probability'] <= 1)).all()
            np.testing.assert_allclose(saved['probability'].sum(1), 1, atol=2e-6, rtol=2e-6)
            for key, value in [('origin_ns', panel['timestamps'][validation.origins].asi8),
                               ('target', validation.hazard), ('risk_mask', validation.risk_mask),
                               ('eligible', validation.eligible)]:
                np.testing.assert_array_equal(saved[key], value)
            with np.load(old / arm / 'validation_2023_distribution.npz', allow_pickle=False) as original:
                for key in ('origin_ns', 'target', 'risk_mask', 'eligible'):
                    np.testing.assert_array_equal(saved[key], original[key])
        predicted = np.array([min(67, np.searchsorted(row.cumsum(), .5)) + 1 for row in prob])
        np.testing.assert_array_equal(predicted, saved_frame.predicted_hours)
        np.testing.assert_allclose(1 - prob[:, -1], saved_frame.probability_recovered_by_67h, atol=2e-7)
        fresh = duration_records(validation.origins, panel, labels, city.aqi.to_numpy(), predicted, 72)
        columns = ['timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window']
        pd.testing.assert_frame_equal(saved_frame[columns], fresh[columns])
        pd.testing.assert_frame_equal(saved_frame[columns], pd.read_parquet(old / arm / 'validation_2023_recovery.parquet')[columns])
        scores = independent_scores(saved_frame)
        for key in ('row_mae', 'event_equal_mae', 'short_event_equal_mae', 'tail_event_equal_mae'):
            np.testing.assert_allclose(scores[key], item['validation_2023'][key], atol=1e-12)
            np.testing.assert_allclose(scores[key], chosen['validation_2023'][key], atol=1e-12)
        state = torch.load(folder / 'selected_state.pt', map_location='cpu', weights_only=True)
        assert all(torch.isfinite(value).all() for value in state.values())
        model = ARMS[arm](len(panel['feature_names']), panel['coords'], panel['scaler'].mean[-2:],
                          panel['scaler'].scale[-2:]).to(device)
        model.load_state_dict(state)
        model.eval()
        before = tensor_digest(model.state_dict())
        with torch.no_grad():
            replay, replay_logits, _, _ = validation_outputs(model, validation, panel, labels, city.aqi.to_numpy(), args, device)
        assert tensor_digest(model.state_dict()) == before
        np.testing.assert_allclose(replay_logits.numpy(), logits, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(replay.predicted_hours, saved_frame.predicted_hours)
        results[arm] = {'selected_epoch': selection['epoch'], 'validation_2023': scores,
                        'replay_verified': True, 'max_logit_difference': float(abs(replay_logits.numpy() - logits).max())}
        del model
    for key in initial['weather']:
        torch.testing.assert_close(initial['weather'][key], initial['masked_values'][key], rtol=0, atol=0)
    assert actual_input_hashes(ROOT / args.data) == data_hashes
    assert hashes(ROOT / 'artifacts/cnemc_2025_deployment_candidate') == report['deployment_before']
    refreshed, _, refreshed_fields = load_sources(data_hashes)
    assert refreshed == sources and refreshed_fields == field_hashes
    assert sha(report_path) == report_hash
    for group in ('source_code_sha256', 'reference_v1_sha256'):
        assert all(sha(ROOT / path) == digest for path, digest in report[group].items())
    summary = {'verification': 'PASS', 'checkpoint_replay_verified': True, 'arms': results,
               'training_windows': len(training), 'validation_windows': len(validation),
               'weather_train_origins': 279, 'weather_validation_origins': 107,
               'limitations': ['2023_selection_validation_not_independent_test',
                               'saved_schedule_not_optimizer_step_replay', 'single_seed']}
    write_json(run / 'independent_verification.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
