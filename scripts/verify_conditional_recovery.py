"""Independently replay fixed-tail conditional recovery development checkpoints."""

import hashlib
import json

import numpy as np
import pandas as pd
import torch

from haze_forecast.history_state_residual import HistoryStateResidual
from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.verify_recovery_auxiliary import independent_scores, probabilities
from scripts.verify_recovery_weather_formal import tensor_digest

ARMS = {'linear_all': False, 'nonlinear_all': True}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(mapping):
    assert mapping, 'empty source manifest'
    for path, expected in mapping.items():
        assert sha(ROOT / path) == expected, f'source changed: {path}'


def expected_batches(pool):
    generator = torch.Generator().manual_seed(42)
    permutations = [torch.randperm(len(pool), generator=generator).numpy()
                    for _ in range((32000 + len(pool) - 1) // len(pool))]
    return pool[np.concatenate(permutations)[:32000]].reshape(500, 64)


def compare_scores(actual, saved):
    for key, value in actual.items():
        if key.endswith('mae'):
            np.testing.assert_allclose(value, saved[key], rtol=0, atol=1e-12)
        else:
            assert value == saved[key], key


def logsumexp(values):
    maximum = values.max(axis=1, keepdims=True)
    return maximum + np.log(np.exp(values - maximum).sum(axis=1, keepdims=True))


def conditional_log_mass(base_logits, correction):
    assert base_logits.shape == correction.shape and base_logits.shape[1] == 67
    assert np.isfinite(base_logits).all() and np.isfinite(correction).all()
    z, delta = base_logits.astype(np.float64), correction.astype(np.float64)
    log_survival = np.cumsum(-np.logaddexp(0., z), axis=1)
    prior = np.concatenate([np.zeros((len(z), 1)), log_survival[:, :-1]], axis=1)
    log_event = -np.logaddexp(0., -z) + prior
    shifted = log_event + delta
    corrected = shifted - logsumexp(shifted) + logsumexp(log_event)
    return np.concatenate([corrected, log_survival[:, -1:]], axis=1)


def score_log_mass(log_mass, identity):
    assert log_mass.shape == (len(identity), 68) and np.isfinite(log_mass).all()
    probability = np.exp(log_mass)
    np.testing.assert_allclose(probability.sum(1), 1., rtol=0, atol=1e-12)
    median = (probability.cumsum(1) < .5).sum(1).clip(max=67) + 1
    frame = identity.copy()
    frame['predicted_hours'] = median
    return median, probability, independent_scores(frame)


def main():
    if not __debug__:
        raise RuntimeError('verification requires assertions enabled')
    run = ROOT / 'artifacts/task01_recovery_head/conditional_recovery_v1'
    cache = ROOT / 'artifacts/task01_recovery_head/frozen_weather_states_v1'
    report = json.loads((run / 'report.json').read_bytes())
    assert report['status'] == 'completed_development_comparison'
    assert set(report['arms']) == set(ARMS)
    expected_config = {'seed': 42, 'updates': 500, 'batch_size': 64, 'validation_every_updates': 5,
                       'learning_rate': .001, 'gradient_clip_norm': 1., 'loss': 'observed_prefix_nll_fixed_base_tail',
                       'selection': 'event_equal_then_row_earliest_step_including_zero',
                       'frozen_base': True, 'fixed_base_tail': True, 'input': 'historical_state_128_only',
                       'pool_sizes': {'covered': 275, 'all': 1517},
                       'stream': 'seed42_torch_randperm_cycles_concatenated_then_truncated_32000',
                       'cycle_boundary_batch_may_repeat_origins': True}
    assert report['configuration'] == expected_config
    protocol = json.loads((run / 'protocol.json').read_bytes())
    assert protocol['status'] == 'running' and protocol['arms'] == {}
    for key in ('configuration', 'source_sha256', 'baseline', 'limitations'):
        assert protocol[key] == report[key], key
    required_sources = {
        'src/haze_forecast/conditional_recovery.py',
        'src/haze_forecast/history_state_residual.py',
        'scripts/train_conditional_recovery.py',
        'scripts/verify_conditional_recovery.py',
        'conditional_recovery_design_20260906.md',
    }
    assert required_sources <= set(report['source_sha256'])
    check_hashes(report['source_sha256'])
    cache_report = json.loads((cache / 'report.json').read_bytes())
    assert cache_report['status'] == 'frozen_backbone_cache_verified'
    check_hashes(cache_report['source_sha256'])
    assert sha(cache / 'states.npz') == cache_report['state_cache_sha256']
    assert sha(cache / 'validation_identity.parquet') == cache_report['validation_identity_sha256']
    with np.load(cache / 'states.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    informative = arrays['train_eligible'] & arrays['train_risk'].any(1)
    pools = {'all': np.flatnonzero(informative),
             'covered': np.flatnonzero(informative & arrays['train_mask'].any((1, 2)))}
    assert len(informative) == 12337 and len(pools['all']) == 1517 and len(pools['covered']) == 275
    assert len(np.setdiff1d(pools['all'], pools['covered'])) == 1242
    for split, count in (('train', 12337), ('validation', 1427)):
        state, logits = arrays[split + '_state'], arrays[split + '_logits']
        risk, target = arrays[split + '_risk'], arrays[split + '_target']
        assert state.shape == (count, 128) and logits.shape == risk.shape == target.shape == (count, 67)
        assert np.isfinite(state).all() and np.isfinite(logits).all()
        assert risk.dtype == np.bool_ and arrays[split + '_eligible'].dtype == np.bool_
        assert np.isin(target, [0, 1]).all() and (target.sum(1) <= 1).all()
        assert not (target.astype(bool) & ~risk).any()
        assert not ((~risk[:, :-1]) & risk[:, 1:]).any()
        event_rows = np.flatnonzero(target.sum(1))
        np.testing.assert_array_equal(target[event_rows].argmax(1), risk[event_rows].sum(1) - 1)
        times = pd.to_datetime(arrays[split + '_origin_ns'], utc=True).tz_convert('Asia/Shanghai')
        assert not times.duplicated().any()
        if split == 'train':
            assert ((times.year >= 2014) & (times.year <= 2022)).all()
            assert times.max() + pd.Timedelta(hours=72) < pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
        else:
            assert (times.year == 2023).all()
    identity = pd.read_parquet(cache / 'validation_identity.parquet')
    np.testing.assert_array_equal(pd.DatetimeIndex(identity.timestamp).asi8, arrays['validation_origin_ns'])
    baseline_log_mass = conditional_log_mass(arrays['validation_logits'], np.zeros_like(arrays['validation_logits']))
    np.testing.assert_allclose(np.exp(baseline_log_mass), probabilities(arrays['validation_logits']), rtol=1e-12, atol=1e-14)
    _, _, baseline = score_log_mass(baseline_log_mass, identity)
    compare_scores(baseline, report['baseline'])
    batches_by_pool = {name: expected_batches(pool) for name, pool in pools.items()}
    results = {}
    query = torch.from_numpy(arrays['validation_state']).cuda()
    train_state = torch.from_numpy(arrays['train_state']).cuda()
    for arm, nonlinear in ARMS.items():
        pool_name = 'all'
        folder, item = run / arm, report['arms'][arm]
        assert json.loads((folder / 'report.json').read_bytes()) == item
        assert item['updates_ran'] == 500
        pool = np.load(folder / 'pool_indices.npy', allow_pickle=False)
        batches = np.load(folder / 'training_batches.npy', allow_pickle=False)
        np.testing.assert_array_equal(pool, pools[pool_name])
        np.testing.assert_array_equal(batches, batches_by_pool[pool_name])
        assert sha(folder / 'pool_indices.npy') == item['pool_sha256']
        assert sha(folder / 'training_batches.npy') == item['batches_sha256']
        assert batches.shape == (500, 64) and np.isin(batches, pool).all()
        counts = np.bincount(batches.ravel(), minlength=12337)[pool]
        assert item['exposures'] == {'total': 32000, 'minimum_per_origin': int(counts.min()),
                                     'maximum_per_origin': int(counts.max()), 'unique_origins': len(pool)}
        training_trace = json.loads((folder / 'training_trace.json').read_bytes())
        assert [row['step'] for row in training_trace] == list(range(1, 501))
        assert [row['cumulative_exposures'] for row in training_trace] == list(range(64, 32001, 64))
        assert all(np.isfinite(row['training_nll']) and row['training_nll'] >= 0 for row in training_trace)
        trace = json.loads((folder / 'trace.json').read_bytes())
        assert [row['step'] for row in trace] == list(range(5, 501, 5))
        candidates = [(0, report['baseline'])] + [(row['step'], row['validation_2023']) for row in trace]
        assert all(np.isfinite([metrics['event_equal_mae'], metrics['row_mae']]).all()
                   for _, metrics in candidates)
        chosen = min(candidates, key=lambda row: (row[1]['event_equal_mae'], row[1]['row_mae'], row[0]))
        assert item['selected_step'] == chosen[0] and item['metrics'] == chosen[1]
        model = HistoryStateResidual(nonlinear=nonlinear).cuda().eval()
        initial = torch.load(folder / 'initial_state.pt', weights_only=True)
        initial_digest = tensor_digest(initial)
        assert initial_digest == item['initial_state_digest']
        model.load_state_dict(initial)
        with torch.no_grad():
            assert torch.count_nonzero(model(query)) == 0
            assert torch.count_nonzero(model(train_state)) == 0
        assert sha(folder / 'selected_state.pt') == item['checkpoint_sha256']
        model.load_state_dict(torch.load(folder / 'selected_state.pt', weights_only=True))
        before = tensor_digest(model.state_dict())
        with torch.no_grad():
            correction = model(query).cpu().numpy()
        assert tensor_digest(model.state_dict()) == before
        if item['selected_step'] == 0:
            assert before == initial_digest and np.count_nonzero(correction) == 0
        log_mass = conditional_log_mass(arrays['validation_logits'], correction)
        saved_log_mass = np.load(folder / 'log_probability.npy', allow_pickle=False)
        np.testing.assert_allclose(log_mass, saved_log_mass, rtol=1e-5, atol=2e-6)
        np.testing.assert_allclose(log_mass[:, -1], baseline_log_mass[:, -1], rtol=0, atol=0)
        median, probability, scores = score_log_mass(log_mass, identity)
        saved_probability = np.exp(saved_log_mass.astype(np.float64))
        np.testing.assert_allclose(saved_probability.sum(1), 1., rtol=0, atol=2e-6)
        np.testing.assert_array_equal((saved_probability.cumsum(1) < .5).sum(1).clip(max=67) + 1, median)
        np.testing.assert_allclose(saved_probability[:, -1], np.exp(baseline_log_mass[:, -1]),
                                   rtol=1e-6, atol=2e-7)
        frame = pd.read_parquet(folder / 'validation.parquet')
        cols = ['timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window']
        pd.testing.assert_frame_equal(frame[cols], identity[cols])
        np.testing.assert_array_equal(frame.predicted_hours, median)
        np.testing.assert_allclose(frame.probability_recovered_by_67h, 1 - probability[:, -1], atol=2e-7)
        compare_scores(scores, item['metrics'])
        results[arm] = {'selected_step': item['selected_step'], 'metrics': scores,
                        'training_samples_exposed': int(batches.size),
                        'distinct_training_origins': int(np.unique(batches).size),
                        'replay_max_difference': float(abs(log_mass - saved_log_mass).max()),
                        'fixed_tail_probability_max_difference': float(abs(
                            saved_probability[:, -1] - np.exp(baseline_log_mass[:, -1])).max()),
                        'development_gate': bool(scores['row_mae'] < baseline['row_mae']
                                                 and scores['event_equal_mae'] < baseline['event_equal_mae']
                                                 and scores['tail_event_equal_mae'] <= baseline['tail_event_equal_mae'])}
    check_hashes(report['source_sha256'])
    check_hashes(cache_report['source_sha256'])
    result = {'verification': 'PASS', 'baseline': baseline, 'arms': results,
              'limitations': ['cached_backbone_reference', 'development_validation_only',
                              'no_optimizer_step_replay', 'equal_updates_not_equal_per_origin_exposure']}
    write_json(run / 'independent_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
