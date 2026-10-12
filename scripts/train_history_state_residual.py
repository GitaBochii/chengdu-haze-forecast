"""Fixed-update historical-state architecture and training-pool comparison."""

import json

import numpy as np
import pandas as pd
import torch

from haze_forecast.history_state_residual import HistoryStateResidual
from haze_forecast.recovery_survival import masked_survival_loss
from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.train_frozen_weather_residual import score
from scripts.train_recovery_weather_expanded import hashes_unchanged
from scripts.train_recovery_weather_formal import sha
from scripts.verify_recovery_auxiliary import independent_scores, probabilities
from scripts.verify_recovery_weather_formal import tensor_digest

ARMS = {'linear_covered': (False, 'covered'), 'linear_all': (False, 'all'),
        'nonlinear_covered': (True, 'covered'), 'nonlinear_all': (True, 'all')}


def shuffled_stream(indices, updates=500, batch_size=64, seed=42):
    indices = np.asarray(indices)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
            or (indices < 0).any() or len(np.unique(indices)) != len(indices)
            or updates < 1 or batch_size < 1):
        raise ValueError('expected unique nonnegative integer pool and positive update budget')
    generator = torch.Generator().manual_seed(seed)
    count = updates * batch_size
    permutations = [torch.randperm(len(indices), generator=generator).numpy()
                    for _ in range((count + len(indices) - 1) // len(indices))]
    return indices[np.concatenate(permutations)[:count]].reshape(updates, batch_size)


def training_pools(arrays):
    state, logits, target, risk, eligible, mask, origin = [arrays['train_' + key] for key in
        ('state', 'logits', 'target', 'risk', 'eligible', 'mask', 'origin_ns')]
    n = len(state)
    if (state.shape != (n, 128) or logits.shape != (n, 67) or target.shape != (n, 67)
            or risk.shape != (n, 67) or eligible.shape != (n,) or origin.shape != (n,)
            or mask.ndim != 3 or len(mask) != n or risk.dtype != bool or eligible.dtype != bool
            or mask.dtype != bool):
        raise ValueError('unaligned cache arrays')
    years = pd.to_datetime(origin, utc=True).tz_convert('Asia/Shanghai').year
    if (not ((years >= 2014) & (years <= 2022)).all()
            or not all(np.isfinite(item).all() for item in (state, logits, target))
            or not np.isin(target, (0, 1)).all() or (target.sum(1) > 1).any()
            or (target.astype(bool) & ~risk).any() or (risk[:, 1:] & ~risk[:, :-1]).any()):
        raise ValueError('invalid years, finite values or observed-prefix labels')
    events = target.sum(1) == 1
    if not np.array_equal(target[events].argmax(1), risk[events].sum(1) - 1):
        raise ValueError('event must be the final observed risk position')
    informative = eligible & risk.any(1)
    return {'covered': np.flatnonzero(informative & mask.any(axis=(1, 2))),
            'all': np.flatnonzero(informative)}


def main():
    cache = ROOT / 'artifacts/task01_recovery_head/frozen_weather_states_v1'
    output = ROOT / 'artifacts/task01_recovery_head/history_state_residual_v1'
    assert not output.exists() and torch.cuda.is_available()
    cache_report = json.loads((cache / 'report.json').read_bytes())
    assert cache_report['status'] == 'frozen_backbone_cache_verified'
    assert hashes_unchanged(cache_report['source_sha256'])
    assert sha(cache / 'states.npz') == cache_report['state_cache_sha256']
    assert sha(cache / 'validation_identity.parquet') == cache_report['validation_identity_sha256']
    tracked = dict(cache_report['source_sha256'])
    for path in (cache / 'states.npz', cache / 'report.json', cache / 'validation_identity.parquet',
                 ROOT / 'src/haze_forecast/history_state_residual.py',
                 ROOT / 'src/haze_forecast/recovery_survival.py',
                 ROOT / 'src/haze_forecast/recovery_selection.py',
                 ROOT / 'scripts/train_frozen_weather_residual.py',
                 ROOT / 'scripts/train_history_state_residual.py',
                 ROOT / 'scripts/verify_history_state_residual.py',
                 ROOT / 'history_state_residual_design_20260906.md'):
        tracked[str(path.relative_to(ROOT))] = sha(path)
    torch.set_num_threads(2)
    with np.load(cache / 'states.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    pools = training_pools(arrays)
    assert len(arrays['train_state']) == 12337
    assert len(pools['covered']) == 275 and len(pools['all']) == 1517
    train = {key: torch.from_numpy(arrays['train_' + key]).cuda()
             for key in ('state', 'logits', 'target', 'risk', 'eligible')}
    validation = {key: torch.from_numpy(arrays['validation_' + key]).cuda()
                  for key in ('state', 'logits')}
    identity = pd.read_parquet(cache / 'validation_identity.parquet')
    np.testing.assert_array_equal(pd.DatetimeIndex(identity.timestamp).asi8, arrays['validation_origin_ns'])
    assert (pd.to_datetime(arrays['validation_origin_ns'], utc=True).tz_convert('Asia/Shanghai').year == 2023).all()
    streams = {name: shuffled_stream(indices) for name, indices in pools.items()}
    output.mkdir(parents=True, exist_ok=False)
    _, baseline = score(validation['logits'], identity)
    report = {'status': 'running', 'configuration': {
        'seed': 42, 'updates': 500, 'batch_size': 64, 'validation_every_updates': 5,
        'learning_rate': .001, 'gradient_clip_norm': 1., 'loss': 'masked_survival_nll',
        'selection': 'event_equal_then_row_earliest_step_including_zero',
        'frozen_base': True, 'input': 'historical_state_128_only',
        'pool_sizes': {key: len(value) for key, value in pools.items()},
        'stream': 'seed42_torch_randperm_cycles_concatenated_then_truncated_32000',
        'cycle_boundary_batch_may_repeat_origins': True},
        'source_sha256': tracked, 'baseline': baseline, 'arms': {}, 'limitations': [
            '2023_development_validation_not_independent_test',
            'base_checkpoint_previously_selected_using_2023',
            'equal_updates_not_equal_per_sample_exposure',
            'pool_year_season_label_composition_confounded',
            'single_seed', '68_hour_capped_metrics', 'no_deployment']}
    write_json(output / 'protocol.json', report)
    initial_digests = {}
    for arm, (nonlinear, pool) in ARMS.items():
        torch.manual_seed(42)
        model = HistoryStateResidual(nonlinear=nonlinear).cuda()
        digest = tensor_digest(model.state_dict())
        assert digest == initial_digests.setdefault(nonlinear, digest)
        with torch.no_grad():
            assert torch.count_nonzero(model(validation['state'])) == 0
        folder = output / arm
        folder.mkdir()
        np.save(folder / 'pool_indices.npy', pools[pool])
        np.save(folder / 'training_batches.npy', streams[pool])
        torch.save(model.state_dict(), folder / 'initial_state.pt')
        torch.save(model.state_dict(), folder / 'selected_state.pt')
        best_key = (baseline['event_equal_mae'], baseline['row_mae'])
        selected_step = 0
        trace, training_trace = [], []
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for step, indices in enumerate(streams[pool], 1):
            batch = torch.from_numpy(indices).cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = masked_survival_loss(train['logits'][batch] + model(train['state'][batch]),
                                        train['target'][batch], train['risk'][batch], train['eligible'][batch])
            assert loss is not None and torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            training_trace.append({'step': step, 'training_nll': float(loss.detach()),
                                   'cumulative_exposures': step * 64})
            if step % 5 == 0:
                model.eval()
                with torch.no_grad():
                    logits = validation['logits'] + model(validation['state'])
                    _, metrics = score(logits, identity)
                trace.append({'step': step, 'validation_2023': metrics})
                key = (metrics['event_equal_mae'], metrics['row_mae'])
                if key < best_key:
                    best_key, selected_step = key, step
                    torch.save(model.state_dict(), folder / 'selected_state.pt')
            if step % 100 == 0:
                write_json(folder / 'trace.json', trace)
                write_json(folder / 'training_trace.json', training_trace)
                print(f'{arm} step={step} selected={selected_step} key={best_key}', flush=True)
        model.load_state_dict(torch.load(folder / 'selected_state.pt', weights_only=True))
        model.eval()
        with torch.no_grad():
            logits = validation['logits'] + model(validation['state'])
        frame, metrics = score(logits, identity)
        for key, value in independent_scores(frame).items():
            if key.endswith('mae'):
                np.testing.assert_allclose(value, metrics[key], atol=1e-12)
        np.testing.assert_array_equal((probabilities(logits.cpu().numpy()).cumsum(1) < .5).sum(1).clip(max=67) + 1,
                                      frame.predicted_hours)
        frame.to_parquet(folder / 'validation.parquet', index=False)
        np.save(folder / 'logits.npy', logits.cpu().numpy())
        counts = np.bincount(streams[pool].ravel(), minlength=len(arrays['train_state']))[pools[pool]]
        item = {'selected_step': selected_step, 'updates_ran': 500, 'metrics': metrics,
                'initial_state_digest': digest, 'checkpoint_sha256': sha(folder / 'selected_state.pt'),
                'pool_sha256': sha(folder / 'pool_indices.npy'),
                'batches_sha256': sha(folder / 'training_batches.npy'),
                'exposures': {'total': int(counts.sum()), 'minimum_per_origin': int(counts.min()),
                              'maximum_per_origin': int(counts.max()), 'unique_origins': len(counts)}}
        report['arms'][arm] = item
        write_json(folder / 'report.json', item)
    assert hashes_unchanged(tracked)
    report['status'] = 'completed_development_comparison'
    write_json(output / 'report.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
