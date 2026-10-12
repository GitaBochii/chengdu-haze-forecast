"""Frozen-backbone observed-CDF loss comparison; no censoring-unbiased claim."""

import json

import numpy as np
import pandas as pd
import torch
from haze_forecast.recovery_cdf_loss import observed_cdf_loss

from haze_forecast.frozen_weather_residual import FrozenWeatherResidual
from haze_forecast.recovery_selection import selection_metrics
from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.train_recovery_weather_expanded import hashes_unchanged
from scripts.train_recovery_weather_formal import sha, validated_distribution
from scripts.verify_recovery_auxiliary import independent_scores, probabilities
from scripts.verify_recovery_weather_formal import tensor_digest

ARMS = {'plain_control': (False, False), 'plain_weather': (True, False),
        'centered_control': (False, True), 'centered_weather': (True, True)}


def score(logits, identity):
    probability = validated_distribution(logits.detach().cpu()).numpy()
    median = (probability.cumsum(1) < .5).sum(1).clip(max=67) + 1
    frame = identity.copy()
    frame['predicted_hours'] = median
    frame['probability_recovered_by_67h'] = 1 - probability[:, -1]
    return frame, selection_metrics(frame)


def main():
    cache = ROOT / 'artifacts/task01_recovery_head/frozen_weather_states_v1'
    output = ROOT / 'artifacts/task01_recovery_head/frozen_weather_cdf_v1'
    assert not output.exists() and torch.cuda.is_available()
    cache_report = json.loads((cache / 'report.json').read_bytes())
    assert cache_report['status'] == 'frozen_backbone_cache_verified'
    assert hashes_unchanged(cache_report['source_sha256'])
    assert sha(cache / 'states.npz') == cache_report['state_cache_sha256']
    assert sha(cache / 'validation_identity.parquet') == cache_report['validation_identity_sha256']
    tracked = {str(path.relative_to(ROOT)): sha(path) for path in
               (cache / 'states.npz', cache / 'report.json', cache / 'validation_identity.parquet',
                ROOT / 'src/haze_forecast/frozen_weather_residual.py',
                ROOT / 'src/haze_forecast/recovery_cdf_loss.py',
                ROOT / 'scripts/train_frozen_weather_cdf.py')}
    torch.set_num_threads(2)
    with np.load(cache / 'states.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    selected = arrays['train_mask'].any(axis=(1, 2)) & arrays['train_eligible'] & arrays['train_risk'].any(1)
    assert selected.sum() == 275 and len(selected) == 12337
    train = {key: torch.from_numpy(arrays['train_' + key][selected]).cuda()
             for key in ('state', 'weather', 'mask', 'logits', 'target', 'risk', 'eligible')}
    validation = {key: torch.from_numpy(arrays['validation_' + key]).cuda()
                  for key in ('state', 'weather', 'mask', 'logits')}
    reference = (train['state'], train['weather'], train['mask'])
    identity = pd.read_parquet(cache / 'validation_identity.parquet')
    np.testing.assert_array_equal(pd.DatetimeIndex(identity.timestamp).asi8, arrays['validation_origin_ns'])
    generator = torch.Generator().manual_seed(42)
    orders = np.stack([torch.randperm(275, generator=generator).numpy() for _ in range(100)])
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'batch_orders.npy', orders)
    np.save(output / 'training_indices.npy', np.flatnonzero(selected))
    _, baseline = score(validation['logits'], identity)
    report = {'status': 'running', 'configuration': {'seed': 42, 'epochs': 100, 'batch_size': 64,
               'learning_rate': .001, 'loss': 'observed_cdf_squared_error_row_mean_no_ipcw', 'selection': 'event_equal_then_row_earliest_epoch_including_zero',
               'frozen_base': True, 'train_origins': 275, 'centering_reference': 'all_275_training_origins_only'},
               'source_sha256': tracked, 'baseline': baseline, 'arms': {},
               'limitations': ['2023_development_validation_not_independent_test',
                               'base_checkpoint_previously_selected_using_2023',
                               'residual_only_training_on_informative_weather_origins',
                               'single_seed', '68_hour_capped_metrics', 'no_deployment', 'observed_score_not_ipcw_unbiased_brier']}
    write_json(output / 'protocol.json', report)
    initial_digest = None
    for arm, (use_weather, centered) in ARMS.items():
        torch.manual_seed(42)
        model = FrozenWeatherResidual(train['state'].shape[1], use_weather=use_weather, centered=centered).cuda()
        digest = tensor_digest(model.state_dict())
        if initial_digest is None:
            initial_digest = digest
        assert digest == initial_digest
        folder = output / arm
        folder.mkdir()
        torch.save(model.state_dict(), folder / 'initial_state.pt')
        torch.save(model.state_dict(), folder / 'selected_state.pt')
        best_key = (baseline['event_equal_mae'], baseline['row_mae'])
        selected_epoch = 0
        trace = []
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for epoch, order in enumerate(orders, 1):
            model.train()
            total = 0.
            for offset in range(0, len(order), 64):
                batch = torch.from_numpy(order[offset:offset + 64]).cuda()
                optimizer.zero_grad(set_to_none=True)
                delta = model(train['state'][batch], train['weather'][batch], train['mask'][batch], reference=reference)
                loss = observed_cdf_loss(train['logits'][batch] + delta, train['target'][batch],
                                            train['risk'][batch], train['eligible'][batch])
                assert loss is not None and torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            model.eval()
            with torch.no_grad():
                logits = validation['logits'] + model(validation['state'], validation['weather'],
                                                       validation['mask'], reference=reference)
                _, metrics = score(logits, identity)
            key = (metrics['event_equal_mae'], metrics['row_mae'])
            trace.append({'epoch': epoch, 'training_cdf_score': total / 275, 'validation_2023': metrics})
            if key < best_key:
                best_key, selected_epoch = key, epoch
                torch.save(model.state_dict(), folder / 'selected_state.pt')
            if epoch % 20 == 0:
                write_json(folder / 'trace.json', trace)
                print(f'{arm} epoch={epoch} selected={selected_epoch} key={best_key}', flush=True)
        model.load_state_dict(torch.load(folder / 'selected_state.pt', weights_only=True))
        model.eval()
        with torch.no_grad():
            logits = validation['logits'] + model(validation['state'], validation['weather'], validation['mask'], reference=reference)
        frame, metrics = score(logits, identity)
        for key, value in independent_scores(frame).items():
            if key.endswith('mae'):
                np.testing.assert_allclose(value, metrics[key], atol=1e-12)
        np.testing.assert_array_equal((probabilities(logits.cpu().numpy()).cumsum(1) < .5).sum(1).clip(max=67) + 1,
                                      frame.predicted_hours)
        frame.to_parquet(folder / 'validation.parquet', index=False)
        np.save(folder / 'logits.npy', logits.cpu().numpy())
        item = {'selected_epoch': selected_epoch, 'epochs_ran': 100, 'metrics': metrics,
                'checkpoint_sha256': sha(folder / 'selected_state.pt')}
        report['arms'][arm] = item
        write_json(folder / 'report.json', item)
    assert hashes_unchanged(tracked) and hashes_unchanged(cache_report['source_sha256'])
    report['status'] = 'completed_development_comparison'
    write_json(output / 'report.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()


