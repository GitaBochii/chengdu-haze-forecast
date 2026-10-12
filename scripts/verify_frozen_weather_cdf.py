"""Replay all frozen residual checkpoints and verify predeclared selection."""

import json

import numpy as np
import pandas as pd
import torch

from haze_forecast.frozen_weather_residual import FrozenWeatherResidual
from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.train_frozen_weather_cdf import ARMS
from scripts.train_recovery_weather_expanded import hashes_unchanged
from scripts.train_recovery_weather_formal import sha
from scripts.verify_recovery_auxiliary import independent_scores, probabilities
from scripts.verify_recovery_weather_formal import tensor_digest


def main():
    run = ROOT / 'artifacts/task01_recovery_head/frozen_weather_cdf_v1'
    cache = ROOT / 'artifacts/task01_recovery_head/frozen_weather_states_v1'
    report = json.loads((run / 'report.json').read_bytes())
    assert report['status'] == 'completed_development_comparison' and hashes_unchanged(report['source_sha256'])
    assert set(report['arms']) == set(ARMS)
    cache_report = json.loads((cache / 'report.json').read_bytes())
    assert hashes_unchanged(cache_report['source_sha256'])
    with np.load(cache / 'states.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    selected = arrays['train_mask'].any((1, 2)) & arrays['train_eligible'] & arrays['train_risk'].any(1)
    np.testing.assert_array_equal(np.load(run / 'training_indices.npy'), np.flatnonzero(selected))
    generator = torch.Generator().manual_seed(42)
    expected_orders = np.stack([torch.randperm(275, generator=generator).numpy() for _ in range(100)])
    np.testing.assert_array_equal(np.load(run / 'batch_orders.npy'), expected_orders)
    ref = tuple(torch.from_numpy(arrays['train_' + key][selected]).cuda() for key in ('state', 'weather', 'mask'))
    query = tuple(torch.from_numpy(arrays['validation_' + key]).cuda() for key in ('state', 'weather', 'mask'))
    initial_digest = None
    results = {}
    identity = pd.read_parquet(cache / 'validation_identity.parquet')
    for arm, (use_weather, centered) in ARMS.items():
        folder = run / arm
        item = report['arms'][arm]
        assert json.loads((folder / 'report.json').read_bytes()) == item and item['epochs_ran'] == 100
        trace = json.loads((folder / 'trace.json').read_bytes())
        assert [row['epoch'] for row in trace] == list(range(1, 101))
        assert all(np.isfinite(row['training_cdf_score']) for row in trace)
        candidates = [(0, report['baseline'])] + [(row['epoch'], row['validation_2023']) for row in trace]
        chosen = min(candidates, key=lambda row: (row[1]['event_equal_mae'], row[1]['row_mae'], row[0]))
        assert item['selected_epoch'] == chosen[0] and item['metrics'] == chosen[1]
        model = FrozenWeatherResidual(ref[0].shape[1], use_weather=use_weather, centered=centered).cuda().eval()
        state = torch.load(folder / 'initial_state.pt', weights_only=True)
        digest = tensor_digest(state)
        if initial_digest is None:
            initial_digest = digest
        assert initial_digest == digest
        model.load_state_dict(state)
        with torch.no_grad():
            assert torch.count_nonzero(model(*query, reference=ref)) == 0
        assert sha(folder / 'selected_state.pt') == item['checkpoint_sha256']
        model.load_state_dict(torch.load(folder / 'selected_state.pt', weights_only=True))
        before = tensor_digest(model.state_dict())
        with torch.no_grad():
            correction = model(*query, reference=ref).cpu().numpy()
            if centered:
                np.testing.assert_allclose(model(*ref, reference=ref).mean(0).cpu(), 0., atol=2e-6)
        assert tensor_digest(model.state_dict()) == before
        assert np.count_nonzero(correction[~arrays['validation_mask'].any((1, 2))]) == 0
        logits = arrays['validation_logits'] + correction
        saved_logits = np.load(folder / 'logits.npy')
        np.testing.assert_allclose(logits, saved_logits, rtol=1e-5, atol=1e-6)
        median = (probabilities(logits).cumsum(1) < .5).sum(1).clip(max=67) + 1
        frame = pd.read_parquet(folder / 'validation.parquet')
        cols = ['timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window']
        pd.testing.assert_frame_equal(frame[cols], identity[cols])
        np.testing.assert_array_equal(frame.predicted_hours, median)
        scores = independent_scores(frame)
        for key, value in scores.items():
            if key.endswith('mae'):
                np.testing.assert_allclose(value, item['metrics'][key], atol=1e-12)
        results[arm] = {'selected_epoch': item['selected_epoch'], 'metrics': scores,
                        'replay_max_difference': float(abs(logits - saved_logits).max())}
    assert hashes_unchanged(report['source_sha256']) and hashes_unchanged(cache_report['source_sha256'])
    result = {'verification': 'PASS', 'arms': results, 'limitations': ['cached_backbone_reference',
              'development_validation_only', 'no_optimizer_step_replay']}
    write_json(run / 'independent_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()


