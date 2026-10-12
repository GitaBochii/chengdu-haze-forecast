"""Cache an immutable trained historical backbone for residual-only experiments."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.benchmark_recovery_head import ROOT, load_experiment
from scripts.benchmark_regional_air_models import write_json
from scripts.train_recovery_weather_expanded import hashes_unchanged, sources_unchanged
from scripts.train_recovery_weather_formal import (
    ARMS,
    FormalWeatherWindows,
    actual_input_hashes,
    sha,
)
from scripts.verify_recovery_weather_formal import tensor_digest


def main():
    run = ROOT / 'artifacts/task01_recovery_head/future_weather_expanded_formal_v1'
    output = ROOT / 'artifacts/task01_recovery_head/frozen_weather_states_v1'
    assert not output.exists()
    report = json.loads((run / 'report.json').read_bytes())
    verified = json.loads((run / 'independent_verification.json').read_bytes())
    assert report['status'] == 'completed_train_validation_only' and verified['verification'] == 'PASS'
    assert verified['checkpoint_replay_verified']
    assert hashes_unchanged(report['source_code_sha256']) and sources_unchanged(report['sources'])
    checkpoint = run / 'base/selected_state.pt'
    assert sha(checkpoint) == report['arms']['base']['checkpoint_sha256']
    tracked = {str(path.relative_to(ROOT)): sha(path) for path in
               (run / 'report.json', run / 'independent_verification.json', checkpoint,
                run / 'weather_inputs.npz', run / 'base/validation_2023_distribution.npz',
                run / 'base/validation_2023_recovery.parquet')}
    args = SimpleNamespace(**report['configuration'])
    assert actual_input_hashes(ROOT / args.data) == report['inputs_before']
    torch.set_num_threads(2)
    panel, city, _, eligible, partitions = load_experiment(args)
    with np.load(run / 'weather_inputs.npz', allow_pickle=False) as saved:
        times = pd.to_datetime(saved['origin_ns'], utc=True)
        indices = panel['timestamps'].get_indexer(times)
        assert (indices >= 0).all()
        mapping = {int(origin): (torch.from_numpy(saved['normalized'][i].copy()),
                                torch.from_numpy(saved['mask'][i].copy())) for i, origin in enumerate(indices)}
    model = ARMS['base'](len(panel['feature_names']), panel['coords'], panel['scaler'].mean[-2:],
                         panel['scaler'].scale[-2:]).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.requires_grad_(False)
    before_model = tensor_digest(model.state_dict())
    arrays, audits = {}, {}
    for split, origins in [('train', np.concatenate([partitions[y] for y in range(2014, 2023)])),
                            ('validation', partitions[2023])]:
        dataset = FormalWeatherWindows(panel, origins, city.aqi.to_numpy(), eligible, mapping)
        assert dataset.audit() == report[split.replace('train', 'training') + '_label_audit'] if split == 'train' else dataset.audit() == report['validation_label_audit']
        states, logits, weather, masks = [], [], [], []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=256, shuffle=False):
                x, _, _, active, _, _, _, w, m = batch
                _, state = model.forward_with_state(x.cuda(), active.cuda())
                states.append(state.cpu().numpy())
                logits.append(model.risk_head(state).cpu().numpy())
                weather.append(w.numpy())
                masks.append(m.numpy())
        for key, chunks in [('state', states), ('logits', logits), ('weather', weather), ('mask', masks)]:
            arrays[f'{split}_{key}'] = np.concatenate(chunks)
        for key, value in [('origin_ns', panel['timestamps'][origins].asi8), ('target', dataset.hazard),
                           ('risk', dataset.risk_mask), ('eligible', dataset.eligible)]:
            arrays[f'{split}_{key}'] = value
        audits[split] = dataset.audit()
    with np.load(run / 'base/validation_2023_distribution.npz', allow_pickle=False) as saved:
        np.testing.assert_allclose(arrays['validation_logits'], saved['logits'], rtol=1e-5, atol=1e-6)
        for key, old in [('origin_ns', 'origin_ns'), ('target', 'target'), ('risk', 'risk_mask'), ('eligible', 'eligible')]:
            np.testing.assert_array_equal(arrays[f'validation_{key}'], saved[old])
    assert tensor_digest(model.state_dict()) == before_model
    assert hashes_unchanged(tracked) and actual_input_hashes(ROOT / args.data) == report['inputs_before']
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'states.npz', **arrays)
    score_frame = pd.read_parquet(run / 'base/validation_2023_recovery.parquet')
    score_frame.to_parquet(output / 'validation_identity.parquet', index=False)
    result = {'status': 'frozen_backbone_cache_verified', 'source_sha256': tracked,
              'backbone_state_digest': before_model, 'input_hashes': report['inputs_before'],
              'state_cache_sha256': sha(output / 'states.npz'),
              'validation_identity_sha256': sha(output / 'validation_identity.parquet'), 'audits': audits,
              'limitations': ['base_checkpoint_selected_using_2023', 'not_independent_test',
                              'full_history_backbone_frozen_no_optimizer_updates']}
    write_json(output / 'report.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
