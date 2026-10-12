"""Frozen-checkpoint branch ablation; descriptive validation evidence only."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from haze_forecast.recovery_selection import selection_metrics
from scripts.benchmark_recovery_head import ROOT, duration_records, load_experiment
from scripts.benchmark_regional_air_models import write_json
from scripts.train_recovery_weather_expanded import hashes_unchanged, sources_unchanged
from scripts.train_recovery_weather_formal import (
    ARMS,
    FormalWeatherWindows,
    actual_input_hashes,
    sha,
    validated_distribution,
)
from scripts.verify_recovery_weather_formal import tensor_digest


def main():
    run = ROOT / 'artifacts/task01_recovery_head/future_weather_expanded_formal_v1'
    output = ROOT / 'artifacts/task01_recovery_head/future_weather_branch_diagnosis_v1'
    assert not output.exists()
    report = json.loads((run / 'report.json').read_bytes())
    checked = json.loads((run / 'independent_verification.json').read_bytes())
    assert report['status'] == 'completed_train_validation_only'
    assert checked['verification'] == 'PASS' and checked['checkpoint_replay_verified']
    args = SimpleNamespace(**report['configuration'])
    assert hashes_unchanged(report['source_code_sha256']) and sources_unchanged(report['sources'])
    assert actual_input_hashes(ROOT / args.data) == report['inputs_before']
    files = [run / 'report.json', run / 'independent_verification.json', run / 'weather_inputs.npz']
    files += [run / arm / name for arm in ARMS for name in
              ('selected_state.pt', 'validation_2023_distribution.npz', 'validation_2023_recovery.parquet')]
    before = {str(path): sha(path) for path in files}
    torch.set_num_threads(2)
    panel, city, labels, eligible, partitions = load_experiment(args)
    assert sha(run / 'weather_inputs.npz') == report['weather_inputs_sha256']
    with np.load(run / 'weather_inputs.npz', allow_pickle=False) as saved:
        times = pd.to_datetime(saved['origin_ns'], utc=True)
        indices = panel['timestamps'].get_indexer(times)
        assert (indices >= 0).all()
        mapping = {int(origin): (torch.from_numpy(saved['normalized'][i].copy()),
                                torch.from_numpy(saved['mask'][i].copy())) for i, origin in enumerate(indices)}
    dataset = FormalWeatherWindows(panel, partitions[2023], city.aqi.to_numpy(), eligible, mapping)
    assert dataset.audit() == report['validation_label_audit']
    results, arrays, predictions = {}, {}, {}
    for arm in ARMS:
        model = ARMS[arm](len(panel['feature_names']), panel['coords'], panel['scaler'].mean[-2:],
                          panel['scaler'].scale[-2:]).cuda().eval()
        assert sha(run / arm / 'selected_state.pt') == report['arms'][arm]['checkpoint_sha256']
        model.load_state_dict(torch.load(run / arm / 'selected_state.pt', weights_only=True))
        model_hash = tensor_digest(model.state_dict())
        full, backbone = [], []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=256, shuffle=False):
                x, _, _, active, _, _, _, weather, mask = [value.cuda() for value in batch]
                _, state = model.forward_with_state(x, active)
                base = model.risk_head(state)
                backbone.append(base.cpu())
                if arm == 'base':
                    full.append(base.cpu())
                else:
                    full.append(model(x, active, weather, mask)[1].cpu())
        assert tensor_digest(model.state_dict()) == model_hash
        full, backbone = torch.cat(full), torch.cat(backbone)
        with np.load(run / arm / 'validation_2023_distribution.npz', allow_pickle=False) as saved:
            np.testing.assert_allclose(full.numpy(), saved['logits'], rtol=1e-5, atol=1e-6)
        for variant, logits in [('full', full), ('backbone_only', backbone)]:
            probability = validated_distribution(logits)
            median = ((probability.cumsum(1) < .5).sum(1).clamp_max(67) + 1).numpy()
            frame = duration_records(dataset.origins, panel, labels, city.aqi.to_numpy(), median, 72)
            score = selection_metrics(frame)
            if variant == 'full':
                for metric in ('row_mae', 'event_equal_mae', 'tail_event_equal_mae'):
                    np.testing.assert_allclose(score[metric], report['arms'][arm]['validation_2023'][metric])
            results[f'{arm}_{variant}'] = score
            arrays[f'{arm}_{variant}'] = logits.numpy()
            predictions[f'{arm}_{variant}'] = frame
        del model
    assert before == {str(path): sha(path) for path in files}
    assert hashes_unchanged(report['source_code_sha256'])
    output.mkdir(parents=True, exist_ok=False)
    for variant, frame in predictions.items():
        frame.to_parquet(output / f'{variant}.parquet', index=False)
    np.savez_compressed(output / 'logits.npz', **arrays)
    result = {'status': 'completed_frozen_checkpoint_ablation', 'input_sha256': before,
              'metrics': results, 'limitations': ['2023_development_validation_not_independent_test',
              'backbone_only_is_post_training_ablation_not_new_trained_model',
              'all_durations_capped_at_68', 'no_optimizer_updates_or_deployment']}
    write_json(output / 'report.json', result)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
