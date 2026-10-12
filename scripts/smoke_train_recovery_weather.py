"""Exercise three-arm optimization on four pilot origins; never report skill."""

import argparse
import hashlib

import numpy as np
import pandas as pd
import torch

from haze_forecast.recovery_survival import masked_survival_loss
from haze_forecast.recovery_weather_network import (
    WeatherControlRecoveryPM25GNN,
    WeatherRecoveryPM25GNN,
)
from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from scripts.benchmark_recovery_head import ROOT, RecoveryWindows, load_experiment
from scripts.benchmark_regional_air_models import hashes, masked_loss, write_json
from scripts.probe_future_weather_sequence import sequence_candidates


def main():
    torch.set_num_threads(2)
    output = ROOT / 'artifacts/task01_recovery_head/future_weather_training_smoke_v1'
    output.mkdir(exist_ok=False)
    deployed = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    before = hashes(deployed)
    args = argparse.Namespace(data='data/processed/regional_air_models_full_history',
                              history=24, horizon=72, validation_year=2023)
    panel, city, _, eligible, partitions = load_experiment(args)
    path = ROOT / 'artifacts/task01_recovery_head/future_weather_inputs_v1/weather_inputs.npz'
    with np.load(path, allow_pickle=False) as saved:
        times = pd.to_datetime(saved['origin_ns'], utc=True)
        weather = torch.from_numpy(saved['normalized'].astype(np.float32))
        weather_mask = torch.from_numpy(saved['mask'])
        training = torch.from_numpy(saved['training_rows'])
    assert training.tolist() == [time.year == 2022 for time in times]
    origins = panel['timestamps'].get_indexer(times)
    assert (origins >= 0).all()
    assert all(index in partitions[time.year] for index, time in zip(origins, times, strict=True))
    dataset = RecoveryWindows(panel, origins, 24, 72, city.aqi.to_numpy(), eligible)
    batches = [torch.stack([dataset[i][part] for i in range(4)]) for part in range(7)]
    summaries = []
    for cls in (RecoveryPM25GNN, WeatherControlRecoveryPM25GNN, WeatherRecoveryPM25GNN):
        torch.manual_seed(42)
        model = cls(len(panel['feature_names']), panel['coords'],
                    panel['scaler'].mean[-2:], panel['scaler'].scale[-2:])
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)

        def loss_for(selected, cls=cls, model=model):
            x, y, mask, active, target, risk, eligible_batch = [item[selected] for item in batches]
            if cls is RecoveryPM25GNN:
                curves, logits = model(x, active)
            else:
                curves, logits = model(x, active, weather[selected], weather_mask[selected])
            survival = masked_survival_loss(logits, target, risk, eligible_batch)
            auxiliary = masked_loss(curves, y, mask)
            if survival is None or auxiliary is None:
                raise ValueError('pilot batch lacks informative labels')
            return survival + .1 * auxiliary

        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        losses = []
        for _ in range(2):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(training)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        assert any(not torch.equal(value, initial[key]) for key, value in model.state_dict().items())
        model.eval()
        frozen = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            validation_loss = loss_for(~training)
        assert torch.isfinite(validation_loss)
        assert all(torch.equal(value, frozen[key]) for key, value in model.state_dict().items())
        summaries.append({'arm': cls.__name__, 'steps': 2, 'train_losses': losses,
                          'validation_loss': float(validation_loss), 'finite_gradients': True,
                          'validation_parameters_unchanged': True})
        write_json(output / 'progress.json', summaries)
    planned, counts = [], {}
    for year in (2022, 2023):
        selected = partitions[year][eligible[partitions[year]]]
        counts[str(year)] = len(selected)
        for index in selected:
            issue = panel['timestamps'][index].tz_convert('UTC')
            for candidate in sequence_candidates(issue):
                planned.append({**candidate, 'partition': 'train' if year == 2022 else 'validation',
                                'pilot_origin_already_collected': issue in times})
    pd.DataFrame(planned).to_csv(output / 'candidate_weather_requests.csv', index=False)
    after = hashes(deployed)
    assert before == after
    write_json(output / 'report.json', {
        'status': 'pass', 'arms': summaries, 'training_origins': 2, 'validation_origins': 2,
        'weather_input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'candidate_eligible_origins': counts, 'candidate_requests': len(planned),
        'candidate_requests_at_uncollected_origins': sum(not r['pilot_origin_already_collected'] for r in planned),
        'planned_requests_sha256': hashlib.sha256((output / 'candidate_weather_requests.csv').read_bytes()).hexdigest(),
        'history_training_years': '2014-2022', 'future_weather_candidate_years': [2022, 2023],
        'deployment_unchanged': before == after,
        'limitations': ['two_optimization_steps_only', 'no_accuracy_or_arm_ranking_conclusion',
                        'no_weights_saved', 'candidate_manifest_not_availability_audited',
                        'pre2022_weather_not_collected', 'noneligible_auxiliary_windows_must_not_be_dropped']})
    print(f'three-arm smoke PASS; candidate origins={counts}, requests={len(planned)}')


if __name__ == '__main__':
    main()
