"""Four real-history forward checks; no optimization or accuracy evaluation."""

import argparse

import numpy as np
import pandas as pd
import torch

from haze_forecast.recovery_weather_network import (
    WeatherControlRecoveryPM25GNN,
    WeatherRecoveryPM25GNN,
)
from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from scripts.benchmark_recovery_head import ROOT, RecoveryWindows, load_experiment
from scripts.benchmark_regional_air_models import write_json


def main():
    torch.set_num_threads(2)
    args = argparse.Namespace(data='data/processed/regional_air_models_full_history',
                              history=24, horizon=72, validation_year=2023)
    panel, city, _, eligible, partitions = load_experiment(args)
    source = ROOT / 'artifacts/task01_recovery_head/future_weather_inputs_v1/weather_inputs.npz'
    with np.load(source, allow_pickle=False) as saved:
        times = pd.to_datetime(saved['origin_ns'], utc=True)
        weather = torch.from_numpy(saved['normalized'].astype(np.float32))
        mask = torch.from_numpy(saved['mask'])
    origins = panel['timestamps'].get_indexer(times)
    assert (origins >= 0).all()
    for index, time in zip(origins, times, strict=True):
        assert index in partitions[time.year]
    dataset = RecoveryWindows(panel, origins, 24, 72, city.aqi.to_numpy(), eligible)
    x = torch.stack([dataset[i][0] for i in range(4)])
    active = torch.stack([dataset[i][3] for i in range(4)])
    models = []
    for cls in (RecoveryPM25GNN, WeatherRecoveryPM25GNN, WeatherControlRecoveryPM25GNN):
        torch.manual_seed(42)
        models.append(cls(len(panel['feature_names']), panel['coords'],
                          panel['scaler'].mean[-2:], panel['scaler'].scale[-2:]).eval())
    with torch.no_grad():
        baseline = models[0](x, active)
        for model in models[1:]:
            outputs = model(x, active, weather, mask)
            for actual, expected in zip(outputs, baseline, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert torch.isfinite(actual).all()
    result = {'status': 'pass', 'origin_count': 4, 'history_shape': list(x.shape),
              'weather_shape': list(weather.shape), 'curve_shape': list(baseline[0].shape),
              'hazard_shape': list(baseline[1].shape),
              'parameters': {type(model).__name__: sum(p.numel() for p in model.parameters()) for model in models},
              'base_equivalence_exact': True, 'history_scaler_cutoff_year': 2023,
              'limitations': ['random_initial_weights_not_trained_checkpoint',
                              'forward_smoke_check_not_forecast_skill', 'no_weights_saved_or_deployed']}
    output = ROOT / 'artifacts/task01_recovery_head/future_weather_network_v1'
    output.mkdir(exist_ok=False)
    write_json(output / 'report.json', result)
    print(result)


if __name__ == '__main__':
    main()
