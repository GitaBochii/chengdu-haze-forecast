"""Isolated causal event-age residual ablation using frozen controls."""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch

from haze_forecast.recovery_age import AgeRecoveryPM25GNN, causal_recovery_age
from haze_forecast.recovery_selection import development_gate
from scripts.benchmark_recovery_auxiliary import compare_fold
from scripts.benchmark_recovery_head import RecoveryWindows, load_experiment
from scripts.benchmark_recovery_selection import (
    ROOT,
    evaluate_selection_pair,
    file_hash,
    train_selection_pair,
    verify_frozen,
)
from scripts.benchmark_regional_air_models import hashes, write_json


class AgeWindows(RecoveryWindows):
    def __getitem__(self, item):
        x, *rest = super().__getitem__(item)
        origin = self.origins[item]
        age = torch.as_tensor(self.panel['recovery_age'][origin], dtype=x.dtype)
        extra = age.expand(*x.shape[:-1], 1)
        return (torch.cat([x, extra], dim=-1), *rest)


def run_age_fold(args, year, output, device, *, model_class=AgeRecoveryPM25GNN):
    output.mkdir(parents=True, exist_ok=False)
    args = argparse.Namespace(**{**vars(args), 'validation_year': year})
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    panel, city, labels, eligible, partitions = load_experiment(args)
    age = causal_recovery_age(city.aqi.to_numpy(), labels.reindex(panel['timestamps']).particle_primary.to_numpy())
    panel['recovery_age'] = (np.log1p(age) / np.log1p(168.)).astype(np.float32)
    aqi = city.aqi.to_numpy()
    origins = np.concatenate([partitions[y] for y in range(2014, year)])
    training = AgeWindows(panel, origins, 24, 72, aqi, eligible)
    validation = AgeWindows(panel, partitions[year], 24, 72, aqi, eligible)
    protocol = {'configuration': vars(args), 'validation_year': year, 'test_years': [year + 1],
                'nodes': panel['nodes'], 'feature_names': panel['feature_names'],
                'coordinates_lon_lat': panel['coords'].tolist(),
                'scaler': {'mean': panel['scaler'].mean.tolist(), 'scale': panel['scaler'].scale.tolist()},
                'training_windows_by_year': {y: len(partitions[y]) for y in range(2014, year)},
                'partitions': {'training': len(training), 'validation': len(validation),
                               str(year + 1): len(partitions[year + 1])},
                'training_label_audit': training.audit(), 'validation_label_audit': validation.audit(),
                'age_definition': 'causal_three_polluted_confirm_six_clean_reset_log1p_scale168',
                'age_training_max_hours': float(age[origins].max()),
                'model_class': model_class.__name__}
    write_json(output / 'protocol.json', protocol)
    pd.DataFrame({'timestamp': panel['timestamps'][origins], 'age_hours': age[origins]}).to_parquet(
        output / 'training_age.parquet', index=False)
    model = model_class(len(panel['feature_names']), panel['coords'],
                        panel['scaler'].mean[-2:], panel['scaler'].scale[-2:]).to(device)
    trained = train_selection_pair(model, training, validation, panel, labels, aqi, args, output, device)
    report = {**protocol, 'epochs_ran': len(trained['trace']), 'selections': trained['selections'],
              'frozen': trained['frozen'], 'years': {}}
    dataset = AgeWindows(panel, partitions[year + 1], 24, 72, aqi, eligible)
    report['years'][str(year + 1)] = evaluate_selection_pair(
        model, dataset, panel, labels, aqi, args, output, device, year + 1, trained['frozen'])
    write_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parsed = parser.parse_args()
    control = ROOT / 'artifacts/task01_recovery_head/nested_selection_v1'
    reference = json.loads((control / 'report.json').read_text('utf-8'))
    args = argparse.Namespace(**{**reference['configuration'], 'output': parsed.output})
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    deployment = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    before, deployed = hashes(ROOT / args.data), hashes(deployment)
    assert before == reference['inputs_before']
    control_hashes = {str(p.relative_to(control)): file_hash(p) for p in control.rglob('*') if p.is_file()}
    for year in ('2022', '2023'):
        verify_frozen(control / f'validation_{year}', reference['folds'][year]['frozen'])
    report = {'configuration': vars(args), 'control': str(control), 'control_sha256': control_hashes,
              'inputs_before': before, 'deployment_before': deployed, 'folds': {}, 'decision': 'running'}
    write_json(output / 'protocol.json', report)
    try:
        torch.set_num_threads(2)
        pairs = []
        for year in (2022, 2023):
            candidate = run_age_fold(args, year, output / f'validation_{year}', torch.device(args.device))
            original = reference['folds'][str(year)]
            compare_fold(original, candidate)
            identity = ['timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window']
            a = pd.read_parquet(control / f'validation_{year}/legacy_{year+1}_recovery.parquet')
            b = pd.read_parquet(output / f'validation_{year}/legacy_{year+1}_recovery.parquet')
            pd.testing.assert_frame_equal(a[identity], b[identity])
            pairs.append({'legacy': original['years'][str(year + 1)]['legacy'],
                          'balanced': candidate['years'][str(year + 1)]['legacy']})
            report['folds'][str(year)] = candidate
            write_json(output / 'progress.json', report)
        report['gate'] = development_gate(pairs)
        report['decision'] = 'development_candidate_only' if report['gate']['accepted'] else 'reject'
    except Exception as exc:
        report['decision'], report['error'] = 'error', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['inputs_unchanged'] = hashes(ROOT / args.data) == before
        report['deployment_unchanged'] = hashes(deployment) == deployed
        report['control_unchanged'] = all(file_hash(control / p) == h for p, h in control_hashes.items())
        integrity = all(report[k] for k in ('inputs_unchanged', 'deployment_unchanged', 'control_unchanged'))
        if not integrity:
            report['decision'] = 'error'
        write_json(output / 'report.json', report)
    if not integrity:
        raise RuntimeError('input, deployment or control mutated')


if __name__ == '__main__':
    main()
