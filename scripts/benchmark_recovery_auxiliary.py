"""Single-variable auxiliary-loss ablation with frozen chronological controls."""

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from haze_forecast.recovery_selection import development_gate
from scripts.benchmark_recovery_selection import ROOT, file_hash, run_fold, verify_frozen
from scripts.benchmark_regional_air_models import hashes, write_json


def compare_fold(control, candidate):
    candidate = json.loads(json.dumps(candidate))
    for key in ('nodes', 'feature_names', 'coordinates_lon_lat', 'scaler', 'partitions',
                'training_windows_by_year', 'training_label_audit', 'validation_label_audit'):
        if control[key] != candidate[key]:
            raise ValueError(f'control mismatch: {key}')
    for key in ('data', 'epochs', 'batch_size', 'learning_rate', 'seed', 'device', 'history', 'horizon'):
        if control['configuration'][key] != candidate['configuration'][key]:
            raise ValueError(f'configuration mismatch: {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    resume = args.resume
    control = ROOT / 'artifacts/task01_recovery_head/nested_selection_v1'
    reference = json.loads((control / 'report.json').read_text('utf-8'))
    config = reference['configuration']
    assert config['auxiliary_weight'] == .1 and config['epochs'] == 12 and config['seed'] == 42
    output_arg = args.output
    args = argparse.Namespace(**{**config, 'output': output_arg, 'auxiliary_weight': 0.})
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=resume)
    deployment = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    before, deployed = hashes(ROOT / args.data), hashes(deployment)
    assert before == reference['inputs_before']
    control_hashes = {str(path.relative_to(control)): file_hash(path)
                      for path in control.rglob('*') if path.is_file()}
    if resume:
        previous = json.loads((output / 'report.json').read_text('utf-8'))
        assert previous['configuration'] == vars(args)
        assert previous['inputs_before'] == before
        assert previous['deployment_before'] == deployed
        assert previous['control_sha256'] == control_hashes
        write_json(output / 'previous_failure.json', previous)
    for year in ('2022', '2023'):
        verify_frozen(control / f'validation_{year}', reference['folds'][year]['frozen'])
    report = {'configuration': vars(args), 'control': str(control),
              'control_sha256': control_hashes, 'inputs_before': before,
              'deployment_before': deployed, 'folds': {}, 'decision': 'running'}
    write_json(output / 'protocol.json', report)
    try:
        torch.set_num_threads(2)
        pairs = []
        for year in (2022, 2023):
            folder = output / f'validation_{year}'
            if resume and (folder / 'report.json').exists():
                candidate = json.loads((folder / 'report.json').read_text('utf-8'))
                assert candidate['configuration']['auxiliary_weight'] == 0.
                assert candidate['epochs_ran'] == 12
                verify_frozen(folder, candidate['frozen'])
            else:
                candidate = run_fold(args, year, [year + 1], folder, torch.device(args.device))
            original = reference['folds'][str(year)]
            compare_fold(original, candidate)
            a = pd.read_parquet(control / f'validation_{year}/legacy_{year+1}_recovery.parquet')
            b = pd.read_parquet(output / f'validation_{year}/legacy_{year+1}_recovery.parquet')
            identity = ['timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window']
            pd.testing.assert_frame_equal(a[identity], b[identity])
            pairs.append({'legacy': original['years'][str(year + 1)]['legacy'],
                          'balanced': candidate['years'][str(year + 1)]['legacy']})
            report['folds'][str(year)] = candidate
            write_json(output / 'progress.json', report)
        gate = development_gate(pairs)
        report['gate'] = {'accepted': gate['accepted'], 'checks': gate['checks'],
                          'independent_tail_events': gate['independent_tail_events'],
                          'annual_means': {'auxiliary_0_1': gate['annual_equal_means']['legacy'],
                                           'auxiliary_0': gate['annual_equal_means']['balanced']}}
        report['decision'] = 'development_candidate_only' if gate['accepted'] else 'reject'
    except Exception as exc:
        report['decision'], report['error'] = 'error', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['inputs_unchanged'] = hashes(ROOT / args.data) == before
        report['deployment_unchanged'] = hashes(deployment) == deployed
        report['control_unchanged'] = all(file_hash(control / path) == digest
                                          for path, digest in control_hashes.items())
        integrity = all(report[key] for key in ('inputs_unchanged', 'deployment_unchanged', 'control_unchanged'))
        if not integrity:
            report['decision'] = 'error'
        write_json(output / 'report.json', report)
    if not integrity:
        raise RuntimeError('input, deployment or control mutated')


if __name__ == '__main__':
    main()
