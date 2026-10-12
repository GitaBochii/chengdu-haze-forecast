"""Capacity-controlled nonlinear recovery-head development experiment."""

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from haze_forecast.recovery_conditional import StateAgeRecoveryPM25GNN, StateOnlyRecoveryPM25GNN
from haze_forecast.recovery_selection import development_gate
from scripts.benchmark_recovery_age import run_age_fold
from scripts.benchmark_recovery_auxiliary import compare_fold
from scripts.benchmark_recovery_selection import ROOT, file_hash, verify_frozen
from scripts.benchmark_regional_air_models import hashes, write_json


def paired_gate(reference_folds, candidate_folds):
    return development_gate([
        {'legacy': reference_folds[str(y)]['years'][str(y + 1)]['legacy'],
         'balanced': candidate_folds[str(y)]['years'][str(y + 1)]['legacy']}
        for y in (2022, 2023)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parsed = parser.parse_args()
    control = ROOT / 'artifacts/task01_recovery_head/nested_selection_v1'
    reference = json.loads((control / 'report.json').read_text('utf-8'))
    args = argparse.Namespace(**{**reference['configuration'], 'output': parsed.output})
    assert args.seed == 42 and args.epochs == 12 and args.auxiliary_weight == .1
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=parsed.resume)
    deployment = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    before, deployed = hashes(ROOT / args.data), hashes(deployment)
    assert before == reference['inputs_before']
    control_hashes = {str(p.relative_to(control)): file_hash(p) for p in control.rglob('*') if p.is_file()}
    for year in ('2022', '2023'):
        verify_frozen(control / f'validation_{year}', reference['folds'][year]['frozen'])
    if parsed.resume:
        previous = json.loads((output / 'report.json').read_text('utf-8'))
        assert previous['configuration'] == vars(args)
        assert previous['inputs_before'] == before and previous['deployment_before'] == deployed
        assert previous['control_sha256'] == control_hashes
        if (output / 'previous_attempt.json').exists():
            raise ValueError('only one explicitly audited resume is supported')
        write_json(output / 'previous_attempt.json', previous)
    report = {'configuration': vars(args), 'control': str(control), 'control_sha256': control_hashes,
              'inputs_before': before, 'deployment_before': deployed, 'arms': {}, 'decision': 'running',
              'architecture': 'base_logits + Linear129_16_Tanh_Linear16_67_zero_output',
              'final_years_evaluated': False}
    write_json(output / 'protocol.json', report)
    try:
        torch.set_num_threads(2)
        for arm, model_class in (('state_only', StateOnlyRecoveryPM25GNN),
                                  ('state_age', StateAgeRecoveryPM25GNN)):
            report['arms'][arm] = {}
            for year in (2022, 2023):
                folder = output / arm / f'validation_{year}'
                if parsed.resume and (folder / 'report.json').exists():
                    fold = json.loads((folder / 'report.json').read_text('utf-8'))
                    assert fold['model_class'] == model_class.__name__ and fold['epochs_ran'] == 12
                    verify_frozen(folder, fold['frozen'])
                else:
                    fold = run_age_fold(args, year, folder, torch.device(args.device), model_class=model_class)
                compare_fold(reference['folds'][str(year)], fold)
                identity = ['timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window']
                a = pd.read_parquet(control / f'validation_{year}/legacy_{year+1}_recovery.parquet')
                b = pd.read_parquet(folder / f'legacy_{year+1}_recovery.parquet')
                pd.testing.assert_frame_equal(a[identity], b[identity])
                report['arms'][arm][str(year)] = fold
                write_json(output / 'progress.json', report)
                print({'completed_arm': arm, 'validation_year': year}, flush=True)
        report['gates'] = {
            f'{arm}_vs_control': paired_gate(reference['folds'], report['arms'][arm])
            for arm in ('state_only', 'state_age')}
        report['gates']['state_age_vs_state_only'] = paired_gate(
            report['arms']['state_only'], report['arms']['state_age'])
        report['decision'] = 'development_complete'
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
