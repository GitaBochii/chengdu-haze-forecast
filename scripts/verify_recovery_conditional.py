"""Independently verify the matched-capacity development comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from haze_forecast.recovery_conditional import StateAgeRecoveryPM25GNN, StateOnlyRecoveryPM25GNN
from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from scripts.benchmark_recovery_head import load_experiment
from scripts.verify_recovery_age import check_age
from scripts.verify_recovery_auxiliary import (
    ARMS,
    METRICS,
    check_hashes,
    check_predictions,
    check_selection,
    independent_gate,
    read_json,
)

MODELS = {'state_only': StateOnlyRecoveryPM25GNN, 'state_age': StateAgeRecoveryPM25GNN}
PAIRS = {'state_only_vs_control': ('control', 'state_only'),
         'state_age_vs_control': ('control', 'state_age'),
         'state_age_vs_state_only': ('state_only', 'state_age')}


def check_parameters(folder, panel, model_class):
    arguments = (len(panel['feature_names']), panel['coords'],
                 panel['scaler'].mean[-2:], panel['scaler'].scale[-2:])
    base = RecoveryPM25GNN(*arguments)
    model = model_class(*arguments)
    expected = model.state_dict()
    checkpoint = torch.load(folder / 'legacy_state.pt', map_location='cpu', weights_only=True)
    assert set(checkpoint) == set(expected)
    for key, tensor in checkpoint.items():
        assert tensor.shape == expected[key].shape, key
        assert torch.isfinite(tensor).all(), key
    base_count = sum(p.numel() for p in base.parameters())
    count = sum(p.numel() for p in model.parameters())
    assert count - base_count == 3219
    extra = set(checkpoint) - set(base.state_dict())
    assert extra == {'conditional_head.0.weight', 'conditional_head.0.bias',
                     'conditional_head.2.weight', 'conditional_head.2.bias'}
    assert sum(checkpoint[k].numel() for k in extra) == 3219
    if model_class is StateOnlyRecoveryPM25GNN:
        # The unused age column receives zero gradient and must retain its initialization.
        torch.manual_seed(42)
        initial = model_class(*arguments)
        torch.testing.assert_close(checkpoint['conditional_head.0.weight'][:, -1],
                                   initial.conditional_head[0].weight[:, -1], rtol=0, atol=0)
    return {'base_parameters': base_count, 'parameters': count, 'added_parameters': count - base_count,
            'state_shapes': {k: list(v.shape) for k, v in checkpoint.items()}}


def check_gate(saved, results, old, new):
    pairs = [{ARMS[0]: result[old], ARMS[1]: result[new]} for result in results.values()]
    independent = independent_gate(pairs)
    assert independent['accepted'] == saved['accepted']
    assert independent['independent_tail_events'] == saved['independent_tail_events']
    for generic, stored in zip(ARMS, ('legacy', 'balanced'), strict=True):
        for metric in METRICS:
            np.testing.assert_allclose(independent['annual_means'][generic][metric],
                                       saved['annual_equal_means'][stored][metric], atol=1e-12)
    means = independent['annual_means']
    checks = {'sufficient_tail_events': independent['independent_tail_events'] >= 3
              and all(pair[ARMS[0]]['tail_events'] for pair in pairs),
              'finite_metrics': all(np.isfinite(means[a][m]) for a in ARMS for m in METRICS),
              'row_not_worse': means[ARMS[1]]['row_mae'] <= means[ARMS[0]]['row_mae'],
              'event_equal_not_worse': means[ARMS[1]]['event_equal_mae'] <= means[ARMS[0]]['event_equal_mae'],
              'tail_better': means[ARMS[1]]['tail_event_equal_mae'] < means[ARMS[0]]['tail_event_equal_mae']}
    assert checks == saved['checks']
    return independent


def verify(run):
    root = Path(__file__).resolve().parents[1]
    run = root / run
    report = read_json(run / 'report.json')
    assert report['decision'] == 'development_complete'
    assert set(report['arms']) == set(MODELS)
    assert set(report['gates']) == set(PAIRS)
    control = root / report['control']
    inputs = root / report['configuration']['data']
    deployment = root / 'artifacts/cnemc_2025_deployment_candidate'
    sources = ((control, report['control_sha256']), (inputs, report['inputs_before']),
               (deployment, report['deployment_before']))
    for folder, manifest in sources:
        check_hashes(folder, manifest)
    for arm in MODELS:
        assert set(report['arms'][arm]) == {'2022', '2023'}
        assert not (run / arm / 'validation_2024').exists()
        assert not list((run / arm).rglob('*_2025_*'))
        assert not list((run / arm).rglob('*_2026_*'))
    results, ages, parameters = {}, {}, {}
    for validation_year in (2022, 2023):
        key, year = str(validation_year), validation_year + 1
        base_folder = control / f'validation_{validation_year}'
        baseline = read_json(base_folder / 'report.json')
        configuration = baseline['configuration']
        for name, expected in (('seed', 42), ('epochs', 12), ('auxiliary_weight', .1),
                               ('history', 24), ('horizon', 72)):
            assert configuration[name] == expected
        panel, city, labels, eligible, partitions = load_experiment(SimpleNamespace(**configuration))
        np.testing.assert_array_equal(panel['scaler'].mean, baseline['scaler']['mean'])
        np.testing.assert_array_equal(panel['scaler'].scale, baseline['scaler']['scale'])
        assert panel['nodes'] == baseline['nodes']
        check_selection(base_folder, baseline)
        identity, scores = check_predictions(base_folder, year, baseline, panel, city, labels, eligible, partitions)
        results[key], ages[key], parameters[key] = {'control': scores}, {}, {}
        for arm, model_class in MODELS.items():
            fold = report['arms'][arm][key]
            folder = run / arm / f'validation_{validation_year}'
            assert read_json(folder / 'report.json') == fold
            assert fold['model_class'] == model_class.__name__
            assert fold['validation_year'] == validation_year
            assert fold['test_years'] == baseline['test_years'] == [year]
            assert set(fold['years']) == {str(year)}
            for name in ('seed', 'learning_rate', 'batch_size', 'epochs', 'data', 'history',
                         'horizon', 'validation_year', 'auxiliary_weight'):
                assert fold['configuration'][name] == configuration[name], name
            for name in ('nodes', 'feature_names', 'coordinates_lon_lat', 'scaler', 'training_windows_by_year',
                         'partitions', 'training_label_audit', 'validation_label_audit'):
                assert fold[name] == baseline[name], name
            assert fold['age_definition'] == 'causal_three_polluted_confirm_six_clean_reset_log1p_scale168'
            ages[key][arm] = check_age(folder, fold, panel, city, labels, partitions, validation_year)
            parameters[key][arm] = check_parameters(folder, panel, model_class)
            check_selection(folder, fold)
            candidate_identity, candidate_scores = check_predictions(
                folder, year, fold, panel, city, labels, eligible, partitions)
            pd.testing.assert_frame_equal(identity, candidate_identity)
            results[key][arm] = candidate_scores
        assert parameters[key]['state_only']['state_shapes'] == parameters[key]['state_age']['state_shapes']
    gates = {name: check_gate(report['gates'][name], results, old, new)
             for name, (old, new) in PAIRS.items()}
    means = {arm: {metric: float(np.mean([fold[arm][metric] for fold in results.values()]))
                   for metric in METRICS} for arm in ('control', *MODELS)}
    for folder, manifest in sources:
        check_hashes(folder, manifest)
    summary = {'verification': 'PASS', 'folds': results, 'age_checks': ages, 'parameter_checks': parameters,
               'annual_means': means, 'gates': gates, 'inputs_unchanged': True, 'control_unchanged': True,
               'deployment_unchanged': True,
               'limitations': ['saved_logits_recomputed_without_checkpoint_forward',
                               'initial_weights_not_saved_seed_and_configuration_checked',
                               'age_is_rule_memory_not_continuous_observation_guarantee',
                               'development_years_only_no_2025_2026_evaluation',
                               'single_seed_three_independent_development_tail_events']}
    (run / 'independent_verification.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    summary = verify(parser.parse_args().run)
    print(json.dumps({'verification': summary['verification'], 'annual_means': summary['annual_means'],
                      'gates': {key: gate['accepted'] for key, gate in summary['gates'].items()}}, indent=2))


if __name__ == '__main__':
    main()
