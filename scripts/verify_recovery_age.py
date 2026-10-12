"""Verify the causal-age development comparison without training or final-year scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from haze_forecast.recovery_age import causal_recovery_age
from scripts.benchmark_recovery_head import load_experiment
from scripts.verify_recovery_auxiliary import (
    ARMS,
    METRICS,
    check_hashes,
    check_predictions,
    check_selection,
    independent_gate,
    read_json,
)


def check_age(folder, fold, panel, city, labels, partitions, validation_year):
    aqi = city.aqi.to_numpy()
    particle = labels.reindex(panel['timestamps']).particle_primary.to_numpy()
    age = causal_recovery_age(aqi, particle)
    origins = np.concatenate([partitions[y] for y in range(2014, validation_year)])
    saved = pd.read_parquet(folder / 'training_age.parquet')
    np.testing.assert_array_equal(pd.DatetimeIndex(saved.timestamp).asi8,
                                  panel['timestamps'][origins].asi8)
    np.testing.assert_array_equal(saved.age_hours, age[origins])
    assert not saved.timestamp.duplicated().any()
    assert float(age[origins].max()) == fold['age_training_max_hours']
    cuts = [int(origins.max()) + 1, int(partitions[validation_year].max()) + 1,
            int(partitions[validation_year + 1].max()) + 1]
    for stop in cuts:
        np.testing.assert_array_equal(causal_recovery_age(aqi[:stop], particle[:stop]), age[:stop])
        changed_aqi, changed_particle = aqi.copy(), particle.copy()
        changed_aqi[stop:] = 50.
        changed_particle[stop:] = False
        np.testing.assert_array_equal(causal_recovery_age(changed_aqi, changed_particle)[:stop], age[:stop])
    boundary = pd.Timestamp(f'{validation_year}-01-01', tz='Asia/Shanghai')
    assert panel['timestamps'][origins.max() + 72] < boundary
    return {'training_age_rows': len(saved), 'max_training_age_hours': int(age[origins].max()),
            'causal_prefix_checks': len(cuts), 'training_targets_before_validation': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root / args.run
    report = read_json(run / 'report.json')
    assert report['decision'] in ('development_candidate_only', 'reject')
    assert set(report['folds']) == {'2022', '2023'}
    assert not (run / 'validation_2024').exists()
    control = root / report['control']
    inputs = root / report['configuration']['data']
    deployment = root / 'artifacts/cnemc_2025_deployment_candidate'
    check_hashes(control, report['control_sha256'])
    check_hashes(inputs, report['inputs_before'])
    check_hashes(deployment, report['deployment_before'])
    results, age_checks, independent_pairs = {}, {}, []
    for validation_year in (2022, 2023):
        year = validation_year + 1
        candidate = report['folds'][str(validation_year)]
        candidate_folder = run / f'validation_{validation_year}'
        base_folder = control / f'validation_{validation_year}'
        baseline = read_json(base_folder / 'report.json')
        assert read_json(candidate_folder / 'report.json') == candidate
        assert candidate['configuration']['auxiliary_weight'] == baseline['configuration']['auxiliary_weight'] == .1
        for key in ('seed', 'learning_rate', 'batch_size', 'epochs', 'data', 'history', 'horizon', 'validation_year'):
            assert baseline['configuration'][key] == candidate['configuration'][key], key
        assert candidate['validation_year'] == validation_year
        assert candidate['test_years'] == baseline['test_years'] == [year]
        assert set(candidate['years']) == {str(year)}
        assert candidate['age_definition'] == 'causal_three_polluted_confirm_six_clean_reset_log1p_scale168'
        for key in ('nodes', 'feature_names', 'coordinates_lon_lat', 'scaler', 'training_windows_by_year',
                    'partitions', 'training_label_audit', 'validation_label_audit'):
            assert candidate[key] == baseline[key], key
        panel, city, labels, eligible, partitions = load_experiment(SimpleNamespace(**candidate['configuration']))
        np.testing.assert_array_equal(panel['scaler'].mean, candidate['scaler']['mean'])
        np.testing.assert_array_equal(panel['scaler'].scale, candidate['scaler']['scale'])
        assert panel['nodes'] == candidate['nodes']
        age_checks[str(validation_year)] = check_age(candidate_folder, candidate, panel, city, labels,
                                                    partitions, validation_year)
        identities, scores = [], {}
        for arm, folder, fold in (('control', base_folder, baseline), ('age', candidate_folder, candidate)):
            check_selection(folder, fold)
            identity, metrics = check_predictions(folder, year, fold, panel, city, labels, eligible, partitions)
            identities.append(identity)
            scores[arm] = metrics
        pd.testing.assert_frame_equal(*identities)
        results[str(validation_year)] = scores
        independent_pairs.append({ARMS[0]: scores['control'], ARMS[1]: scores['age']})
    verified_gate = independent_gate(independent_pairs)
    gate = report['gate']
    assert verified_gate['accepted'] == gate['accepted']
    assert verified_gate['independent_tail_events'] == gate['independent_tail_events']
    for generic, saved_arm in ((ARMS[0], 'legacy'), (ARMS[1], 'balanced')):
        for metric in METRICS:
            np.testing.assert_allclose(verified_gate['annual_means'][generic][metric],
                                       gate['annual_equal_means'][saved_arm][metric], atol=1e-12)
    assert (report['decision'] == 'development_candidate_only') == verified_gate['accepted']
    check_hashes(control, report['control_sha256'])
    check_hashes(inputs, report['inputs_before'])
    check_hashes(deployment, report['deployment_before'])
    means = {arm: verified_gate['annual_means'][generic]
             for arm, generic in (('control', ARMS[0]), ('age', ARMS[1]))}
    summary = {'verification': 'PASS', 'folds': results, 'age_checks': age_checks,
               'annual_means': means, 'gate_accepted': verified_gate['accepted'],
               'independent_tail_events': verified_gate['independent_tail_events'],
               'inputs_unchanged': True, 'control_unchanged': True, 'deployment_unchanged': True,
               'limitations': ['saved_logits_recomputed_without_checkpoint_forward',
                               'initial_weights_not_saved_same_seed_and_configuration_checked',
                               'age_is_rule_memory_not_continuous_observation_guarantee',
                               'development_years_only_no_2025_2026_evaluation']}
    (run / 'independent_verification.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'verification': 'PASS', 'annual_means': means,
                      'gate_accepted': verified_gate['accepted']}, indent=2))


if __name__ == '__main__':
    main()
