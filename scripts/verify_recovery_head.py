"""Independently recompute saved recovery-head predictions and paired errors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.benchmark_recovery_head import RecoveryWindows, load_experiment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors(frame):
    selected = frame.loc[frame.label_verified_in_scoring_window & frame.actual_hours.notna()].copy()
    predicted = np.minimum(np.nan_to_num(selected.predicted_hours.to_numpy(), nan=68), 68)
    selected['error'] = np.abs(np.minimum(selected.actual_hours, 68) - predicted)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = json.loads((args.run / 'report.json').read_text('utf-8'))
    frozen = json.loads((args.run / 'frozen_before_evaluation.json').read_text('utf-8'))
    assert digest(args.run / 'recovery_head_state.pt') == frozen['checkpoint_sha256'] == report['checkpoint_sha256']
    assert digest(args.run / 'selection.json') == frozen['selection_sha256'] == report['selection_sha256']
    for directory, mapping in ((root / report['configuration']['data'], report['inputs_before']),
                                (root / report['configuration']['reference'], report['reference_before']),
                                (root / 'artifacts/cnemc_2025_deployment_candidate', report['deployment_before'])):
        for name, expected in mapping.items():
            assert digest(directory / name) == expected, name
    reference = root / report['configuration']['reference']
    configuration = SimpleNamespace(**report['configuration'])
    panel, city, _, eligible, partitions = load_experiment(configuration)
    training = RecoveryWindows(panel, np.concatenate([partitions[year] for year in range(2014, 2024)]),
                               24, 72, city.aqi.to_numpy(), eligible)
    at_risk = (training.risk_mask & training.eligible[:, None]).sum(axis=0)
    event_counts = (training.hazard * training.eligible[:, None]).sum(axis=0)
    prior_hazard = np.divide(event_counts, at_risk, out=np.zeros(67), where=at_risk > 0)
    prior_survival = np.cumprod(1 - prior_hazard)
    prior_median = int((1 - prior_survival < .5).sum() + 1)
    result = {'verification': 'PASS', 'training_prior_median_hours': prior_median,
              'training_prior_recovery_probability': float(1 - prior_survival[-1]), 'years': {}}
    event_frames = []
    for year in ('2025', '2026'):
        current = pd.read_parquet(args.run / f'recovery_head_{year}_recovery.parquet')
        baseline = pd.read_parquet(reference / f'pm25_gnn_history_{year}_recovery.parquet')
        aft = pd.read_parquet(reference / f'matched_aft_{year}_recovery.parquet')
        for key in ('timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window'):
            pd.testing.assert_series_equal(current[key], baseline[key])
            pd.testing.assert_series_equal(current[key], aft[key])
        arrays = np.load(args.run / f'recovery_head_{year}_distribution.npz')
        np.testing.assert_array_equal(pd.DatetimeIndex(current.timestamp).asi8, arrays['origin_ns'])
        hazards = 1 / (1 + np.exp(-arrays['logits'].astype(np.float64)))
        survival = np.cumprod(1 - hazards, axis=1)
        probability = np.concatenate([np.concatenate([np.ones((len(hazards), 1)), survival[:, :-1]], axis=1)
                                      * hazards, survival[:, -1:]], axis=1)
        np.testing.assert_allclose(probability, arrays['probability'], atol=2e-7, rtol=2e-5)
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-7)
        medians = (np.cumsum(probability, axis=1) < .5).sum(axis=1) + 1
        np.testing.assert_array_equal(medians, current.predicted_hours)
        evaluated, old, matched = errors(current), errors(baseline), errors(aft)
        scores = report['years'][year]['duration']
        np.testing.assert_allclose(evaluated.error.mean(), scores['restricted_time_mae'], atol=1e-12)
        event_mean = evaluated.groupby('event_id').error.mean()
        np.testing.assert_allclose(event_mean.mean(), scores['event_equal_restricted_time_mae'], atol=1e-12)
        event = evaluated.groupby('event_id').error.agg(['sum', 'count', 'mean'])
        event['baseline_mean'] = old.groupby('event_id').error.mean()
        event['baseline_sum'] = old.groupby('event_id').error.sum()
        event['year'] = year
        event_frames.append(event.reset_index())
        rng = np.random.default_rng(42)
        sample = rng.integers(0, len(event), (10000, len(event)))
        delta = (event['sum'].to_numpy()[sample].sum(axis=1)
                 - event.baseline_sum.to_numpy()[sample].sum(axis=1)) / event['count'].to_numpy()[sample].sum(axis=1)
        long = evaluated.loc[evaluated.actual_hours.ge(68)]
        prior_error = np.abs(np.minimum(evaluated.actual_hours, 68) - prior_median)
        result['years'][year] = {
            'rows': len(evaluated), 'events': len(event),
            'candidate_mae': evaluated.error.mean(), 'baseline_mae': old.error.mean(),
            'matched_aft_mae': matched.error.mean(),
            'matched_aft_event_equal': matched.groupby('event_id').error.mean().mean(),
            'relative_improvement_pct': 100 * (old.error.mean() - evaluated.error.mean()) / old.error.mean(),
            'events_improved': int((event['mean'] < event.baseline_mean).sum()),
            'descriptive_event_bootstrap_delta_ci95': np.quantile(delta, [.025, .975]).tolist(),
            'nonrecovering_rows': len(long), 'nonrecovering_mae': long.error.mean(),
            'always_recovered_brier': float((evaluated.actual_hours.ge(68)).mean()),
            'training_prior_mae': float(prior_error.mean()),
            'training_prior_event_equal_mae': float(prior_error.groupby(evaluated.event_id).mean().mean()),
        }
    (args.run / 'independent_verification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pd.concat(event_frames).to_csv(args.run / 'independent_event_errors.csv', index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
