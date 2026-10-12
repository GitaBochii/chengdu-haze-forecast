"""A fixed two-expert distribution, with no validation-based update selection."""

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.interval_mixture_math import mixture_observation_terms
from haze_forecast.paired_interval_score import paired_error_bounds, paired_error_summary
from haze_forecast.two_expert_aft import TwoExpertAFT
from scripts.benchmark_endpoint_l1 import select_frozen_year, write_json
from scripts.benchmark_observed_business_aft import ROOT, sha

YEARS = (2018, 2019)
MEMBERS = ['timestamp', 'continuity_block', 'label_available_at']


def responsibility_support(blocks, responsibility):
    mass = pd.DataFrame(responsibility).groupby(np.asarray(blocks), sort=False).sum().to_numpy()
    total = mass.sum(axis=0)
    return {
        'effective_blocks': [float(total[k]**2 / np.sum(mass[:, k]**2)) if total[k] > 0 else None
                             for k in range(2)],
        'largest_block_fraction': [float(mass[:, k].max() / total[k]) if total[k] > 0 else None
                                   for k in range(2)],
    }


def save_states(model, frame, names, out, year):
    matrix = xgb.DMatrix(frame[names].astype(float), missing=np.nan)
    trace = []
    for index, state in enumerate(model.states):
        prefix = f'{year}_state_{index}'
        locations, gate = model.state_parameters(state, matrix)
        nll, responsibility = mixture_observation_terms(
            frame.lower_bound_hours, frame.upper_bound_hours, locations, gate)
        np.savez_compressed(out / f'{prefix}_training.npz', locations=locations, gate_margin=gate,
                            nll=nll, responsibilities=responsibility)
        np.testing.assert_allclose(np.mean(nll), model.trace[index]['mean_nll'], rtol=0, atol=1e-12)
        for component, expert in enumerate(state['experts']):
            (out / f'{prefix}_expert_{component}.json').write_bytes(expert.save_raw(raw_format='json'))
        if state['gate'] is not None:
            (out / f'{prefix}_gate.json').write_bytes(state['gate'].save_raw(raw_format='json'))
        write_json(out / f'{prefix}_metadata.json', {
            'offsets': state['offsets'], 'expert_training_configs': state['expert_training_configs'],
            'gate_training_config': state['gate_training_config'],
        })
        trace.append({**model.trace[index],
                      **responsibility_support(frame.continuity_block, responsibility)})
    write_json(out / f'{year}_trace.json', trace)
    return trace


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source, early = base / 'transition_earlier_recovery_v1', base / 'early_temporal_predictions_v1'
    out = base / 'two_expert_aft_2018_2019_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / '2020_training.parquet', source / 'protocol.json',
             early / 'protocol.json', early / 'predictions.parquet', Path(__file__)]
    paths += [early / f'{year}_{suffix}' for year in YEARS
              for suffix in ('model.joblib', 'training_members.parquet')]
    paths += [ROOT / f'src/haze_forecast/{name}.py' for name in (
        'two_expert_aft', 'interval_mixture_math', 'lognormal_aft_audit', 'observed_aft',
        'training', 'cohort_scorecard', 'paired_interval_score', 'interval_endpoint_regression',
    )]
    paths += [ROOT / 'scripts/benchmark_endpoint_l1.py',
              ROOT / 'scripts/benchmark_observed_business_aft.py',
              ROOT / 'docs/superpowers/specs/2026-09-12-two-expert-aft-design.md']
    deployed = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    paths += list(deployed.glob('*.joblib')) + list(deployed.glob('*.json'))
    before = {str(path): sha(path) for path in paths}
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    if len(names) != 104:
        raise ValueError('Require frozen 104-feature baseline')
    data = pd.read_parquet(source / '2020_training.parquet')
    frozen = pd.read_parquet(early / 'predictions.parquet')
    splits = {year: select_frozen_year(data, frozen, pd.read_parquet(
        early / f'{year}_training_members.parquet'), year) for year in YEARS}
    protocol = {
        'created_at': datetime.now(UTC).isoformat(), 'years': list(YEARS),
        'features': names, 'input_sha256': before,
        'versions': {name: version(name) for name in ('xgboost', 'numpy', 'pandas', 'pyarrow',
                                                    'scipy', 'scikit-learn', 'joblib')},
        'components': 2, 'scale': 1.2, 'expert_rounds': 500, 'expert_seed': 42,
        'initial_location_offsets': [-.6, .6], 'initial_gate_margin': 0.,
        'gate_parameters': TwoExpertAFT.gate_parameters(), 'gate_rounds': 100,
        'max_updates': 5, 'weighting': 'raw posterior responsibilities, no component rescaling',
        'acceptance': 'strictly lower full training observed NLL; reject and stop, no retry',
        'prediction': 'float64 stable mixture CDF median, 64 log-time bisections, no horizon cap',
        'gate': 'accepted_updates >= 1; each year exact row/process MAE lower, tail24 process no higher',
        'execution': 'both years complete, no validation-based checkpoint or parameter selection',
        'limitations': [
            'Added expert and gate capacity; not a pure causal isolation of multimodality',
            'Finite regularized trees approximate the M-step; no convergence or global optimum claim',
            'Historical population already conditioned on pre-2020 maturity, previously used years',
            'No complete early-year GEFS support; not a future-weather usefulness test',
            'Responsibility block effective size is descriptive, not an independent sample count',
            'Latent components do not identify physical regimes',
            'No deployment authorization from this experiment alone',
        ],
    }
    out.mkdir()
    write_json(out / 'protocol.json', protocol)
    reports = {}
    for year, (train, valid) in splits.items():
        baseline_model = joblib.load(early / f'{year}_model.joblib')
        baseline = baseline_model.predict(valid[names])
        np.testing.assert_array_equal(baseline, valid.temporal_prediction)
        model = TwoExpertAFT(max_updates=5).fit_bounds(
            train[names], train.lower_bound_hours, train.upper_bound_hours, baseline_model)
        trace = save_states(model, train, names, out, year)
        path = out / f'{year}_model.joblib'
        joblib.dump(model, path)
        prediction = model.predict(valid[names])
        np.testing.assert_array_equal(prediction, joblib.load(path).predict(valid[names]))
        control, candidate = score_predictions(valid, baseline), score_predictions(valid, prediction)
        paired = paired_error_bounds(valid, baseline, prediction)
        result = valid.assign(mixture_prediction=prediction,
                              paired_error_delta_lower=paired.lower.to_numpy(),
                              paired_error_delta_upper=paired.upper.to_numpy())
        result.to_parquet(out / f'{year}_predictions.parquet', index=False)
        train.to_parquet(out / f'{year}_training.parquet', index=False)
        train[MEMBERS].to_parquet(out / f'{year}_training_members.parquet', index=False)
        checks = {key: candidate[key] < control[key] for key in ('exact_row_mae', 'exact_block_mae')}
        checks['tail24_block_mae'] = candidate['tail24_block_mae'] <= control['tail24_block_mae']
        checks['accepted_update'] = model.accepted_updates >= 1
        reports[str(year)] = {
            'training_rows': len(train), 'training_blocks': int(train.continuity_block.nunique()),
            'training_start': train.timestamp.min().isoformat(),
            'training_end': train.timestamp.max().isoformat(),
            'baseline_training_nll': model.baseline_training_nll,
            'initial_mixture_training_nll': trace[0]['mean_nll'],
            'selected_training_nll': trace[model.selected_index]['mean_nll'],
            'attempted_updates': len(model.states) - 1, 'accepted_updates': model.accepted_updates,
            'selected_index': model.selected_index, 'stop_reason': model.stop_reason,
            'baseline': control, 'candidate': candidate,
            'paired_difference': paired_error_summary(valid, baseline, prediction),
            'checks': checks, 'passed': all(checks.values()),
        }
        write_json(out / f'{year}_decision.json', reports[str(year)])
        print(json.dumps({str(year): reports[str(year)]}, indent=2, allow_nan=False), flush=True)
    if before != {str(path): sha(path) for path in paths}:
        raise RuntimeError('An input or deployed artifact changed during the experiment')
    write_json(out / 'report.json', {
        'years': reports, 'both_years_passed': all(row['passed'] for row in reports.values()),
        'deployment_authorized': False,
        'output_sha256': {path.name: sha(path) for path in out.iterdir()},
    })


if __name__ == '__main__':
    main()
