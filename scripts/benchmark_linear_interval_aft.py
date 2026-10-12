"""Frozen two-year linear AFT comparison with an independent convergence gate."""

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.linear_interval_aft import LinearIntervalAFT, linear_state
from haze_forecast.paired_interval_score import paired_error_bounds, paired_error_summary
from scripts.benchmark_endpoint_l1 import select_frozen_year, write_json
from scripts.benchmark_observed_business_aft import ROOT, sha

YEARS = (2018, 2019)
MEMBERS = ['timestamp', 'continuity_block', 'label_available_at']


def solver_gate(final, states, objectives):
    if (len(states) != 51 or len(objectives) != 51
            or [s['iteration'] for s in states] != list(range(949, 1000))):
        raise ValueError('Require all 51 predeclared final optimization states')
    parameters = np.array([s['weights'] + [s['intercept']] for s in states], dtype=float)
    objectives = np.asarray(objectives, dtype=float)
    gradient = np.asarray(final['gradient_inf_norm'], dtype=float)
    finite = bool(np.isfinite(parameters).all() and np.isfinite(objectives).all()
                  and np.isfinite(gradient))
    change = float(np.abs(parameters - parameters[-1]).max()) if finite else None
    objective_change = float(np.ptp(objectives)) if finite else None
    return {
        'finite': finite,
        'parameter_change': change,
        'objective_change': objective_change,
        'gradient_inf_norm': float(final['gradient_inf_norm']) if finite else None,
        'converged': bool(finite and change <= 1e-4 and objective_change <= 1e-5
                          and final['gradient_inf_norm'] <= 1e-3),
    }


def optimization_audit(model, train, names):
    from haze_forecast.lognormal_aft_audit import linear_aft_diagnostics

    matrix = model.transform(train[names]).to_numpy()
    lo, hi = train.lower_bound_hours.to_numpy(), train.upper_bound_hours.to_numpy()
    margin = model.predict_margin(train[names])
    weights, intercept = linear_state(model.booster)
    reconstructed = matrix @ weights + intercept
    np.testing.assert_allclose(margin, reconstructed, atol=1e-5, rtol=1e-5)
    final = linear_aft_diagnostics(matrix, lo, hi, margin, weights)
    trace, objectives = [], []
    for state in model.training_trace:
        state_margin = matrix @ np.asarray(state['weights']) + state['intercept']
        diagnostics = linear_aft_diagnostics(matrix, lo, hi, state_margin, state['weights'])
        trace.append({**state, 'independent': diagnostics})
        objectives.append(diagnostics['penalized_objective'])
    return margin, {
        'final': final, 'last_51_states': trace,
        'gate': solver_gate(final, model.training_trace, objectives),
        'max_native_margin_reconstruction_difference': float(np.max(abs(margin - reconstructed))),
    }


def main():
    source = ROOT / 'artifacts/task01_recovery_head/transition_earlier_recovery_v1'
    early = ROOT / 'artifacts/task01_recovery_head/early_temporal_predictions_v1'
    out = ROOT / 'artifacts/task01_recovery_head/linear_interval_aft_2018_2019_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / '2020_training.parquet', source / 'protocol.json',
             early / 'protocol.json', early / 'predictions.parquet', Path(__file__)]
    paths += [early / f'{year}_{suffix}' for year in YEARS
              for suffix in ('model.joblib', 'training_members.parquet')]
    paths += [ROOT / f'src/haze_forecast/{name}.py' for name in (
        'training', 'observed_aft', 'cohort_scorecard', 'paired_interval_score',
        'linear_interval_aft', 'lognormal_aft_audit', 'interval_endpoint_regression',
    )]
    paths += [ROOT / 'scripts/benchmark_endpoint_l1.py',
              ROOT / 'scripts/benchmark_observed_business_aft.py',
              ROOT / 'docs/superpowers/specs/2026-09-12-linear-interval-aft-design.md']
    deployed = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    paths += list(deployed.glob('*.joblib')) + list(deployed.glob('*.json'))
    hashes = {str(p): sha(p) for p in paths}
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    data = pd.read_parquet(source / '2020_training.parquet')
    frozen = pd.read_parquet(early / 'predictions.parquet')
    splits = {
        year: select_frozen_year(data, frozen, pd.read_parquet(
            early / f'{year}_training_members.parquet'), year)
        for year in YEARS
    }
    protocol = {
        'created_at': datetime.now(UTC).isoformat(), 'years': list(YEARS),
        'features': names, 'parameters': LinearIntervalAFT.parameters(), 'rounds': 1000,
        'input_sha256': hashes,
        'versions': {name: version(name) for name in (
            'xgboost', 'numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'joblib', 'lightgbm',
        )},
        'preprocessing': 'fit on training only: drop all-missing, median with missing indicators; '
                         'drop constant and exactly duplicated imputed vectors; standard scale',
        'solver_gate': {'gradient_inf_norm_max': 1e-3, 'last_50_parameter_change_max': 1e-4,
                        'last_50_penalized_objective_change_max': 1e-5},
        'performance_gate': 'each year exact row and process MAE lower; tail24 process MAE no higher',
        'execution': 'both years, no early stopping, no parameter or year search',
        'limitations': [
            'Historical development population already conditioned on maturity before 2020',
            'Not new held-out data, not deployed-model or online accuracy',
            'Change of representation, regularization and missing handling is not a pure capacity ablation',
            'Early years lack complete GEFS support; process blocks are not proven independent',
            'gblinear is deprecated; frozen XGBoost 3.4.1 research use only',
            'Solver failure is not evidence that the whole linear model family is ineffective',
        ],
    }
    out.mkdir()
    write_json(out / 'protocol.json', protocol)
    reports = {}
    for year, (train, valid) in splits.items():
        baseline = joblib.load(early / f'{year}_model.joblib').predict(valid[names])
        np.testing.assert_array_equal(baseline, valid.temporal_prediction)
        control = score_predictions(valid, baseline)
        model = LinearIntervalAFT().fit_bounds(train[names], train.lower_bound_hours,
                                              train.upper_bound_hours)
        model_path = out / f'{year}_model.joblib'
        joblib.dump(model, model_path)
        (out / f'{year}_model.json').write_bytes(model.booster.save_raw(raw_format='json'))
        prediction, margin = model.predict(valid[names]), model.predict_margin(valid[names])
        restored = joblib.load(model_path)
        np.testing.assert_array_equal(prediction, restored.predict(valid[names]))
        np.testing.assert_array_equal(margin, restored.predict_margin(valid[names]))
        np.testing.assert_allclose(prediction, np.exp(margin), rtol=2e-6)
        with threadpool_limits(limits=1):
            train_margin, optimizer = optimization_audit(model, train, names)
        write_json(out / f'{year}_optimizer.json', optimizer)
        training_output = train[MEMBERS + ['lower_bound_hours', 'upper_bound_hours']].copy()
        training_output['linear_margin'] = train_margin
        training_output.to_parquet(out / f'{year}_training_audit.parquet', index=False)
        transform = {
            'original_features': model.feature_names,
            'all_missing_features': model.all_missing_features,
            'observed_features': model.observed_features,
            'kept_features': model.kept_features,
            'dropped_design_features': model.dropped_design_features,
            'imputer_statistics': model.imputer.statistics_.tolist(),
            'keep_indices': model.keep_indices.tolist(),
            'scaler_mean': model.scaler.mean_.tolist(),
            'scaler_scale': model.scaler.scale_.tolist(),
        }
        write_json(out / f'{year}_preprocessing.json', transform)
        candidate = score_predictions(valid, prediction)
        paired = paired_error_bounds(valid, baseline, prediction)
        result = valid.assign(linear_prediction=prediction, linear_margin=margin,
                              paired_error_delta_lower=paired.lower.to_numpy(),
                              paired_error_delta_upper=paired.upper.to_numpy())
        result.to_parquet(out / f'{year}_predictions.parquet', index=False)
        checks = {key: candidate[key] < control[key]
                  for key in ('exact_row_mae', 'exact_block_mae')}
        checks['tail24_block_mae'] = candidate['tail24_block_mae'] <= control['tail24_block_mae']
        reports[str(year)] = {
            'training_rows': len(train), 'training_blocks': int(train.continuity_block.nunique()),
            'linear_design_columns': len(model.kept_features),
            'training_start': train.timestamp.min().isoformat(),
            'training_end': train.timestamp.max().isoformat(),
            'baseline': control, 'candidate': candidate,
            'paired_difference': paired_error_summary(valid, baseline, prediction),
            'solver': optimizer['gate'], 'performance_checks': checks,
            'passed': bool(optimizer['gate']['converged'] and all(checks.values())),
        }
        write_json(out / f'{year}_decision.json', reports[str(year)])
        print(json.dumps({str(year): reports[str(year)]}, indent=2, allow_nan=False), flush=True)
    if hashes != {str(p): sha(p) for p in paths}:
        raise RuntimeError('Input or deployed artifact changed during experiment')
    write_json(out / 'report.json', {
        'years': reports, 'both_years_passed': all(r['passed'] for r in reports.values()),
        'deployment_authorized': False,
        'output_sha256': {p.name: sha(p) for p in out.iterdir()},
    })


if __name__ == '__main__':
    main()
