"""A fixed two-feature increment to the frozen 104-feature AFT baseline."""

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from haze_forecast.paired_interval_score import paired_error_bounds, paired_error_summary
from scripts.benchmark_endpoint_l1 import select_frozen_year, write_json
from scripts.benchmark_observed_business_aft import ROOT, sha

YEARS = (2018, 2019)
RUN_FEATURES = ['observed_particle_run_hours', 'particle_run_start_observed']
INPUT_COLUMNS = ['timestamp', 'aqi', 'pm2_5_iaqi', 'pm10_iaqi']
MEMBERS = ['timestamp', 'continuity_block', 'label_available_at']


def attach_run_features(frame, run):
    if (frame.empty or frame.timestamp.isna().any() or not frame.timestamp.is_unique
            or run.timestamp.isna().any() or not run.timestamp.is_unique):
        raise ValueError('Require nonempty unique nonmissing forecast origins and run timestamps')
    if set(RUN_FEATURES) & set(frame.columns):
        raise ValueError('New feature names must not overwrite any frozen fields')
    if not frame.timestamp.isin(run.timestamp).all():
        raise ValueError('Every forecast origin must have run features; do not drop rows')
    values = run.set_index('timestamp').loc[frame.timestamp, RUN_FEATURES].to_numpy(float)
    if (not np.isfinite(values).all() or (values[:, 0] < 1).any()
            or (values[:, 0] % 1 != 0).any() or not np.isin(values[:, 1], [0., 1.]).all()):
        raise ValueError('Eligible origins require valid positive observed length and binary boundary')
    result = frame.copy()
    result[RUN_FEATURES] = values
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    return result


def main():
    from haze_forecast.observed_pollution_run import build_observed_pollution_run

    base = ROOT / 'artifacts/task01_recovery_head'
    source, early = base / 'transition_earlier_recovery_v1', base / 'early_temporal_predictions_v1'
    hourly_path = base / 'observed_business_dataset_v1/features.parquet'
    out = base / 'observed_run_age_2018_2019_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / '2020_training.parquet', source / 'protocol.json', hourly_path,
             early / 'protocol.json', early / 'predictions.parquet', Path(__file__)]
    paths += [early / f'{year}_{suffix}' for year in YEARS
              for suffix in ('model.joblib', 'training_members.parquet')]
    paths += [ROOT / f'src/haze_forecast/{name}.py' for name in (
        'observed_pollution_run', 'observed_business', 'observed_hourly_features',
        'observed_recovery_labels', 'features', 'labels', 'observed_aft', 'training',
        'cohort_scorecard', 'paired_interval_score', 'interval_endpoint_regression',
    )]
    paths += [ROOT / 'scripts/benchmark_endpoint_l1.py',
              ROOT / 'scripts/benchmark_observed_business_aft.py',
              ROOT / 'docs/superpowers/specs/2026-09-12-observed-run-age-design.md']
    deployed = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    paths += list(deployed.glob('*.joblib')) + list(deployed.glob('*.json'))
    before = {str(p): sha(p) for p in paths}
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    if len(names) != 104 or set(names) & set(RUN_FEATURES):
        raise ValueError('Require the frozen 104-feature baseline without either new feature')
    candidate_names = names + RUN_FEATURES
    data = pd.read_parquet(source / '2020_training.parquet')
    frozen = pd.read_parquet(early / 'predictions.parquet')
    splits = {
        year: select_frozen_year(data, frozen, pd.read_parquet(
            early / f'{year}_training_members.parquet'), year)
        for year in YEARS
    }
    hourly = pd.read_parquet(hourly_path, columns=INPUT_COLUMNS)
    hourly = hourly.loc[hourly.timestamp.lt(pd.Timestamp('2020-01-01', tz='Asia/Shanghai'))]
    run = build_observed_pollution_run(hourly)
    prefix_rows = {}
    for year in YEARS:
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        prefix = build_observed_pollution_run(hourly.loc[hourly.timestamp.lt(cut)])
        pd.testing.assert_frame_equal(prefix, run.loc[run.timestamp.lt(cut)])
        prefix_rows[str(year)] = len(prefix)
    legacy = pd.read_parquet(hourly_path, columns=['timestamp', 'pollution_age_hours'])
    paired_legacy = legacy.set_index('timestamp').loc[run.timestamp, 'pollution_age_hours'].to_numpy()
    confirmed = run.observed_particle_run_hours.gt(0).to_numpy()
    np.testing.assert_array_equal(run.observed_particle_run_hours.to_numpy()[confirmed],
                                  paired_legacy[confirmed])
    for train, valid in splits.values():
        attach_run_features(train, run)
        attach_run_features(valid, run)
    protocol = {
        'created_at': datetime.now(UTC).isoformat(), 'years': list(YEARS),
        'baseline_features': names, 'features': candidate_names, 'added_features': RUN_FEATURES,
        'input_sha256': before,
        'versions': {name: version(name) for name in ('xgboost', 'numpy', 'pandas', 'pyarrow',
                                                    'scikit-learn', 'lightgbm', 'joblib')},
        'learner': 'Unchanged ObservedAFTSurvivalRegressor.fit_bounds, normal, sigma 1.2',
        'seed': 42, 'rounds': 500, 'nthread': 1,
        'semantics': 'current uninterrupted confirmed-pollution observations, lower bound after '
                     'unknown/left boundary; start flag refers only to an observed hourly boundary',
        'feature_verification': {'real_prefix_rows': prefix_rows,
                                 'legacy_confirmed_age_rows_equal': int(confirmed.sum())},
        'gate': 'each year: exact row and process MAE lower; tail24 process MAE no higher',
        'execution': 'both years complete; no early stopping, age transform, bins or parameter search',
        'limitations': [
            'Existing long-history counters are retained; this is not first use of long history',
            'Previously used historical population conditioned on maturity before 2020',
            'Early years have no complete GEFS support; not a weather forecast usefulness test',
            'Same tree algorithm and parameters; adding columns also changes feature sampling',
            'Process blocks are not proven statistically independent',
            'No deployment authorization from this screen alone',
        ],
    }
    out.mkdir()
    write_json(out / 'protocol.json', protocol)
    run.to_parquet(out / 'hourly_run_features.parquet', index=False)
    reports = {}
    for year, (frozen_train, frozen_valid) in splits.items():
        train, valid = attach_run_features(frozen_train, run), attach_run_features(frozen_valid, run)
        baseline = joblib.load(early / f'{year}_model.joblib').predict(valid[names])
        np.testing.assert_array_equal(baseline, frozen_valid.temporal_prediction)
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
            train[candidate_names], train.lower_bound_hours, train.upper_bound_hours,
        )
        model_path = out / f'{year}_model.joblib'
        joblib.dump(model, model_path)
        (out / f'{year}_model.json').write_bytes(model.booster.save_raw(raw_format='json'))
        write_json(out / f'{year}_training_config.json', json.loads(model.booster.save_config()))
        prediction = model.predict(valid[candidate_names])
        np.testing.assert_array_equal(prediction, joblib.load(model_path).predict(valid[candidate_names]))
        control, candidate = score_predictions(valid, baseline), score_predictions(valid, prediction)
        paired = paired_error_bounds(valid, baseline, prediction)
        result = valid.assign(run_age_prediction=prediction,
                              paired_error_delta_lower=paired.lower.to_numpy(),
                              paired_error_delta_upper=paired.upper.to_numpy())
        result.to_parquet(out / f'{year}_predictions.parquet', index=False)
        train.to_parquet(out / f'{year}_training.parquet', index=False)
        train[MEMBERS].to_parquet(out / f'{year}_training_members.parquet', index=False)
        checks = {key: candidate[key] < control[key]
                  for key in ('exact_row_mae', 'exact_block_mae')}
        checks['tail24_block_mae'] = candidate['tail24_block_mae'] <= control['tail24_block_mae']
        reports[str(year)] = {
            'training_rows': len(train), 'training_blocks': int(train.continuity_block.nunique()),
            'training_start': train.timestamp.min().isoformat(),
            'training_end': train.timestamp.max().isoformat(),
            'training_unknown_start_rows': int(train.particle_run_start_observed.eq(0).sum()),
            'validation_unknown_start_rows': int(valid.particle_run_start_observed.eq(0).sum()),
            'baseline': control, 'candidate': candidate,
            'paired_difference': paired_error_summary(valid, baseline, prediction),
            'checks': checks, 'passed': all(checks.values()),
        }
        write_json(out / f'{year}_decision.json', reports[str(year)])
        print(json.dumps({str(year): reports[str(year)]}, indent=2, allow_nan=False), flush=True)
    if before != {str(p): sha(p) for p in paths}:
        raise RuntimeError('Input or deployed artifacts changed during experiment')
    write_json(out / 'report.json', {
        'years': reports, 'both_years_passed': all(r['passed'] for r in reports.values()),
        'deployment_authorized': False,
        'output_sha256': {p.name: sha(p) for p in out.iterdir()},
    })


if __name__ == '__main__':
    main()
