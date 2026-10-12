"""Two predeclared historical years, with no first-year stopping or tuning."""

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.interval_endpoint_regression import (
    expand_interval_endpoints,
    make_endpoint_model,
)
from scripts.benchmark_observed_business_aft import ROOT, sha

YEARS = (2018, 2019)
MEMBER_COLUMNS = ['timestamp', 'continuity_block', 'label_available_at']


def select_frozen_year(data, frozen_predictions, members, year):
    cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
    expected = data.loc[data.timestamp.ge(cut) & data.timestamp.lt(end)].copy()
    valid = frozen_predictions.loc[
        frozen_predictions.timestamp.ge(cut) & frozen_predictions.timestamp.lt(end)
    ].copy()
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True), valid[data.columns].reset_index(drop=True),
    )
    train = data.loc[
        data.timestamp.lt(cut) & data.label_available_at.lt(cut)
        & ~data.continuity_block.isin(valid.continuity_block)
    ].copy()
    if train.empty or valid.empty:
        raise ValueError('Require nonempty fixed training and evaluation cohorts')
    pd.testing.assert_frame_equal(
        train[MEMBER_COLUMNS].reset_index(drop=True), members.reset_index(drop=True),
    )
    return train, valid


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def save_native_model(model, path):
    # Python handles Windows Unicode paths that the native file writer rejects.
    path.write_text(model.booster_.model_to_string(), encoding='utf-8')


def main():
    from haze_forecast.paired_interval_score import paired_error_bounds, paired_error_summary

    source = ROOT / 'artifacts/task01_recovery_head/transition_earlier_recovery_v1'
    early = ROOT / 'artifacts/task01_recovery_head/early_temporal_predictions_v1'
    out = ROOT / 'artifacts/task01_recovery_head/endpoint_l1_2018_2019_v2'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / '2020_training.parquet', source / 'protocol.json',
             early / 'protocol.json', early / 'predictions.parquet', Path(__file__)]
    paths += [early / f'{year}_{suffix}' for year in YEARS
              for suffix in ('model.joblib', 'training_members.parquet')]
    paths += [ROOT / f'src/haze_forecast/{name}.py' for name in (
        'training', 'observed_aft', 'cohort_scorecard', 'interval_endpoint_regression',
        'paired_interval_score',
    )]
    paths += [ROOT / 'docs/superpowers/specs/2026-09-12-endpoint-l1-design.md',
              ROOT / 'scripts/benchmark_observed_business_aft.py',
              ROOT / 'scripts/benchmark_2024_2025_model_families.py']
    deployed = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    paths += list(deployed.glob('*.joblib')) + list(deployed.glob('*.json'))
    before = {str(p): sha(p) for p in paths}
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    data = pd.read_parquet(source / '2020_training.parquet')
    frozen = pd.read_parquet(early / 'predictions.parquet')
    splits = {
        year: select_frozen_year(data, frozen, pd.read_parquet(
            early / f'{year}_training_members.parquet'), year)
        for year in YEARS
    }
    protocol = {
        'created_at': datetime.now(UTC).isoformat(),
        'years': list(YEARS), 'features': names, 'input_sha256': before,
        'versions': {name: version(name) for name in (
            'lightgbm', 'xgboost', 'numpy', 'pandas', 'pyarrow', 'scikit-learn', 'joblib',
        )},
        'parameters': {str(year): make_endpoint_model(len(train)).get_params()
                       for year, (train, _) in splits.items()},
        'target': 'half-weight lower/upper endpoints; interval absolute distance plus constant',
        'sampling': 'no bagging/GOSS; histogram bin construction uses every endpoint row',
        'prediction': 'native hours, projected to nonnegative; preserve raw predictions',
        'gate': 'each year: row and process exact MAE lower; tail24 process MAE no higher',
        'execution': 'run both years regardless of first-year outcome; no search or early stopping',
        'retry': 'v1 stopped at native Unicode-path export before any validation scoring; '
                 'v2 changes export I/O only, with identical parameters and cohorts',
        'limitations': [
            'Previously used historical development, not untouched testing or deployment',
            'Source cohort already conditioned on label maturity before 2020',
            'Learner, objective and optimization change together; not loss-only causal evidence',
            'Interval-distance minima do not identify exact truth inside a wide interval',
            'No complete early-year forecast support; cannot infer GEFS utility from this run',
            'Paired aggregate bounds are conservative, not necessarily sharp for shared event truth',
            'Passing both years would only pass this screen, not authorize deployment',
        ],
    }
    out.mkdir()
    write_json(out / 'protocol.json', protocol)
    reports = {}
    for year, (train, valid) in splits.items():
        baseline = joblib.load(early / f'{year}_model.joblib').predict(valid[names])
        np.testing.assert_array_equal(baseline, valid.temporal_prediction)
        control_score = score_predictions(valid, baseline)
        # Native missing values and paired weights are preserved through fitting.
        features, labels, weights = expand_interval_endpoints(
            train[names], train.lower_bound_hours, train.upper_bound_hours,
        )
        model = make_endpoint_model(len(train))
        model.fit(features, labels, sample_weight=weights)
        model_path = out / f'{year}_model.joblib'
        joblib.dump(model, model_path)
        save_native_model(model, out / f'{year}_model.txt')
        raw = model.predict(valid[names].astype(float))
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite predictions: do not drop evaluation origins')
        np.testing.assert_array_equal(
            raw, joblib.load(model_path).predict(valid[names].astype(float)),
        )
        prediction = np.maximum(raw, 0.0)
        candidate_score = score_predictions(valid, prediction)
        paired = paired_error_bounds(valid, baseline, prediction)
        result = valid.copy()
        result['endpoint_l1_raw'] = raw
        result['endpoint_l1_prediction'] = prediction
        result['paired_error_delta_lower'] = paired.lower.to_numpy()
        result['paired_error_delta_upper'] = paired.upper.to_numpy()
        result.to_parquet(out / f'{year}_predictions.parquet', index=False)
        train[MEMBER_COLUMNS].to_parquet(out / f'{year}_training_members.parquet', index=False)
        checks = {key: candidate_score[key] < control_score[key]
                  for key in ('exact_row_mae', 'exact_block_mae')}
        checks['tail24_block_mae'] = (
            candidate_score['tail24_block_mae'] <= control_score['tail24_block_mae']
        )
        reports[str(year)] = {
            'training_rows': len(train), 'endpoint_rows': len(labels),
            'training_blocks': int(train.continuity_block.nunique()),
            'training_start': train.timestamp.min().isoformat(),
            'training_end': train.timestamp.max().isoformat(),
            'latest_training_label_maturity': train.label_available_at.max().isoformat(),
            'all_missing_training_features': train[names].columns[
                train[names].isna().all()].tolist(),
            'negative_raw_predictions': int((raw < 0).sum()),
            'baseline': control_score, 'candidate': candidate_score,
            'paired_difference': paired_error_summary(valid, baseline, prediction),
            'checks': checks, 'passed': all(checks.values()),
        }
        write_json(out / f'{year}_decision.json', reports[str(year)])
        print(json.dumps({str(year): reports[str(year)]}, indent=2, allow_nan=False), flush=True)
    if before != {str(p): sha(p) for p in paths}:
        raise RuntimeError('An input or deployed artifact changed during the experiment')
    report = {
        'years': reports, 'both_years_passed': all(r['passed'] for r in reports.values()),
        'deployment_authorized': False,
        'output_sha256': {p.name: sha(p) for p in out.iterdir()},
    }
    write_json(out / 'report.json', report)


if __name__ == '__main__':
    main()
