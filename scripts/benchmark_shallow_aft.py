"""Fixed depth-2 versus depth-4 AFT; no feature or interaction changes."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary
from scripts.benchmark_gefs_72h_aft import verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, sha


def shallow_params(original):
    if original.get('max_depth') != 4 or 'interaction_constraints' in original:
        raise ValueError('Expected unconstrained depth-4 baseline')
    return {**original, 'max_depth': 2}


def tree_depth(tree):
    if 'leaf' in tree:
        return 0
    return 1 + max(tree_depth(child) for child in tree['children'])


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    previous = base / 'grouped_forecast_aft_2023_20260916'
    old = base / 'reusable_gefs_aft_v1'
    out = base / 'shallow_aft_2023_20260916'
    if out.exists():
        raise FileExistsError(out)
    frozen = json.loads((previous / 'protocol.json').read_bytes())
    report = json.loads((previous / 'report.json').read_bytes())
    inputs = dict(frozen['input_sha256'])
    for name, digest in report['output_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Invalid frozen filename')
        inputs[str(previous / name)] = digest
    verify_hashes(inputs)
    dependencies = [Path(__file__).resolve(), previous / 'report.json',
                    ROOT / 'src/haze_forecast/paired_interval_score.py']
    inputs.update({str(p): sha(p) for p in dependencies})
    names, params = frozen['features'], frozen['params']
    candidate_params = shallow_params(params)
    train = pd.read_parquet(base / 'gefs_2021_extension_v1/f024_training_rows.parquet')
    val = pd.read_parquet(base / 'gefs_2021_extension_v1/inference_rows.parquet')
    cohorts = {key: pd.read_parquet(previous / f'{key}.parquet')
               for key in ('original', 'expanded')}
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    if not (train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()):
        raise ValueError('Training crosses cutoff')
    if (not val.timestamp.is_unique or not train.timestamp.is_unique
            or not val.timestamp.ge(cutoff).all()
            or not val.timestamp.lt(pd.Timestamp('2024-01-01', tz='Asia/Shanghai')).all()):
        raise ValueError('Invalid origin timestamps')
    for cohort in cohorts.values():
        if set(train.continuity_block) & set(cohort.continuity_block):
            raise ValueError('Training/evaluation block overlap')
    matrix = xgb.DMatrix(train[names].astype(float), missing=np.nan)
    matrix.set_float_info('label_lower_bound', train.lower_bound_hours)
    matrix.set_float_info('label_upper_bound', train.upper_bound_hours)
    inference = xgb.DMatrix(val[names].astype(float), missing=np.nan)
    baseline = joblib.load(old / 'model.joblib').predict(val[names])
    indexed_baseline = pd.Series(baseline, index=val.timestamp)
    for cohort in cohorts.values():
        np.testing.assert_array_equal(indexed_baseline.loc[cohort.timestamp], cohort.forecast_aft)
    manifest = {'input_sha256': inputs, 'features': names,
                'baseline_params': params, 'candidate_params': candidate_params, 'rounds': 500,
                'training_rows': len(train), 'inference_rows': len(val),
                'gate': 'All three exact MAEs strictly lower in both frozen 2023 cohorts',
                'selection': 'Only depth 2; stop if failed, no depth search or later-year rescue',
                'label_policy': frozen['label_policy'], 'deployment_authorized': False,
                'limitations': ['Reused development year, not untouched test data',
                                'No new real-time availability evidence',
                                'Paired bounds are label-compatible bounds, not confidence intervals',
                                'Original and expanded cohorts overlap'],
                'xgboost_version': xgb.__version__}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    control = xgb.train(params, matrix, num_boost_round=500)
    np.testing.assert_array_equal(control.predict(inference), baseline)
    candidate = xgb.train(candidate_params, matrix, num_boost_round=500)
    depths = [tree_depth(json.loads(tree)) for tree in candidate.get_dump(dump_format='json')]
    if len(depths) != 500 or max(depths) > 2:
        raise ValueError('Candidate tree depth does not match protocol')
    candidate.save_model(out / 'candidate.ubj')
    restored = xgb.Booster()
    restored.load_model(out / 'candidate.ubj')
    prediction = candidate.predict(inference)
    np.testing.assert_array_equal(prediction, restored.predict(inference))
    all_predictions = val[['timestamp']].copy()
    all_predictions['baseline'] = baseline
    all_predictions['candidate'] = prediction
    all_predictions.to_parquet(out / 'all_predictions.parquet', index=False)
    indexed = pd.Series(prediction, index=val.timestamp)
    scores, checks, paired = {}, {}, {}
    for key, frame in cohorts.items():
        frame['candidate'] = indexed.loc[frame.timestamp].to_numpy()
        scores[key] = {arm: score_predictions(frame, frame[column].to_numpy()) for arm, column in (
            ('baseline', 'forecast_aft'), ('candidate', 'candidate'))}
        paired[key] = paired_error_summary(frame, frame.forecast_aft, frame.candidate)
        checks[key] = all(scores[key]['candidate'][name] < scores[key]['baseline'][name]
                          for name in ('exact_row_mae', 'exact_block_mae', 'tail24_block_mae'))
        frame.to_parquet(out / f'{key}.parquet', index=False)
    verify_hashes(inputs)
    result = {'scores': scores, 'paired': paired, 'checks': checks, 'passed': all(checks.values()),
              'max_tree_depth': max(depths), 'trees': len(depths),
              'baseline_retraining_exact_match': True, 'checkpoint_replay_exact_match': True,
              'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
