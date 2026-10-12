"""One fixed 2023 AFT experiment: separate history and forecast interactions."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from scripts.benchmark_gefs_72h_aft import AGES, verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, sha


def feature_groups(names):
    if (len(names) != 104 or len(set(names)) != 104 or names[101:] != AGES
            or not all(n.startswith('gefs_') for n in names[92:101])):
        raise ValueError('Unexpected feature schema')
    return [names[:92], names[92:]]


def audit_paths(trees, groups):
    membership = {name: i for i, group in enumerate(groups) for name in group}
    leaves = 0
    for tree in trees:
        stack = [(tree, frozenset())]
        while stack:
            node, seen = stack.pop()
            if 'split' not in node:
                leaves += 1
                continue
            if node['split'] not in membership:
                raise ValueError('Unknown split feature')
            updated = seen | {membership[node['split']]}
            if len(updated) > 1:
                raise ValueError('Mixed history/forecast path')
            stack.extend((child, updated) for child in node['children'])
    return {'trees': len(trees), 'leaf_paths': leaves, 'mixed_paths': 0}


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    old = base / 'reusable_gefs_aft_v1'
    source = base / 'gefs_2021_extension_v1'
    out = base / 'grouped_forecast_aft_2023_20260916'
    if out.exists():
        raise FileExistsError(out)
    inputs = {}
    for folder in (old, source):
        report = json.loads((folder / 'report.json').read_bytes())
        for name, value in report['output_sha256'].items():
            if Path(name).name != name:
                raise ValueError('Invalid artifact name')
            inputs[str(folder / name)] = value
        inputs[str(folder / 'report.json')] = sha(folder / 'report.json')
    verify_hashes(inputs)
    names = json.loads((old / 'protocol.json').read_bytes())['features']
    groups = feature_groups(names)
    train = pd.read_parquet(source / 'f024_training_rows.parquet')
    val = pd.read_parquet(source / 'inference_rows.parquet')
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    if not (train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()):
        raise ValueError('Training crosses cutoff')
    if not (val.timestamp.ge(cutoff).all()
            and val.timestamp.lt(pd.Timestamp('2024-01-01', tz='Asia/Shanghai')).all()):
        raise ValueError('Wrong evaluation year')
    frames = {key: pd.read_parquet(old / filename) for key, filename in [
        ('original', 'predictions.parquet'), ('expanded', 'expanded_predictions.parquet')]}
    baseline = joblib.load(old / 'model.joblib').predict(val[names])
    label_path = base / 'interval_business_labels_v1/origins.parquet'
    if sha(label_path) != json.loads((label_path.parent / 'report.json').read_bytes())['output_sha256']:
        raise ValueError('Canonical label hash changed')
    labels = pd.read_parquet(label_path).set_index('timestamp')
    cohorts = {}
    for key, frame in frames.items():
        pd.testing.assert_series_equal(frame.timestamp, val.timestamp)
        np.testing.assert_array_equal(baseline, frame.forecast_aft)
        if set(train.continuity_block) & set(frame.continuity_block):
            raise ValueError('Block overlap')
        chosen = frame.loc[frame.score_allowed].copy()
        # Legacy original intervals are incompatible; the original exact cohort
        # is unchanged. All canonical interval rows remain in expanded scoring.
        if key == 'original':
            chosen = chosen.loc[chosen.exact_hours.notna()].copy()
        canonical = labels.loc[chosen.timestamp]
        for col in ('continuity_block', 'lower_bound_hours', 'upper_bound_hours', 'exact_hours'):
            np.testing.assert_allclose(chosen[col], canonical[col], equal_nan=True)
        chosen['label_available_at'] = canonical.label_available_at.to_numpy()
        if not chosen.label_available_at.lt(pd.Timestamp('2024-01-01', tz='Asia/Shanghai')).all():
            raise ValueError('Immature scoring label')
        cohorts[key] = chosen
    protected = list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    protected += [Path(__file__).resolve(), label_path,
                  ROOT / 'src/haze_forecast/observed_aft.py',
                  ROOT / 'src/haze_forecast/training.py',
                  ROOT / 'src/haze_forecast/cohort_scorecard.py',
                  ROOT / 'scripts/benchmark_gefs_72h_aft.py',
                  ROOT / 'scripts/benchmark_observed_business_aft.py',
                  ROOT / 'src/haze_forecast/app.py', ROOT / 'src/haze_forecast/workbench_ui.py']
    inputs.update({str(p): sha(p) for p in protected})
    params = {'objective': 'survival:aft', 'eval_metric': 'aft-nloglik',
              'aft_loss_distribution': 'normal', 'aft_loss_distribution_scale': 1.2,
              'eta': 0.05, 'max_depth': 4, 'min_child_weight': 8, 'subsample': 0.9,
              'colsample_bytree': 0.8, 'lambda': 3., 'seed': 42, 'tree_method': 'hist',
              'verbosity': 0, 'nthread': 1}
    manifest = {'input_sha256': inputs, 'features': names, 'groups': groups,
                'params': params, 'rounds': 500, 'training_rows': len(train),
                'inference_rows': len(val), 'training_support': 'Original f024; no added 2021 forecasts',
                'gate': 'All three exact MAEs strictly lower in both fixed 2023 cohorts',
                'selection': 'One grouping, no tuning; failure stops before 2024/2025',
                'label_policy': 'Original 196 exact rows unchanged; original legacy intervals rejected. Expanded 396 rows retain all 174 canonical intervals.',
                'limitations': ['Reused 2023 development year; no untouched holdout',
                                'No new real-time input availability evidence',
                                'Constraints can remove useful physical interactions',
                                'Original and expanded cohorts overlap; not independent replications'],
                'deployment_authorized': False, 'xgboost_version': xgb.__version__}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    matrix = xgb.DMatrix(train[names].astype(float), missing=np.nan)
    matrix.set_float_info('label_lower_bound', train.lower_bound_hours)
    matrix.set_float_info('label_upper_bound', train.upper_bound_hours)
    inference = xgb.DMatrix(val[names].astype(float), missing=np.nan)
    control = xgb.train(params, matrix, num_boost_round=500)
    np.testing.assert_array_equal(control.predict(inference), baseline)
    candidate = xgb.train({**params, 'interaction_constraints': groups}, matrix, num_boost_round=500)
    audit = audit_paths([json.loads(t) for t in candidate.get_dump(dump_format='json')], groups)
    candidate.save_model(out / 'candidate.ubj')
    restored = xgb.Booster()
    restored.load_model(out / 'candidate.ubj')
    prediction = candidate.predict(inference)
    np.testing.assert_array_equal(prediction, restored.predict(inference))
    saved = val[['timestamp']].copy()
    saved['baseline'] = baseline
    saved['candidate'] = prediction
    saved.to_parquet(out / 'all_predictions.parquet', index=False)
    indexed = pd.Series(prediction, index=val.timestamp)
    scores, checks = {}, {}
    for key, frame in cohorts.items():
        frame['candidate'] = indexed.loc[frame.timestamp].to_numpy()
        scores[key] = {arm: score_predictions(frame, frame[column].to_numpy())
                       for arm, column in [('baseline', 'forecast_aft'), ('candidate', 'candidate')]}
        checks[key] = all(scores[key]['candidate'][k] < scores[key]['baseline'][k]
                          for k in ('exact_row_mae', 'exact_block_mae', 'tail24_block_mae'))
        frame.to_parquet(out / f'{key}.parquet', index=False)
    verify_hashes(inputs)
    report = {'scores': scores, 'checks': checks, 'passed': all(checks.values()),
              'tree_audit': audit, 'baseline_retraining_exact_match': True,
              'checkpoint_replay_exact_match': True, 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
