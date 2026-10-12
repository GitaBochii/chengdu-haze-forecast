"""Single fixed pre-2022 transition-model experiment, isolated from deployment."""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.run_length_recovery import recovery_start_distribution
from haze_forecast.run_length_training import observed_transitions, recovery_median
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_temporal_calibration_data_v1'
    out = base / 'run_length_transition_2022_v2'
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((source / 'report.json').read_bytes())['output_sha256']
    for name, digest in manifest.items():
        assert sha(source / name) == digest
    grid_path = base / 'observed_business_dataset_v1/features.parquet'
    paths = [source / name for name in manifest]
    paths += [source / 'report.json', grid_path, Path(__file__),
              ROOT / 'src/haze_forecast/run_length_recovery.py',
              ROOT / 'src/haze_forecast/run_length_training.py',
              ROOT / 'src/haze_forecast/cohort_scorecard.py', ROOT / 'src/haze_forecast/app.py']
    paths += [p for p in (ROOT / 'artifacts/cnemc_2025_deployment_candidate').iterdir() if p.is_file()]
    before = {str(p): sha(p) for p in paths}
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    train = pd.read_parquet(source / 'training_rows.parquet')
    val = pd.read_parquet(source / 'predictions.parquet')
    cutoff = pd.Timestamp('2022-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()
    assert not set(train.continuity_block) & set(val.continuity_block)
    assert len(val) == 411 and val.score_allowed.all() and val.timestamp.dt.year.eq(2022).all()
    grid = pd.read_parquet(grid_path)
    grid = grid.loc[grid.timestamp.lt(cutoff)].set_index('timestamp')
    assert grid.index.is_unique and grid.index.is_monotonic_increasing
    assert np.all(np.diff(grid.index.asi8) == pd.Timedelta(hours=1).value)
    assert np.isfinite(train.upper_bound_hours).all()
    horizon = int(train.upper_bound_hours.max()) + 5
    params = {'objective': 'binary', 'verbosity': -1, 'num_threads': 1,
              'learning_rate': .04, 'num_leaves': 15, 'min_data_in_leaf': 40,
              'feature_fraction': .8, 'lambda_l2': 2., 'seed': 42, 'force_col_wise': True}
    protocol = {'input_sha256': before, 'features': names, 'horizon': horizon,
                'params': params, 'rounds': 300, 'train_cutoff': str(cutoff),
                'supervision': 'Equal transition weights; stop at first missing or six-hour confirmation',
                'selection': 'One fixed candidate; row/block MAE lower and tail24 no worse',
                'limitations': ['Same origin pool does not mean identical supervision or loss',
                                'Longer observed prefixes contribute more transitions',
                                'Conditional state probabilities simplify unobserved path dependence',
                                'State-time combinations may have little training support',
                                '2022 reused development, no later-year evaluation'],
                'unresolved': 'Keep all origins, mark failed; no median capping',
                'deployment_authorized': False}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    supervision = []
    for owner, row in enumerate(train.itertuples()):
        end = row.timestamp + pd.Timedelta(hours=row.upper_bound_hours + 5)
        assert end < cutoff
        future = grid.loc[row.timestamp + pd.Timedelta(hours=1):end, 'aqi']
        assert len(future) == int(row.upper_bound_hours) + 5
        hours, states, targets = observed_transitions(future)
        supervision.extend(zip([owner] * len(hours), hours, states, targets, strict=True))
    transitions = pd.DataFrame(supervision, columns=['owner', 'horizon_hours', 'clean_run', 'target'])
    transitions.to_parquet(out / 'transitions.parquet', index=False)
    owner = transitions.owner.to_numpy()
    features = train[names].iloc[owner].reset_index(drop=True)
    features['horizon_hours'] = transitions.horizon_hours
    features['clean_run'] = transitions.clean_run
    model = lgb.train(params, lgb.Dataset(features, label=transitions.target), num_boost_round=300)
    (out / 'model.txt').write_text(model.model_to_string(), encoding='utf-8')
    loaded = lgb.Booster(model_str=(out / 'model.txt').read_text(encoding='utf-8'))
    distributions = []
    for start in range(0, len(val), 32):
        batch = val[names].iloc[start:start + 32]
        expanded = np.repeat(batch.to_numpy(dtype=float), horizon * 6, axis=0)
        x = pd.DataFrame(expanded, columns=names)
        x['horizon_hours'] = np.tile(np.repeat(np.arange(1, horizon + 1), 6), len(batch))
        x['clean_run'] = np.tile(np.arange(6), len(batch) * horizon)
        probability = model.predict(x, num_threads=1)
        np.testing.assert_array_equal(probability, loaded.predict(x, num_threads=1))
        distributions.append(recovery_start_distribution(probability.reshape(len(batch), horizon, 6)))
    distribution = np.concatenate(distributions)
    np.save(out / 'distribution.npy', distribution)
    pred = recovery_median(distribution)
    baseline = joblib.load(source / 'model.joblib').predict(val[names])
    np.testing.assert_array_equal(baseline, val.forecast_aft)
    result = val.copy()
    result['transition_prediction'] = pred
    result['unconfirmed_tail_probability'] = distribution[:, -1]
    result.to_parquet(out / 'predictions.parquet', index=False)
    control = score_predictions(val, baseline)
    candidate = score_predictions(val, pred) if np.isfinite(pred).all() else None
    passed = bool(candidate is not None and candidate['exact_row_mae'] < control['exact_row_mae']
                  and candidate['exact_block_mae'] < control['exact_block_mae']
                  and candidate['tail24_block_mae'] <= control['tail24_block_mae'])
    report = {'training_origins': len(train), 'supervised_origins': int(transitions.owner.nunique()),
              'transitions': len(transitions), 'state_counts': transitions.clean_run.value_counts().to_dict(),
              'control': control, 'candidate': candidate, 'passed': passed,
              'unresolved_rows': int(np.isnan(pred).sum()), 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    assert before == {str(p): sha(Path(p)) for p in before}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
