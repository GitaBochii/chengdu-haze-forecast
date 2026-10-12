"""Fixed trajectory-feature ablation on interval AFT, gated on development data."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.features import TRAJECTORY_FEATURE_NAMES
from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    dev = base / 'interval_business_aft_v1'
    out = base / 'interval_trajectory_v1'
    if out.exists():
        raise FileExistsError(out)
    labels_path = base / 'interval_business_labels_v1/origins.parquet'
    paths = [labels_path, dev / 'training_rows.parquet', dev / 'predictions.parquet',
             dev / 'protocol.json', Path(__file__), ROOT / 'src/haze_forecast/features.py',
             ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py',
             ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    names = json.loads((dev / 'protocol.json').read_bytes())['features'] + list(TRAJECTORY_FEATURE_NAMES)
    assert len(names) == len(set(names)) == 98
    train = pd.read_parquet(dev / 'training_rows.parquet')
    val = pd.read_parquet(dev / 'predictions.parquet')
    labels = pd.read_parquet(labels_path).set_index('timestamp')
    pd.testing.assert_frame_equal(val[names].reset_index(drop=True), labels.loc[val.timestamp, names].reset_index(drop=True))
    pd.testing.assert_frame_equal(train[names].reset_index(drop=True), labels.loc[train.timestamp, names].reset_index(drop=True))
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'seed': 42, 'rounds': 500,
                'gate': 'both row/block MAE lower and tail not worse than 87-feature interval AFT on both original and expanded exact development cohorts',
                'selection': 'single fixed existing 11-feature trajectory addition; no parameter search',
                'limitations': ['reused 2023 development data', 'trajectory resets with unknown pollution status',
                                'exact subsets still do not cover all origins']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(
        train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'trajectory_aft.joblib')
    pred = model.predict(val[names])
    assert np.isfinite(pred).all()
    np.testing.assert_array_equal(joblib.load(out / 'trajectory_aft.joblib').predict(val[names]), pred)
    val['trajectory_aft'] = pred
    expanded = val.copy()
    for name in ('exact_hours', 'lower_bound_hours', 'upper_bound_hours', 'label_available_at', 'informative'):
        expanded[name] = labels.loc[val.timestamp, name].to_numpy()
    end = pd.Timestamp('2024-01-01', tz='Asia/Shanghai')
    expanded['score_allowed'] = val.score_allowed & expanded.label_available_at.lt(end)
    results = {}
    for cohort, frame in [('original', val), ('expanded', expanded)]:
        scored = frame.loc[frame.score_allowed]
        a = metrics(scored, scored.trajectory_aft.to_numpy())
        b = metrics(scored, scored.interval_bounds.to_numpy())
        results[cohort] = {'candidate': a, 'control': b, 'passed': bool(
            a['row_mae'] < b['row_mae'] and a['block_mae'] < b['block_mae'] and
            a['tail_24h_block_mae'] <= b['tail_24h_block_mae'])}
    val.to_parquet(out / 'predictions.parquet', index=False)
    expanded.to_parquet(out / 'expanded_predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'cohorts': results, 'passes_development_gate': all(x['passed'] for x in results.values()),
              'inputs_unchanged': True,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
