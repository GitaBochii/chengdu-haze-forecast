"""Forecast-value ablation preserving feature names, availability and ages."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    dev = base / 'reusable_gefs_aft_v1'
    out = base / 'gefs_metadata_control_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [dev / 'protocol.json', dev / 'predictions.parquet', dev / 'expanded_predictions.parquet',
        base / 'interval_business_aft_v1/training_rows.parquet', base / 'trajectory_reset_audit_v1/origins.parquet',
        base / 'reusable_gefs_snapshots_v1/features.parquet', Path(__file__),
        ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py', ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    names = json.loads((dev / 'protocol.json').read_bytes())['features']
    values = names[92:101]
    assert len(values) == 9 and all(n.startswith('gefs_') for n in values)
    train = pd.read_parquet(base / 'interval_business_aft_v1/training_rows.parquet')
    h = pd.read_parquet(base / 'trajectory_reset_audit_v1/origins.parquet').set_index('timestamp')
    g = pd.read_parquet(base / 'reusable_gefs_snapshots_v1/features.parquet').set_index('timestamp')
    train[names[87:92]] = h.loc[train.timestamp, names[87:92]].to_numpy()
    train[names[92:]] = g.loc[train.timestamp, names[92:]].to_numpy()
    val = pd.read_parquet(dev / 'predictions.parquet')
    for frame in (train, val):
        prior = frame[names].copy()
        frame[values] = frame[values].where(frame[values].isna(), 0.)
        pd.testing.assert_frame_equal(frame[values].isna(), prior[values].isna())
        pd.testing.assert_frame_equal(frame[names].drop(columns=values), prior.drop(columns=values))
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'masked_value_fields': values, 'seed': 42, 'rounds': 500,
        'intervention': 'all available forecast values set to zero; missing values and three age fields unchanged',
        'selection': 'diagnostic only, not a deployable weather representation',
        'limitations': ['constant placeholders are not physical weather', 'reused 2023 development year',
                        'model split opportunities change when value information is removed']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    pred = model.predict(val[names])
    assert np.isfinite(pred).all()
    np.testing.assert_array_equal(joblib.load(out / 'model.joblib').predict(val[names]), pred)
    val['metadata_only_aft'] = pred
    expanded = pd.read_parquet(dev / 'expanded_predictions.parquet')
    np.testing.assert_array_equal(expanded.timestamp, val.timestamp)
    expanded['metadata_only_aft'] = pred
    scores = {}
    for cohort, frame in [('original', val), ('expanded', expanded)]:
        frame = frame.loc[frame.score_allowed]
        scores[cohort] = {arm: metrics(frame, frame[arm].to_numpy()) for arm in
                         ('metadata_only_aft', 'forecast_aft', 'history_counts_aft')}
    val.to_parquet(out / 'predictions.parquet', index=False)
    expanded.to_parquet(out / 'expanded_predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'scores': scores, 'inputs_unchanged': True,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
