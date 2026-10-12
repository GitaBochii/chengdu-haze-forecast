"""Fixed temperature/humidity forecast correction ablation on development data."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    control = base / 'reusable_gefs_aft_v1'
    out = base / 'corrected_gefs_aft_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [control / 'protocol.json', control / 'predictions.parquet', control / 'expanded_predictions.parquet',
             base / 'interval_business_aft_v1/training_rows.parquet',
             base / 'trajectory_reset_audit_v1/origins.parquet',
             base / 'reusable_gefs_snapshots_v1/features.parquet',
             base / 'causal_gefs_bias_v1/features.parquet', base / 'causal_gefs_bias_v1/report.json',
             Path(__file__), ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py',
             ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    names = json.loads((control / 'protocol.json').read_bytes())['features']
    changed = ['gefs_mean_temperature_c_f024', 'gefs_mean_relative_humidity_pct_f024']
    forecast = pd.read_parquet(base / 'reusable_gefs_snapshots_v1/features.parquet').set_index('timestamp')
    corrected = pd.read_parquet(base / 'causal_gefs_bias_v1/features.parquet').set_index('timestamp')
    history = pd.read_parquet(base / 'trajectory_reset_audit_v1/origins.parquet').set_index('timestamp')
    train = pd.read_parquet(base / 'interval_business_aft_v1/training_rows.parquet')
    train[names[87:92]] = history.loc[train.timestamp, names[87:92]].to_numpy()
    train[names[92:]] = forecast.loc[train.timestamp, names[92:]].to_numpy()
    val = pd.read_parquet(control / 'predictions.parquet')
    for frame in (train, val):
        original = frame[names].copy()
        frame[changed] = corrected.loc[frame.timestamp, changed].to_numpy()
        pd.testing.assert_frame_equal(original.drop(columns=changed), frame[names].drop(columns=changed))
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'changed_fields': changed, 'seed': 42, 'rounds': 500,
                'selection': 'one temperature/humidity correction, all other inputs and settings fixed',
                'gate': 'both main MAEs lower and tail no worse on original and expanded 2023 cohorts',
                'limitations': ['24h weather observation availability is assumed, not certified',
                                'reused development data; meteorological diagnostics also inspected later years']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    pred = model.predict(val[names])
    assert np.isfinite(pred).all()
    np.testing.assert_array_equal(joblib.load(out / 'model.joblib').predict(val[names]), pred)
    val['corrected_forecast_aft'] = pred
    expanded = pd.read_parquet(control / 'expanded_predictions.parquet')
    np.testing.assert_array_equal(expanded.timestamp, val.timestamp)
    expanded['corrected_forecast_aft'] = pred
    scores = {}
    for cohort, frame in [('original', val), ('expanded', expanded)]:
        frame = frame.loc[frame.score_allowed]
        a = metrics(frame, frame.corrected_forecast_aft.to_numpy())
        b = metrics(frame, frame.forecast_aft.to_numpy())
        scores[cohort] = {'candidate': a, 'control': b, 'passed': bool(
            a['row_mae'] < b['row_mae'] and a['block_mae'] < b['block_mae'] and
            a['tail_24h_block_mae'] <= b['tail_24h_block_mae'])}
    val.to_parquet(out / 'predictions.parquet', index=False)
    expanded.to_parquet(out / 'expanded_predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'cohorts': scores, 'passed': all(s['passed'] for s in scores.values()),
              'inputs_unchanged': True, 'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
