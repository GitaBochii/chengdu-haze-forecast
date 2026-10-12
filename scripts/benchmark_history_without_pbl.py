"""Development ablation of unavailable boundary-layer features, with fixed AFT."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    dev = base / 'interval_history_counts_v1'
    out = base / 'history_without_pbl_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [dev / 'protocol.json', dev / 'predictions.parquet', dev / 'expanded_predictions.parquet',
             base / 'interval_business_aft_v1/training_rows.parquet',
             base / 'trajectory_reset_audit_v1/origins.parquet', Path(__file__),
             ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py',
             ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    original_names = json.loads((dev / 'protocol.json').read_bytes())['features']
    removed = [n for n in original_names if n.startswith('boundary_layer_height_m')]
    names = [n for n in original_names if n not in removed]
    train = pd.read_parquet(base / 'interval_business_aft_v1/training_rows.parquet')
    history = pd.read_parquet(base / 'trajectory_reset_audit_v1/origins.parquet').set_index('timestamp')
    train[original_names[87:]] = history.loc[train.timestamp, original_names[87:]].to_numpy()
    val = pd.read_parquet(dev / 'predictions.parquet')
    assert val.boundary_layer_height_m.isna().all()
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'removed_features': removed,
                'selection': 'one fixed boundary-layer ablation, no parameter or threshold search',
                'gate': 'both main MAEs improve and tail not worse on original and expanded development cohorts',
                'seed': 42, 'rounds': 500}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    pred = model.predict(val[names])
    assert np.isfinite(pred).all()
    np.testing.assert_array_equal(joblib.load(out / 'model.joblib').predict(val[names]), pred)
    val['no_pbl_aft'] = pred
    expanded = pd.read_parquet(dev / 'expanded_predictions.parquet')
    np.testing.assert_array_equal(expanded.timestamp, val.timestamp)
    expanded['no_pbl_aft'] = pred
    results = {}
    for name, frame in [('original', val), ('expanded', expanded)]:
        frame = frame.loc[frame.score_allowed]
        a = metrics(frame, frame.no_pbl_aft.to_numpy())
        b = metrics(frame, frame.history_counts_aft.to_numpy())
        results[name] = {'candidate': a, 'control': b, 'passed': bool(
            a['row_mae'] < b['row_mae'] and a['block_mae'] < b['block_mae'] and
            a['tail_24h_block_mae'] <= b['tail_24h_block_mae'])}
    val.to_parquet(out / 'predictions.parquet', index=False)
    expanded.to_parquet(out / 'expanded_predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'cohorts': results, 'passed': all(x['passed'] for x in results.values()),
              'inputs_unchanged': True, 'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
