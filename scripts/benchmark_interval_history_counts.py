"""Development-only addition of explicitly observed historical counts."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    dev = base / 'interval_business_aft_v1'
    out = base / 'interval_history_counts_v1'
    if out.exists():
        raise FileExistsError(out)
    history_path = base / 'trajectory_reset_audit_v1/origins.parquet'
    expanded_path = base / 'interval_trajectory_v1/expanded_predictions.parquet'
    paths = [history_path, expanded_path, dev / 'training_rows.parquet', dev / 'predictions.parquet',
             dev / 'protocol.json', Path(__file__), ROOT / 'src/haze_forecast/observed_aft.py',
             ROOT / 'src/haze_forecast/training.py', ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    extra = ['hours_since_first_observed_high', 'observed_high_hours', 'past_missing_hours',
             'past_short_clean_hours', 'past_high_nonparticle_hours']
    names = json.loads((dev / 'protocol.json').read_bytes())['features'] + extra
    history = pd.read_parquet(history_path).set_index('timestamp')
    train = pd.read_parquet(dev / 'training_rows.parquet')
    val = pd.read_parquet(dev / 'predictions.parquet')
    for frame in (train, val):
        frame[extra] = history.loc[frame.timestamp, extra].to_numpy()
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'seed': 42, 'rounds': 500,
                'gate': 'both main MAEs improve and tail does not worsen on original and expanded exact development cohorts',
                'selection': 'single fixed five-count history addition to 87-feature interval AFT',
                'limitations': ['elapsed time does not imply continuous pollution', 'reused 2023 development year']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    pred = model.predict(val[names])
    assert np.isfinite(pred).all()
    np.testing.assert_array_equal(joblib.load(out / 'model.joblib').predict(val[names]), pred)
    val['history_counts_aft'] = pred
    expanded = pd.read_parquet(expanded_path)
    np.testing.assert_array_equal(expanded.timestamp, val.timestamp)
    expanded['history_counts_aft'] = pred
    results = {}
    for cohort, frame in [('original', val), ('expanded', expanded)]:
        frame = frame.loc[frame.score_allowed]
        candidate = metrics(frame, frame.history_counts_aft.to_numpy())
        control = metrics(frame, frame.interval_bounds.to_numpy())
        results[cohort] = {'candidate': candidate, 'control': control, 'passed': bool(
            candidate['row_mae'] < control['row_mae'] and candidate['block_mae'] < control['block_mae'] and
            candidate['tail_24h_block_mae'] <= control['tail_24h_block_mae'])}
    val.to_parquet(out / 'predictions.parquet', index=False)
    expanded.to_parquet(out / 'expanded_predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'cohorts': results, 'passed': all(r['passed'] for r in results.values()),
              'inputs_unchanged': True, 'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
