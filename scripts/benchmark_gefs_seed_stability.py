"""Prespecified five-seed stability diagnostic and unweighted median ensemble."""

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
    out = base / 'gefs_seed_stability_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [dev / 'protocol.json', dev / 'model.joblib', dev / 'predictions.parquet', dev / 'expanded_predictions.parquet',
             base / 'interval_business_aft_v1/training_rows.parquet', base / 'trajectory_reset_audit_v1/origins.parquet',
             base / 'reusable_gefs_snapshots_v1/features.parquet', Path(__file__),
             ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py', ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    names = json.loads((dev / 'protocol.json').read_bytes())['features']
    train = pd.read_parquet(base / 'interval_business_aft_v1/training_rows.parquet')
    history = pd.read_parquet(base / 'trajectory_reset_audit_v1/origins.parquet').set_index('timestamp')
    forecast = pd.read_parquet(base / 'reusable_gefs_snapshots_v1/features.parquet').set_index('timestamp')
    train[names[87:92]] = history.loc[train.timestamp, names[87:92]].to_numpy()
    train[names[92:]] = forecast.loc[train.timestamp, names[92:]].to_numpy()
    val = pd.read_parquet(dev / 'predictions.parquet')
    expanded = pd.read_parquet(dev / 'expanded_predictions.parquet')
    np.testing.assert_array_equal(val.timestamp, expanded.timestamp)
    out.mkdir(parents=True)
    seeds = [42, 43, 44, 45, 46]
    protocol = {'input_sha256': before, 'features': names, 'seeds': seeds, 'rounds': 500,
                'aggregation': 'fixed pointwise median over all five seeds',
                'gate': 'both main MAEs lower and tail no worse than seed42 in both development cohorts',
                'selection': 'no best-seed selection, weighting, or ensemble-size search',
                'limitations': ['reused 2023 development year; seed stability is not independent generalization']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    columns = []
    for seed in seeds:
        model = joblib.load(dev / 'model.joblib') if seed == 42 else ObservedAFTSurvivalRegressor(seed=seed, nthread=1).fit_bounds(
            train[names], train.lower_bound_hours, train.upper_bound_hours)
        joblib.dump(model, out / f'seed_{seed}.joblib')
        pred = model.predict(val[names])
        assert np.isfinite(pred).all()
        if seed == 42:
            np.testing.assert_array_equal(pred, val.forecast_aft)
        column = f'seed_{seed}'
        val[column] = expanded[column] = pred
        columns.append(column)
    val['median_ensemble'] = expanded['median_ensemble'] = np.median(val[columns].to_numpy(), axis=1)
    scores = {}
    for cohort, frame in [('original', val), ('expanded', expanded)]:
        frame = frame.loc[frame.score_allowed]
        scores[cohort] = {c: metrics(frame, frame[c].to_numpy()) for c in columns + ['median_ensemble']}
    passed = True
    for cohort in scores.values():
        a, b = cohort['median_ensemble'], cohort['seed_42']
        passed &= a['row_mae'] < b['row_mae'] and a['block_mae'] < b['block_mae'] and a['tail_24h_block_mae'] <= b['tail_24h_block_mae']
    val.to_parquet(out / 'predictions.parquet', index=False)
    expanded.to_parquet(out / 'expanded_predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'scores': scores, 'passed': bool(passed), 'inputs_unchanged': True,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'scores': scores, 'passed': bool(passed)}, indent=2))


if __name__ == '__main__':
    main()
