"""Same-seed no-forecast controls for the frozen five-seed forecast diagnostic."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    treated = base / 'gefs_seed_stability_v1'
    control = base / 'interval_history_counts_v1'
    out = base / 'gefs_paired_seeds_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [treated / 'protocol.json', treated / 'predictions.parquet', treated / 'expanded_predictions.parquet',
             control / 'protocol.json', control / 'model.joblib', control / 'predictions.parquet',
             base / 'interval_business_aft_v1/training_rows.parquet', base / 'trajectory_reset_audit_v1/origins.parquet',
             Path(__file__), ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py',
             ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    names = json.loads((control / 'protocol.json').read_bytes())['features']
    seeds = json.loads((treated / 'protocol.json').read_bytes())['seeds']
    train = pd.read_parquet(base / 'interval_business_aft_v1/training_rows.parquet')
    history = pd.read_parquet(base / 'trajectory_reset_audit_v1/origins.parquet').set_index('timestamp')
    train[names[87:]] = history.loc[train.timestamp, names[87:]].to_numpy()
    val = pd.read_parquet(treated / 'predictions.parquet')
    expanded = pd.read_parquet(treated / 'expanded_predictions.parquet')
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'seeds': seeds, 'rounds': 500,
                'selection': 'none; all five same-seed differences reported',
                'limitations': ['reused development year; seeds are not independent new pollution events',
                    'adding columns changes feature subsampling even with identical seed; does not isolate semantic forecast content']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    scores = {'original': {}, 'expanded': {}}
    for seed in seeds:
        model = joblib.load(control / 'model.joblib') if seed == 42 else ObservedAFTSurvivalRegressor(seed=seed, nthread=1).fit_bounds(
            train[names], train.lower_bound_hours, train.upper_bound_hours)
        joblib.dump(model, out / f'control_{seed}.joblib')
        pred = model.predict(val[names])
        assert np.isfinite(pred).all()
        if seed == 42:
            np.testing.assert_array_equal(pred, pd.read_parquet(control / 'predictions.parquet').history_counts_aft)
        val[f'control_{seed}'] = expanded[f'control_{seed}'] = pred
        for cohort, frame in [('original', val), ('expanded', expanded)]:
            frame = frame.loc[frame.score_allowed]
            a = metrics(frame, frame[f'seed_{seed}'].to_numpy())
            b = metrics(frame, frame[f'control_{seed}'].to_numpy())
            keys = ['row_mae', 'block_mae', 'tail_24h_block_mae']
            delta = {k: a[k] - b[k] for k in keys}
            scores[cohort][str(seed)] = {'forecast': a, 'control': b, 'difference': delta,
                'all_three_improved': all(v < 0 for v in delta.values())}
    val.to_parquet(out / 'predictions.parquet', index=False)
    expanded.to_parquet(out / 'expanded_predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'scores': scores, 'inputs_unchanged': True,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({cohort: {seed: row['difference'] for seed, row in rows.items()} for cohort, rows in scores.items()}, indent=2))


if __name__ == '__main__':
    main()
