"""Cross-year confirmation of the fixed development-approved forecast features."""

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
    out = base / 'reusable_gefs_crossyear_v1'
    if out.exists():
        raise FileExistsError(out)
    assert json.loads((dev / 'report.json').read_bytes())['passed']
    names = json.loads((dev / 'protocol.json').read_bytes())['features']
    paths = [dev / 'protocol.json', dev / 'report.json', Path(__file__),
             base / 'reusable_gefs_snapshots_v1/features.parquet', base / 'trajectory_reset_audit_v1/origins.parquet',
             ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py', ROOT / 'src/haze_forecast/app.py']
    for year in (2024, 2025):
        paths += [base / f'interval_crossyear_v1/{year}/training_rows.parquet']
        paths += [base / f'interval_history_crossyear_v1/{year}_{suffix}.parquet' for suffix in ('predictions', 'expanded', 'common', 'inference_features')]
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    forecasts = pd.read_parquet(base / 'reusable_gefs_snapshots_v1/features.parquet').set_index('timestamp')
    history = pd.read_parquet(base / 'trajectory_reset_audit_v1/origins.parquet').set_index('timestamp')
    out.mkdir(parents=True)
    (out / 'protocol.json').write_text(json.dumps({'input_sha256': before, 'features': names,
        'selection': 'none; fixed 104 features, 500 rounds, seed42',
        'limitations': ['historical diagnostics, not pristine holdout', 'archive availability proxy only']}, indent=2), encoding='utf-8')
    results = {}
    for year in (2024, 2025):
        train = pd.read_parquet(base / f'interval_crossyear_v1/{year}/training_rows.parquet')
        train[names[87:92]] = history.loc[train.timestamp, names[87:92]].to_numpy()
        train[names[92:]] = forecasts.loc[train.timestamp, names[92:]].to_numpy()
        val = pd.read_parquet(base / f'interval_history_crossyear_v1/{year}_predictions.parquet')
        x = pd.read_parquet(base / f'interval_history_crossyear_v1/{year}_inference_features.parquet')
        x[names[92:]] = forecasts.loc[val.timestamp, names[92:]].to_numpy()
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(train[names], train.lower_bound_hours, train.upper_bound_hours)
        joblib.dump(model, out / f'{year}.joblib')
        pred = model.predict(x[names])
        assert np.isfinite(pred).all()
        np.testing.assert_array_equal(joblib.load(out / f'{year}.joblib').predict(x[names]), pred)
        by_time = pd.Series(pred, index=val.timestamp)
        scores = {}
        for cohort in ('predictions', 'expanded', 'common'):
            frame = pd.read_parquet(base / f'interval_history_crossyear_v1/{year}_{cohort}.parquet')
            frame['forecast_aft'] = by_time.loc[frame.timestamp].to_numpy()
            arms = ['forecast_aft', 'history_aft'] + (['strict_prediction_hours'] if cohort == 'common' else [])
            scores[cohort] = {a: metrics(frame, frame[a].to_numpy()) for a in arms}
            frame.to_parquet(out / f'{year}_{cohort}.parquet', index=False)
        x.to_parquet(out / f'{year}_features.parquet', index=False)
        results[str(year)] = scores
    assert before == {str(p): sha(p) for p in paths}
    report = {'years': results, 'inputs_unchanged': True,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
