"""Cross-year diagnostic of the development-approved fixed trajectory AFT."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    dev = base / 'interval_trajectory_v1'
    out = base / 'interval_trajectory_crossyear_v1'
    if out.exists():
        raise FileExistsError(out)
    assert json.loads((dev / 'report.json').read_bytes())['passes_development_gate']
    names = json.loads((dev / 'protocol.json').read_bytes())['features']
    paths = [dev / 'report.json', dev / 'protocol.json', Path(__file__),
             base / 'interval_business_labels_v1/origins.parquet',
             base / 'ghcnh_weather_repair_v1/repaired_features.parquet',
             ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py',
             ROOT / 'src/haze_forecast/features.py', ROOT / 'src/haze_forecast/app.py']
    paths += [base / f'interval_crossyear_v1/{y}/{n}.parquet' for y in (2024, 2025) for n in ('training_rows', 'predictions')]
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    labels = pd.read_parquet(base / 'interval_business_labels_v1/origins.parquet').set_index('timestamp')
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'selection': 'fixed development-approved model; no cross-year tuning'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    results = {}
    for year in (2024, 2025):
        train = pd.read_parquet(base / f'interval_crossyear_v1/{year}/training_rows.parquet')
        val = pd.read_parquet(base / f'interval_crossyear_v1/{year}/predictions.parquet')
        x = val[names].copy()
        if year == 2025:
            repair = pd.read_parquet(base / 'ghcnh_weather_repair_v1/repaired_features.parquet')
            assert len(repair) == len(x)
            x[repair.columns] = repair.to_numpy()
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(
            train[names], train.lower_bound_hours, train.upper_bound_hours)
        joblib.dump(model, out / f'{year}.joblib')
        pred = model.predict(x)
        assert np.isfinite(pred).all()
        np.testing.assert_array_equal(joblib.load(out / f'{year}.joblib').predict(x), pred)
        val['trajectory_aft'] = pred
        expanded = val.copy()
        for col in ('exact_hours', 'lower_bound_hours', 'upper_bound_hours', 'informative'):
            expanded[col] = labels.loc[val.timestamp, col].to_numpy()
        results[str(year)] = {}
        for cohort, frame in [('original', val), ('expanded', expanded)]:
            results[str(year)][cohort] = {arm: metrics(frame, frame[arm].to_numpy())
                for arm in ('trajectory_aft', 'interval_bounds', 'legacy_training')}
        val.to_parquet(out / f'{year}_predictions.parquet', index=False)
        expanded.to_parquet(out / f'{year}_expanded.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'years': results, 'inputs_unchanged': True,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
