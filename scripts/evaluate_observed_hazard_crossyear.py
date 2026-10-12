"""Annual fixed-configuration evaluation of the development-passing hazard arm."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_hazard import ObservedHazardRegressor
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head/observed_annual_update_v1'
    dev = ROOT / 'artifacts/task01_recovery_head/observed_hazard_development_v1'
    out = ROOT / 'artifacts/task01_recovery_head/observed_hazard_crossyear_v1'
    if out.exists():
        raise FileExistsError(out)
    assert json.loads((dev / 'report.json').read_bytes())['scores']['observed_bounds']['passes_gate']
    names = json.loads((dev / 'protocol.json').read_bytes())['features']
    paths = [base / str(y) / n for y in (2024, 2025) for n in ('training_rows.parquet', 'predictions.parquet')]
    paths += [dev / 'report.json', dev / 'protocol.json', Path(__file__),
        ROOT / 'src/haze_forecast/observed_hazard.py', ROOT / 'src/haze_forecast/discrete_survival.py',
        ROOT / 'scripts/benchmark_observed_business_aft.py', ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'selection': 'observed_bounds development gate passed',
        'configuration': 'same LightGBM 300 rounds seed42; annual expanding training; no retuning',
        'limitations': ['reused crossyear data', 'exact-only scoring', 'not actual website replay',
                       'unresolved medians excluded from metrics and separately reported; no clipping']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    results = {}
    for year in (2024, 2025):
        train = pd.read_parquet(base / str(year) / 'training_rows.parquet')
        val = pd.read_parquet(base / str(year) / 'predictions.parquet')
        model = ObservedHazardRegressor(seed=42).fit_bounds(train[names], train.lower_bound_hours, train.upper_bound_hours)
        joblib.dump(model, out / f'{year}.joblib')
        pred = model.predict(val[names])
        val['hazard'] = pred
        ok = np.isfinite(pred)
        val.to_parquet(out / f'{year}_predictions.parquet', index=False)
        results[str(year)] = {'candidate': metrics(val.loc[ok], pred[ok]),
            'legacy_same_rows': metrics(val.loc[ok], val.loc[ok, 'legacy_training'].to_numpy()),
            'unresolved_origins': int((~ok).sum()), 'horizon': model.max_horizon_hours}
    assert before == {str(p): sha(p) for p in paths}
    report = {'status': 'completed_crossyear', 'results': results, 'inputs_unchanged': True,
        'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
