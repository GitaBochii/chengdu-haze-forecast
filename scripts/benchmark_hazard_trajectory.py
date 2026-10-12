"""Fixed causal trajectory ablation for the observed-prefix hazard model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.features import TRAJECTORY_FEATURE_NAMES
from haze_forecast.observed_hazard import ObservedHazardRegressor
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    out = ROOT / 'artifacts/task01_recovery_head/hazard_trajectory_v1'
    if out.exists():
        raise FileExistsError(out)
    annual = ROOT / 'artifacts/task01_recovery_head/observed_annual_update_v1'
    dev = ROOT / 'artifacts/task01_recovery_head/observed_hazard_development_v1'
    repair = ROOT / 'artifacts/task01_recovery_head/ghcnh_weather_repair_v1'
    old_names = json.loads((dev / 'protocol.json').read_bytes())['features']
    names = old_names + list(TRAJECTORY_FEATURE_NAMES)
    assert len(names) == len(set(names)) == 98
    paths = [annual / str(y) / n for y in (2023, 2024, 2025) for n in ('training_rows.parquet', 'predictions.parquet')]
    paths += [dev / 'protocol.json', dev / 'predictions.parquet', dev / 'report.json',
        repair / 'repaired_features.parquet', repair / 'predictions.parquet', Path(__file__),
        ROOT / 'src/haze_forecast/features.py', ROOT / 'src/haze_forecast/observed_hazard.py',
        ROOT / 'src/haze_forecast/discrete_survival.py', ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    out.mkdir(parents=True)
    protocol = {'input_sha256': hashes, 'features': names, 'new_features': list(TRAJECTORY_FEATURE_NAMES),
        'selection': 'single prespecified candidate; 2023 both main metrics improve no-trajectory hazard; tail not worse',
        'unchanged_settings': 'LightGBM 300 trees, seed42, same observed prefixes and annual splits',
        '2025_weather': 'both arms use GHCNh-repaired weather; historical diagnostic, no realtime certification'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    train = pd.read_parquet(annual / '2023/training_rows.parquet')
    val = pd.read_parquet(dev / 'predictions.parquet')
    model = ObservedHazardRegressor(seed=42).fit_bounds(train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / '2023.joblib')
    pred = model.predict(val[names])
    val['trajectory_hazard'] = pred
    allowed = val.score_allowed & np.isfinite(pred)
    candidate = metrics(val.loc[allowed], pred[allowed])
    control = metrics(val.loc[allowed], val.loc[allowed, 'observed_bounds_hazard'].to_numpy())
    passed = bool(np.isfinite(pred).all() and candidate['row_mae'] < control['row_mae'] and
        candidate['block_mae'] < control['block_mae'] and candidate['tail_24h_block_mae'] <= control['tail_24h_block_mae'])
    val.to_parquet(out / '2023_predictions.parquet', index=False)
    selection = {'candidate': candidate, 'control': control, 'passed': passed,
                 'unresolved': int(np.isnan(pred).sum())}
    (out / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    crossyear = {}
    if passed:
        for year in (2024, 2025):
            train = pd.read_parquet(annual / str(year) / 'training_rows.parquet')
            val = pd.read_parquet(annual / str(year) / 'predictions.parquet')
            if year == 2025:
                repaired = pd.read_parquet(repair / 'repaired_features.parquet')
                val[old_names] = repaired[old_names]
                reference = pd.read_parquet(repair / 'predictions.parquet').hazard_after.to_numpy()
            else:
                reference_path = ROOT / 'artifacts/task01_recovery_head/observed_hazard_crossyear_v1/2024_predictions.parquet'
                reference = pd.read_parquet(reference_path).hazard.to_numpy()
            model = ObservedHazardRegressor(seed=42).fit_bounds(train[names], train.lower_bound_hours, train.upper_bound_hours)
            joblib.dump(model, out / f'{year}.joblib')
            pred = model.predict(val[names])
            ok = np.isfinite(pred)
            val['trajectory_hazard'], val['control_hazard'] = pred, reference
            val.to_parquet(out / f'{year}_predictions.parquet', index=False)
            crossyear[str(year)] = {'candidate': metrics(val.loc[ok], pred[ok]),
                'control': metrics(val.loc[ok], reference[ok]), 'unresolved': int((~ok).sum())}
    assert hashes == {str(p): sha(p) for p in paths}
    report = {'status': 'completed_trajectory_ablation', 'selection': selection, 'crossyear': crossyear,
        'inputs_unchanged': True, 'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'selection': selection, 'crossyear': crossyear}, indent=2))


if __name__ == '__main__':
    main()
