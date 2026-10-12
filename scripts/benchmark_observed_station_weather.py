"""Matched observed-station weather ablation for causal hazard training."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cnemc_weather import add_noaa_weather
from haze_forecast.ghcnh_weather import prepare_ghcnh_weather
from haze_forecast.noaa_isd import prepare_noaa_hourly
from haze_forecast.observed_hazard import ObservedHazardRegressor
from haze_forecast.weather_source_ablation import WEATHER_BASE, uniform_weather_rows
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    root = ROOT / 'artifacts/task01_recovery_head'
    out = root / 'observed_station_weather_v1'
    if out.exists():
        raise FileExistsError(out)
    data = root / 'observed_business_dataset_v1'
    annual = root / 'observed_annual_update_v1'
    dev = root / 'observed_hazard_development_v1'
    repair = root / 'ghcnh_weather_repair_v1'
    old_names = json.loads((dev / 'protocol.json').read_bytes())['features']
    names = [n for n in old_names if not n.startswith('boundary_layer_height_m')]
    raw_paths = [ROOT / f'data/raw/noaa_isd/{y}/56294099999.csv' for y in range(2014, 2026)]
    ghcnh_path = ROOT / 'data/raw/ghcnh_discovery_20260906/GHCNh_CHI0000ZUUU_2025.parquet'
    paths = raw_paths + [ghcnh_path, data / 'features.parquet', dev / 'protocol.json', dev / 'predictions.parquet', repair / 'repaired_features.parquet']
    paths += [annual / str(y) / n for y in (2023, 2024, 2025) for n in ('training_rows.parquet', 'predictions.parquet')]
    paths += [Path(__file__)] + [ROOT / ('src/haze_forecast/' + n + '.py') for n in
        ('weather_source_ablation', 'noaa_isd', 'cnemc_weather', 'ghcnh_weather', 'features', 'observed_hazard', 'discrete_survival', 'app')]
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    full = pd.read_parquet(data / 'features.parquet')
    weather = pd.concat([prepare_noaa_hourly(pd.read_csv(p)) for p in raw_paths], ignore_index=True)
    assert not weather.timestamp.duplicated().any()
    proxy = add_noaa_weather(full[['timestamp']], weather)[['timestamp', *WEATHER_BASE]]
    new_weather = prepare_ghcnh_weather(pd.read_parquet(ghcnh_path)).set_index('timestamp')
    aligned = new_weather.reindex(pd.DatetimeIndex(proxy.timestamp)).reset_index(drop=True)
    for c in WEATHER_BASE:
        mask = proxy.timestamp.gt(pd.Timestamp('2025-08-25 05:00', tz='Asia/Shanghai')) & proxy[c].isna()
        proxy.loc[mask, c] = aligned.loc[mask, c]
    station = uniform_weather_rows(full, proxy).set_index('timestamp')
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'arms': ['mixed_no_pbl', 'station_no_pbl'],
        'selection': '2023 station both main MAEs improve matched mixed arm, tail not worse; otherwise stop',
        'weather': 'Shuangliu observed NOAA ISD 2014-2025 then source223 GHCNh after ISD retirement; no interpolation',
        'model': 'observed-prefix hazard 300 trees seed42 unchanged',
        'limitations': ['reused historical diagnostic', 'station vs city representativeness changes',
                       'archival availability not realtime proven', 'both omit PBL; not compared to deployed model']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    proxy.to_parquet(out / 'station_weather.parquet', index=False)
    scores = {}
    for year in (2023, 2024, 2025):
        train = pd.read_parquet(annual / str(year) / 'training_rows.parquet')
        val = pd.read_parquet(dev / 'predictions.parquet' if year == 2023 else annual / str(year) / 'predictions.parquet')
        if year == 2025:
            val[old_names] = pd.read_parquet(repair / 'repaired_features.parquet')[old_names]
        allowed = val.score_allowed.to_numpy() if year == 2023 else np.ones(len(val), dtype=bool)
        results = {}
        for arm in protocol['arms']:
            x = train[names] if arm == 'mixed_no_pbl' else station.loc[train.timestamp, names].set_axis(train.index)
            z = val[names] if arm == 'mixed_no_pbl' else station.loc[val.timestamp, names].set_axis(val.index)
            model = ObservedHazardRegressor(seed=42).fit_bounds(x, train.lower_bound_hours, train.upper_bound_hours)
            joblib.dump(model, out / f'{year}_{arm}.joblib')
            pred = model.predict(z)
            val[arm] = pred
            ok = allowed & np.isfinite(pred)
            results[arm] = metrics(val.loc[ok], pred[ok])
            results[arm]['unresolved'] = int(np.isnan(pred).sum())
        val.to_parquet(out / f'{year}_predictions.parquet', index=False)
        a, b = results['station_no_pbl'], results['mixed_no_pbl']
        passed = bool(a['unresolved'] == b['unresolved'] == 0 and a['row_mae'] < b['row_mae'] and
            a['block_mae'] < b['block_mae'] and a['tail_24h_block_mae'] <= b['tail_24h_block_mae'])
        scores[str(year)] = {'scores': results, 'passed': passed}
        if year == 2023:
            (out / 'selection.json').write_text(json.dumps(scores[str(year)], indent=2), encoding='utf-8')
            if not passed:
                break
    assert before == {str(p): sha(p) for p in paths}
    report = {'status': 'completed_station_ablation', 'scores': scores, 'inputs_unchanged': True,
        'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
