"""Frozen 2025 model replay changing only post-retirement observed weather."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.ghcnh_weather import prepare_ghcnh_weather
from haze_forecast.observed_hourly_features import build_observed_hourly_features
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    out = ROOT / 'artifacts/task01_recovery_head/ghcnh_weather_repair_v1'
    if out.exists():
        raise FileExistsError(out)
    data = ROOT / 'artifacts/task01_recovery_head/observed_business_dataset_v1'
    annual = ROOT / 'artifacts/task01_recovery_head/observed_annual_update_v1/2025'
    hazard = ROOT / 'artifacts/task01_recovery_head/observed_hazard_crossyear_v1'
    raw_path = ROOT / 'data/raw/ghcnh_discovery_20260906/GHCNh_CHI0000ZUUU_2025.parquet'
    model_paths = {k: annual / f'{k}.joblib' for k in ['legacy_training', 'exact_only', 'observed_bounds']}
    model_paths['hazard'] = hazard / '2025.joblib'
    paths = [raw_path, data / 'features.parquet', data / 'origins.parquet', data / 'report.json',
        annual / 'predictions.parquet', hazard / '2025_predictions.parquet', Path(__file__),
        ROOT / 'src/haze_forecast/ghcnh_weather.py', ROOT / 'src/haze_forecast/cnemc_weather.py',
        ROOT / 'src/haze_forecast/observed_hourly_features.py', ROOT / 'src/haze_forecast/features.py',
        ROOT / 'src/haze_forecast/labels.py', ROOT / 'src/haze_forecast/app.py', *model_paths.values()]
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    full = pd.read_parquet(data / 'features.parquet')
    names = json.loads((data / 'report.json').read_bytes())['feature_names']
    weather = prepare_ghcnh_weather(pd.read_parquet(raw_path)).set_index('timestamp')
    cutoff = pd.Timestamp('2025-08-25 05:00', tz='Asia/Shanghai')
    columns = ['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']
    aligned = weather.reindex(pd.DatetimeIndex(full.timestamp)).reset_index(drop=True)
    overlap = {}
    patched = full.copy()
    counts = {}
    for column in columns:
        both = full.timestamp.le(cutoff) & full[column].notna() & aligned[column].notna()
        overlap[column] = {'rows': int(both.sum()),
            'mae_difference': float((full.loc[both, column] - aligned.loc[both, column]).abs().mean())}
        mask = full.timestamp.gt(cutoff) & full[column].isna() & aligned[column].notna()
        patched.loc[mask, column] = aligned.loc[mask, column]
        counts[column] = int(mask.sum())
    rebuilt = build_observed_hourly_features(patched)
    # PBL is untouched; preserve its audited missingness indicators exactly.
    for name in names:
        if name.endswith('_is_missing'):
            rebuilt[name] = full[name]
    pd.testing.assert_frame_equal(rebuilt.loc[full.timestamp.le(cutoff), names], full.loc[full.timestamp.le(cutoff), names])
    origins = pd.read_parquet(data / 'origins.parquet')
    val = origins.loc[origins.timestamp.dt.year.eq(2025)].copy().reset_index(drop=True)
    new_x = rebuilt.set_index('timestamp').loc[val.timestamp, names].reset_index(drop=True)
    old_predictions = pd.read_parquet(annual / 'predictions.parquet')
    old_predictions['hazard'] = pd.read_parquet(hazard / '2025_predictions.parquet').hazard
    np.testing.assert_array_equal(val.timestamp, old_predictions.timestamp)
    results = {}
    for arm, path in model_paths.items():
        model = joblib.load(path)
        original = model.predict(val[names])
        np.testing.assert_array_equal(original, old_predictions[arm])
        pred = model.predict(new_x)
        np.testing.assert_array_equal(pred[val.timestamp.le(cutoff)], original[val.timestamp.le(cutoff)])
        assert np.isfinite(pred).all()
        val[arm + '_before'] = original
        val[arm + '_after'] = pred
        affected = val.timestamp.gt(cutoff)
        results[arm] = {'before': metrics(val, original), 'after': metrics(val, pred),
            'late_before': metrics(val.loc[affected], original[affected]),
            'late_after': metrics(val.loc[affected], pred[affected])}
    assert before == {str(p): sha(p) for p in paths}
    out.mkdir(parents=True)
    new_x.to_parquet(out / 'repaired_features.parquet', index=False)
    val.to_parquet(out / 'predictions.parquet', index=False)
    weather.reset_index().to_parquet(out / 'quality_filtered_weather.parquet', index=False)
    report = {'status': 'completed_frozen_input_repair', 'input_sha256': before, 'features': names,
        'patch_after': str(cutoff), 'patched_grid_cells': counts, 'overlap': overlap,
        'results': results, 'inputs_unchanged': True,
        'limitations': ['2025 previously inspected; diagnostic not pristine holdout',
            'archive timestamp does not establish real-time publication', 'no model refit or deployment',
            'source223 quality1 only; missing and suspect excluded; no interpolation',
            'full-year exact scores still exclude censored outcomes'],
        'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'overlap': overlap, 'patched': counts, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
