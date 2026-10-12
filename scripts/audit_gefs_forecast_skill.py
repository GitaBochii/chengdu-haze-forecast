"""Check forecast meteorology against observed valid-time weather, not haze labels."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'gefs_forecast_skill_audit_v1'
    if out.exists():
        raise FileExistsError(out)
    forecast_path = base / 'reusable_gefs_snapshots_v1/features.parquet'
    weather_path = base / 'observed_business_dataset_v1/features.parquet'
    paths = [forecast_path, weather_path, Path(__file__),
             base / 'reusable_gefs_crossyear_v1/2024_common.parquet', ROOT / 'src/haze_forecast/app.py']
    before = {str(p): sha(p) for p in paths}
    forecast = pd.read_parquet(forecast_path)
    fields = ['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']
    weather = pd.read_parquet(weather_path)[['timestamp', *fields]]
    # 2025 original weather archive is truncated; avoid claiming an uncorrected comparison.
    forecast = forecast.loc[forecast.timestamp.dt.year.between(2022, 2024) & forecast.known_at.notna()].copy()
    frame = forecast.merge(weather.rename(columns={c: 'origin_' + c for c in fields}), on='timestamp', validate='one_to_one')
    frame = frame.merge(weather.rename(columns={'timestamp': 'valid_at', **{c: 'actual_' + c for c in fields}}),
                        on='valid_at', how='left', validate='many_to_one')
    results = {}
    for year, part in frame.groupby(frame.timestamp.dt.year):
        scores = {}
        for field in fields:
            predicted = 'gefs_mean_' + field + '_f024'
            usable = part[[predicted, 'origin_' + field, 'actual_' + field]].notna().all(axis=1)
            sample = part.loc[usable].copy()
            sample['forecast_error'] = (sample[predicted] - sample['actual_' + field]).abs()
            sample['persistence_error'] = (sample['origin_' + field] - sample['actual_' + field]).abs()
            sample['forecast_bias'] = sample[predicted] - sample['actual_' + field]
            scores[field] = {'available_origins': len(part), 'scored_origins': len(sample),
                'unique_valid_times': int(sample.valid_at.nunique()),
                'forecast_mae': float(sample.forecast_error.mean()),
                'persistence_mae': float(sample.persistence_error.mean()),
                'valid_time_balanced_forecast_mae': float(sample.groupby('valid_at').forecast_error.mean().mean()),
                'valid_time_balanced_persistence_mae': float(sample.groupby('valid_at').persistence_error.mean().mean()),
                'forecast_bias': float(sample.forecast_bias.mean())}
        results[str(year)] = scores
    haze = pd.read_parquet(paths[3]).merge(forecast[['timestamp', 'forecast_remaining_lead_hours']], on='timestamp', how='left', validate='one_to_one')
    haze = haze.loc[haze.exact_hours.notna()].copy()
    haze['forecast_error'] = (haze.forecast_aft - haze.exact_hours).abs()
    haze['control_error'] = (haze.history_aft - haze.exact_hours).abs()
    haze['error_difference'] = haze.forecast_error - haze.control_error
    blocks = haze.groupby('continuity_block').agg(origins=('timestamp', 'size'), first_origin=('timestamp', 'min'),
        forecast_mae=('forecast_error', 'mean'), control_mae=('control_error', 'mean'),
        difference=('error_difference', 'mean'), remaining_lead_mean=('forecast_remaining_lead_hours', 'mean'))
    out.mkdir(parents=True)
    frame.to_parquet(out / 'weather_comparison.parquet', index=False)
    blocks.to_csv(out / '2024_blocks.csv')
    assert before == {str(p): sha(p) for p in paths}
    report = {'input_sha256': before, 'years': results, 'inputs_unchanged': True,
              'limitations': ['measurement availability selects scoring subset; no weather imputation',
                  'future actual weather used for scoring only', 'grid forecast versus station observations',
                  'weather error does not establish cause of haze-duration error',
                  'repeated valid times additionally scored with equal valid-time weights', '2025 excluded due to original weather truncation'],
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
