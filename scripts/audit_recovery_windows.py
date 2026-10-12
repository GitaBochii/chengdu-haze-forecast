"""Independent six-hour enumeration and causal-input coverage on a frozen cohort."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def scan_windows(values):
    values = np.asarray(values, dtype=float)
    padded = np.r_[values, np.full(6, np.nan)]
    earliest, confirmed = None, np.inf
    for start in range(1, len(values) + 1):
        window = padded[start:start + 6]
        if earliest is None and not np.any(window > 100):
            earliest = start
        if len(window) == 6 and np.all(np.isfinite(window)) and np.all(window <= 100):
            confirmed = start
            break
    return float(earliest), float(confirmed)


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'grouped_forecast_aft_2023_20260916'
    out = base / 'recovery_window_audit_2023_20260916'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / 'expanded.parquet', source / 'protocol.json', source / 'report.json',
             base / 'observed_business_dataset_v1/features.parquet',
             base / 'observed_business_dataset_v1/report.json',
             ROOT / 'data/interim/cnemc_chengdu_2023_weather.parquet', Path(__file__).resolve(),
             base / 'gefs_2021_extension_v1/inference_rows.parquet']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    dataset_report = json.loads(paths[4].read_bytes())
    if hashes[str(paths[3])] != dataset_report['output_sha256']['features.parquet']:
        raise ValueError('Frozen hourly features changed')
    if hashes[str(paths[5])] != dataset_report['input_sha256'][str(paths[5])]:
        raise ValueError('Original 2023 source changed')
    experiment = json.loads(paths[2].read_bytes())
    if hashes[str(paths[0])] != experiment['output_sha256']['expanded.parquet']:
        raise ValueError('Scored cohort changed')
    frame = pd.read_parquet(paths[0])
    full = pd.read_parquet(paths[3]).set_index('timestamp')
    raw = pd.read_parquet(paths[5]).set_index('timestamp')
    protocol = json.loads(paths[1].read_bytes())
    names = protocol['features']
    if hashes[str(paths[7])] != protocol['input_sha256'][str(paths[7])]:
        raise ValueError('Frozen inference inputs changed')
    inference = pd.read_parquet(paths[7]).set_index('timestamp')
    if not full.index.is_unique or not raw.index.is_unique:
        raise ValueError('Duplicate hourly timestamps')
    pollutants = ['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3']
    shared = raw.index.intersection(full.index)
    np.testing.assert_allclose(raw.loc[shared, pollutants], full.loc[shared, pollutants],
                               rtol=0, atol=0, equal_nan=True)
    rows = []
    for _, origin in frame.iterrows():
        timestamp = pd.Timestamp(origin.timestamp)
        end = timestamp + pd.Timedelta(hours=float(origin.upper_bound_hours) + 5)
        grid = pd.date_range(timestamp, end, freq='h')
        hourly = raw.reindex(grid)
        lo, hi = scan_windows(hourly.aqi.to_numpy())
        np.testing.assert_array_equal([lo, hi], [origin.lower_bound_hours, origin.upper_bound_hours])
        past = raw.reindex(pd.date_range(timestamp - pd.Timedelta(hours=23), timestamp, freq='h'))
        prediction = float(origin.forecast_aft)
        rows.append({'timestamp': timestamp, 'continuity_block': int(origin.continuity_block),
                     'lower_bound_hours': lo, 'upper_bound_hours': hi,
                     'baseline': prediction,
                     'minimum_error': max(lo - prediction, prediction - hi, 0),
                     'below_earliest_recovery': prediction < lo,
                     'past24_complete_pollution_hours': int(past[pollutants].notna().all(axis=1).sum()),
                     'past24_complete_weather_hours': int(past[
                         ['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']
                     ].notna().all(axis=1).sum()),
                     'forecast_values_present': int(inference.loc[timestamp, names[92:101]].notna().sum()),
                     'future_aqi_missing_hours': int(hourly.aqi.iloc[1:].isna().sum()),
                     'interval_width_hours': hi - lo})
    result = pd.DataFrame(rows)
    blocks = result.groupby('continuity_block').agg(
        rows=('timestamp', 'size'), start=('timestamp', 'min'), end=('timestamp', 'max'),
        minimum_mae=('minimum_error', 'mean'), too_early_fraction=('below_earliest_recovery', 'mean'),
        median_interval_width=('interval_width_hours', 'median'),
        min_past24_pollution_hours=('past24_complete_pollution_hours', 'min'),
        min_past24_weather_hours=('past24_complete_weather_hours', 'min'),
        median_past24_weather_hours=('past24_complete_weather_hours', 'median'),
        min_forecast_values=('forecast_values_present', 'min'),
        max_future_missing=('future_aqi_missing_hours', 'max')).sort_values('minimum_mae', ascending=False)
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Input changed: {path}')
    out.mkdir()
    result.to_parquet(out / 'origins.parquet', index=False)
    blocks.to_csv(out / 'blocks.csv')
    report = {'input_sha256': hashes, 'verified_origins': len(result), 'verified_blocks': len(blocks),
              'all_bounds_independently_reproduced': True, 'source_pollutants_match_features': True,
              'limitations': ['Source is archived city-hour table, not a raw station audit',
                              'Nonmissing weather is not proof of quality or real-time availability',
                              'Future missingness and label widths are retrospective diagnostics, never predictors',
                              'No model training or cohort filtering; errors are lower bounds where labels are intervals'],
              'deployment_authorized': False,
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(blocks.head(5).to_string())


if __name__ == '__main__':
    main()
