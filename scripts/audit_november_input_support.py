"""Descriptive support audit for a previously selected error block; no fitting."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_temporal_calibration_data_v1'
    out = base / 'november_input_support_v1'
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((source / 'report.json').read_bytes())['output_sha256']
    for name, digest in manifest.items():
        assert sha(source / name) == digest
    grid_path = base / 'observed_business_dataset_v1/features.parquet'
    paths = [source / name for name in manifest]
    paths += [source / 'report.json', grid_path, Path(__file__),
              ROOT / 'src/haze_forecast/app.py']
    paths += [p for p in (ROOT / 'artifacts/cnemc_2025_deployment_candidate').iterdir() if p.is_file()]
    before = {str(p): sha(p) for p in paths}
    train = pd.read_parquet(source / 'training_rows.parquet')
    val = pd.read_parquet(source / 'predictions.parquet')
    query = val.loc[val.continuity_block.eq(1232)].copy()
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    assert len(query) == 30 and query.timestamp.dt.year.eq(2022).all()
    model = joblib.load(source / 'model.joblib')
    np.testing.assert_array_equal(model.predict(val[names]), val.forecast_aft)
    forecast = [n for n in names if n.startswith('gefs_')]
    features = []
    for name in names:
        ref = train[name].dropna().astype(float)
        values = query[name].dropna().astype(float)
        lo, hi = ref.quantile([.01, .99])
        features.append({'feature': name, 'training_missing_fraction': float(train[name].isna().mean()),
                         'block_missing_fraction': float(query[name].isna().mean()),
                         'training_p01': lo, 'training_p99': hi,
                         'block_min': values.min(), 'block_max': values.max(),
                         'outside_training_p01_p99_count': int(((values < lo) | (values > hi)).sum())})
    # A deliberately limited analogue definition: current seven-pollutant state only.
    cols = ['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3']
    eligible = train[cols].notna().all(axis=1)
    donors = train.loc[eligible].copy()
    assert query[cols].notna().all().all()
    scale = donors[cols].quantile(.75) - donors[cols].quantile(.25)
    assert scale.gt(0).all()
    analogues = []
    for row in query.itertuples():
        values = query.loc[query.timestamp.eq(row.timestamp), cols].iloc[0]
        distances = (((donors[cols] - values) / scale) ** 2).mean(axis=1) ** .5
        ranked = donors.assign(distance=distances).sort_values('distance', kind='stable')
        chosen = ranked.drop_duplicates('continuity_block').head(5)
        for rank, neighbor in enumerate(chosen.itertuples(), start=1):
            analogues.append({'query_timestamp': row.timestamp, 'rank': rank,
                              'training_timestamp': neighbor.timestamp,
                              'training_block': neighbor.continuity_block,
                              'distance': neighbor.distance,
                              'lower_bound_hours': neighbor.lower_bound_hours,
                              'upper_bound_hours': neighbor.upper_bound_hours})
    neighbors = pd.DataFrame(analogues)
    # Future observations below are outcome auditing only, never model inputs.
    grid = pd.read_parquet(grid_path).set_index('timestamp')
    start = query.timestamp.min()
    end = max(query.timestamp + pd.to_timedelta(query.exact_hours + 5, unit='h'))
    window = grid.loc[start:end, ['aqi']].copy()
    assert len(window) == int((end - start).total_seconds() / 3600) + 1
    clean = window.aqi.le(100) & window.aqi.notna()
    runs = window.assign(clean=clean, group=clean.ne(clean.shift()).cumsum())
    clean_runs = []
    for _, group in runs.loc[runs.clean].groupby('group'):
        clean_runs.append({'start': str(group.index.min()), 'hours': len(group)})
    for row in query.itertuples():
        future = grid.loc[row.timestamp:end, 'aqi']
        six = future.le(100).rolling(6, min_periods=6).sum().eq(6)
        first_start = six.loc[six].index[0] - pd.Timedelta(hours=5)
        assert (first_start - row.timestamp).total_seconds() / 3600 == row.exact_hours
    out.mkdir()
    pd.DataFrame(features).to_csv(out / 'feature_support.csv', index=False)
    neighbors.to_parquet(out / 'pollution_analogues.parquet', index=False)
    query[['timestamp', 'exact_hours', 'forecast_aft', *cols, *forecast]].to_parquet(
        out / 'origin_trace.parquet', index=False)
    window.to_csv(out / 'outcome_aqi.csv')
    err = query.forecast_aft - query.exact_hours
    result = {'input_sha256': before, 'block': 1232, 'rows': len(query),
              'baseline_mae': float(err.abs().mean()), 'baseline_bias': float(err.mean()),
              'forecast_complete_rows': int(query[forecast].notna().all(axis=1).sum()),
              'forecast_fields': len(forecast),
              'training_forecast_complete_rows': int(train[forecast].notna().all(axis=1).sum()),
              'training_forecast_complete_blocks': int(train.loc[train[forecast].notna().all(axis=1),
                                                                'continuity_block'].nunique()),
              'current_wind_missing_rows': int(query.wind_u_ms.isna().sum()),
              'observed_pbl_missing_rows': int(query.boundary_layer_height_m.isna().sum()),
              'outcome_aqi_missing_hours': int(window.aqi.isna().sum()), 'clean_runs': clean_runs,
              'analogue_definition': 'Five distinct blocks, nearest current seven-pollutant IQR-scaled RMS',
              'analogue_training_rows_available': len(donors), 'analogue_iqr': scale.to_dict(),
              'query_rows_with_at_least_one_definite_24h_analogue': int(
                  neighbors.lower_bound_hours.ge(24).groupby(neighbors.query_timestamp).any().sum()),
              'limitations': ['Post-selected development error block, not a generalization test',
                              'Analogues ignore weather, season and history; not conditional recovery estimates',
                              'Univariate percentile exceedance is not proof of multivariate distribution shift',
                              'No imputation, retraining, prediction correction or deployment'],
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    assert before == {str(p): sha(Path(p)) for p in before}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_sha256', 'output_sha256')}, indent=2))


if __name__ == '__main__':
    main()
