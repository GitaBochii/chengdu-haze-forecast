"""Quantify forecast horizon and effective training coverage against recovery bounds."""

import json
from pathlib import Path

import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'forecast_target_horizon_v1'
    if out.exists():
        raise FileExistsError(out)
    origin_path = base / 'interval_business_labels_v1/origins.parquet'
    forecast_path = base / 'reusable_gefs_snapshots_v1/features.parquet'
    train_paths = {2023: base / 'interval_business_aft_v1/training_rows.parquet',
                   **{y: base / f'interval_crossyear_v1/{y}/training_rows.parquet' for y in (2024, 2025)}}
    paths = [origin_path, forecast_path, *train_paths.values(), Path(__file__), ROOT / 'src/haze_forecast/app.py']
    before = {str(p): sha(p) for p in paths}
    forecasts = pd.read_parquet(forecast_path)
    origins = pd.read_parquet(origin_path)
    columns = ['timestamp', 'known_at', 'valid_at', 'forecast_remaining_lead_hours']
    merged = origins[['timestamp', 'continuity_block', 'lower_bound_hours', 'upper_bound_hours', 'exact_hours']].merge(
        forecasts[columns], on='timestamp', how='left', validate='one_to_one')
    reports = {}
    for year, path in train_paths.items():
        train = pd.read_parquet(path)[['timestamp', 'continuity_block']].merge(forecasts[columns], on='timestamp', validate='one_to_one')
        covered = train.loc[train.known_at.notna()]
        val = merged.loc[merged.timestamp.dt.year.eq(year)]
        available = val.loc[val.known_at.notna()]
        long = available.loc[available.exact_hours.ge(24)]
        reports[str(year)] = {'train_rows': len(train), 'train_blocks': int(train.continuity_block.nunique()),
            'forecast_train_rows': len(covered), 'forecast_train_blocks': int(covered.continuity_block.nunique()),
            'validation_origins': len(val), 'forecast_origins': len(available),
            'remaining_lead_min': float(available.forecast_remaining_lead_hours.min()),
            'remaining_lead_max': float(available.forecast_remaining_lead_hours.max()),
            'guaranteed_beyond_forecast_endpoint': int(available.lower_bound_hours.gt(available.forecast_remaining_lead_hours).sum()),
            'possibly_beyond_forecast_endpoint': int(available.upper_bound_hours.gt(available.forecast_remaining_lead_hours).sum()),
            'exact_long_origins': len(long),
            'exact_long_beyond_forecast_endpoint': int(long.exact_hours.gt(long.forecast_remaining_lead_hours).sum())}
    out.mkdir(parents=True)
    merged.to_parquet(out / 'origin_coverage.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'input_sha256': before, 'years': reports, 'inputs_unchanged': True,
        'limitations': ['endpoint mismatch does not prove longer forecasts improve recovery predictions',
            'process blocks are grouping units, not proven statistically independent events',
            'all available forecasts have a single f024 endpoint; not a dense trajectory'],
        'output_sha256': sha(out / 'origin_coverage.parquet')}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
