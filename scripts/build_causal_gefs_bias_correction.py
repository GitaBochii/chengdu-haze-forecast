"""Expanding past-only weather bias correction with a fixed observation lag."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def past_bias(errors, origin, minimum=20):
    usable = errors.loc[errors.available_at.le(origin), 'error'].dropna()
    return (float(usable.median()) if len(usable) >= minimum else 0.0), len(usable)


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'reusable_gefs_snapshots_v1'
    out = base / 'causal_gefs_bias_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / 'snapshots.parquet', source / 'features.parquet',
             base / 'observed_business_dataset_v1/features.parquet', Path(__file__),
             ROOT / 'src/haze_forecast/app.py']
    before = {str(p): sha(p) for p in paths}
    snapshots = pd.read_parquet(paths[0])
    fields = ['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']
    means = ['gefs_mean_' + field + '_f024' for field in fields]
    # Every retained forecast was known strictly before its valid time.
    snapshots = snapshots.loc[snapshots.known_at.lt(snapshots.valid_at)].sort_values('known_at')
    for _, group in snapshots.groupby(['run_at', 'valid_at']):
        assert group[means].nunique().le(1).all()
    unique = snapshots.drop_duplicates(['run_at', 'valid_at'], keep='first')
    assert unique.valid_at.is_unique
    weather = pd.read_parquet(paths[2])[['timestamp', *fields]]
    errors = unique[['run_at', 'valid_at', *means]].merge(
        weather.rename(columns={'timestamp': 'valid_at'}), on='valid_at', how='left', validate='one_to_one')
    errors['available_at'] = errors.valid_at + pd.Timedelta(hours=24)
    original = pd.read_parquet(paths[1])
    result = original.copy()
    report_scores = {}
    for field, mean in zip(fields, means):
        table = errors[['available_at']].copy()
        table['error'] = errors[mean] - errors[field]
        biases, counts = zip(*(past_bias(table, t) for t in original.timestamp))
        result[mean] = original[mean] - np.asarray(biases)
        if field == 'relative_humidity_pct':
            result[mean] = result[mean].clip(0, 100)
        result[field + '_bias_applied'] = biases
        result[field + '_bias_sample_count'] = counts
        errors[field + '_error'] = table.error
        # Future observations below are scores only; never inputs to past_bias at this origin.
        evaluated = original[['timestamp', 'valid_at', mean]].merge(
            weather[['timestamp', field]].rename(columns={'timestamp': 'valid_at'}), on='valid_at', how='left', validate='many_to_one')
        evaluated['corrected'] = result[mean].to_numpy()
        yearly = {}
        for year in (2022, 2023, 2024):
            rows = evaluated.loc[evaluated.timestamp.dt.year.eq(year)].dropna(subset=[mean, field, 'corrected'])
            raw_error = (rows[mean] - rows[field]).abs()
            corrected_error = (rows.corrected - rows[field]).abs()
            yearly[str(year)] = {'rows': len(rows), 'valid_times': int(rows.valid_at.nunique()),
                'raw_mae': float(raw_error.mean()), 'corrected_mae': float(corrected_error.mean()),
                'raw_valid_time_mae': float(raw_error.groupby(rows.valid_at).mean().mean()),
                'corrected_valid_time_mae': float(corrected_error.groupby(rows.valid_at).mean().mean())}
        report_scores[field] = yearly
    pd.testing.assert_frame_equal(original.drop(columns=means), result[original.columns].drop(columns=means))
    assert before == {str(p): sha(p) for p in paths}
    out.mkdir(parents=True)
    result.to_parquet(out / 'features.parquet', index=False)
    errors.to_parquet(out / 'historical_errors.parquet', index=False)
    report = {'input_sha256': before, 'unique_forecast_valid_times': len(unique), 'scores': report_scores,
        'inputs_unchanged': True, 'rule': 'expanding median; minimum20; valid time plus24h observation-availability assumption',
        'limitations': ['24h observation delay is an assumption, not publication certification',
            '2025 original weather truncation leaves fewer new errors; no 2025 weather scoring',
            'no duration-model training, no MAE improvement claim for haze', 'forecast archive timestamp proxy retained'],
        'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report_scores, indent=2))


if __name__ == '__main__':
    main()
