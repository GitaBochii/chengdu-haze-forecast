"""Measure historical trajectory truncation without reading future observations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'trajectory_reset_audit_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [base / 'observed_business_dataset_v1/features.parquet',
             base / 'interval_business_labels_v1/origins.parquet', Path(__file__),
             ROOT / 'src/haze_forecast/features.py', ROOT / 'src/haze_forecast/observed_business.py']
    before = {str(p): sha(p) for p in paths}
    full = pd.read_parquet(paths[0])
    origins = pd.read_parquet(paths[1])
    active = full.aqi.gt(100) & full.particle_primary.fillna(False)
    rows = []
    previous_block = None
    first_high = None
    high_count = missing_count = short_clean_count = unknown_particle_count = 0
    for row, particle in zip(full.itertuples(), active):
        if row.continuity_block != previous_block:
            first_high = None
            high_count = missing_count = short_clean_count = unknown_particle_count = 0
        if np.isfinite(row.aqi) and row.aqi > 100 and first_high is None:
            first_high = row.timestamp
        if first_high is not None:
            high_count += int(np.isfinite(row.aqi) and row.aqi > 100)
            missing_count += int(not np.isfinite(row.aqi))
            short_clean_count += int(np.isfinite(row.aqi) and row.aqi <= 100)
            unknown_particle_count += int(np.isfinite(row.aqi) and row.aqi > 100 and not particle)
        if row.eligible:
            elapsed = (row.timestamp - first_high).total_seconds() / 3600 + 1
            rows.append({'timestamp': row.timestamp, 'continuity_block': row.continuity_block,
                'run_age': row.pollution_age_hours, 'hours_since_first_observed_high': elapsed,
                'observed_high_hours': high_count, 'past_missing_hours': missing_count,
                'past_short_clean_hours': short_clean_count,
                'past_high_nonparticle_hours': unknown_particle_count,
                'truncated_history': elapsed > row.pollution_age_hours})
        previous_block = row.continuity_block
    audit = pd.DataFrame(rows)
    np.testing.assert_array_equal(audit.timestamp, origins.timestamp)
    audit = audit.merge(origins[['timestamp', 'exact_hours', 'lower_bound_hours']], on='timestamp', validate='one_to_one')
    results = {}
    for year, frame in audit.groupby(audit.timestamp.dt.year):
        results[str(year)] = {'origins': len(frame), 'history_truncated': int(frame.truncated_history.sum()),
            'truncated_with_prior_missing': int((frame.truncated_history & frame.past_missing_hours.gt(0)).sum()),
            'truncated_without_prior_missing': int((frame.truncated_history & frame.past_missing_hours.eq(0)).sum())}
    out.mkdir(parents=True)
    audit.to_parquet(out / 'origins.parquet', index=False)
    winter = audit.loc[audit.timestamp.dt.year.eq(2025) & audit.timestamp.dt.month.ge(11)]
    blocks = winter.groupby('continuity_block').agg(origins=('timestamp', 'size'),
        first_origin=('timestamp', 'min'), truncated=('truncated_history', 'sum'),
        prior_missing_max=('past_missing_hours', 'max'), run_age_mean=('run_age', 'mean'),
        observed_high_mean=('observed_high_hours', 'mean'),
        elapsed_mean=('hours_since_first_observed_high', 'mean'))
    blocks.to_csv(out / 'winter_2025_blocks.csv')
    assert before == {str(p): sha(p) for p in paths}
    report = {'input_sha256': before, 'yearly': results, 'inputs_unchanged': True,
        'limitations': ['elapsed history is not proven uninterrupted pollution duration',
            'only past measurements counted; future label columns joined after historical computation',
            'nonparticle means AQI high but old particle_primary is false or unknown',
            'association audit, not evidence that longer memory improves prediction'],
        'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    print(blocks.to_string())


if __name__ == '__main__':
    main()
