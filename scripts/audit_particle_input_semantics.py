"""Independently reproduce particle IAQI without changing labels or models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    source = ROOT / 'artifacts/task01_recovery_head/observed_business_dataset_v1/features.parquet'
    out = ROOT / 'artifacts/task01_recovery_head/particle_input_semantics_audit_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [source, Path(__file__), ROOT / 'src/haze_forecast/labels.py',
             ROOT / 'src/haze_forecast/observed_hourly_features.py', ROOT / 'src/haze_forecast/app.py']
    inputs = {str(p): sha(p) for p in paths}
    frame = pd.read_parquet(source)
    if frame.timestamp.duplicated().any() or not frame.timestamp.diff().iloc[1:].eq(
            pd.Timedelta(hours=1)).all():
        raise ValueError('Expected one complete hourly grid')
    matches = {}
    for raw, saved, breaks in [
        ('pm2_5', 'pm2_5_iaqi', [0, 35, 75, 115, 150, 250, 350, 500]),
        ('pm10', 'pm10_iaqi', [0, 50, 150, 250, 350, 420, 500, 600]),
    ]:
        average = frame[raw].rolling(24, min_periods=24).mean()
        predicted = np.round(np.interp(average, breaks, [0, 50, 100, 150, 200, 300, 400, 500]))
        predicted[(average < 0) | average.isna()] = np.nan
        np.testing.assert_allclose(predicted, frame[saved], equal_nan=True, rtol=0, atol=0)
        matches[saved] = {'matching_rows': len(frame), 'missing_rows': int(frame[saved].isna().sum())}
    particle = frame[['pm2_5_iaqi', 'pm10_iaqi']].max(axis=1)
    expected = frame.aqi.gt(100) & particle.gt(100)
    np.testing.assert_array_equal(expected, frame.particle_primary)
    rows = []
    for year, group in frame.groupby(frame.timestamp.dt.year):
        valid = group.aqi.notna() & particle.loc[group.index].notna()
        gap = particle.loc[group.index] - group.aqi
        rows.append({'year': int(year), 'valid_pairs': int(valid.sum()),
                     'particle_iaqi_exceeds_aqi_by_over_one': int((valid & gap.gt(1)).sum()),
                     'aqi_high_particle_not_high': int((group.aqi.gt(100) & particle.loc[group.index].le(100)).sum())})
    assert inputs == {str(p): sha(p) for p in paths}
    out.mkdir()
    report = {'input_sha256': inputs, 'rows': len(frame), 'independent_recalculation': matches,
              'particle_flag_mismatches': int(expected.ne(frame.particle_primary).sum()),
              'years': rows, 'deployment_authorized': False,
              'interpretation': 'Recalculation consistency only, not certification of source AQI semantics.',
              'particle_flag_semantics': 'AQI > 100 and either particle 24-hour IAQI > 100, not an all-pollutant maximum comparison.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
