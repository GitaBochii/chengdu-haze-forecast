"""Reuse complete audited f024 snapshots at causally compatible later origins."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.forecast_snapshot_reuse import reuse_forecast_snapshots
from haze_forecast.operational_gefs_benchmark import publication_audited_features
from haze_forecast.operational_gefs_dataset import OPERATIONAL_GEFS_FEATURES
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'reusable_gefs_snapshots_v1'
    if out.exists():
        raise FileExistsError(out)
    sources = [ROOT / f'data/processed/operational_gefs_audited/operational_gefs_event_features_{y}.parquet'
               for y in (2022, 2023, 2024, 2025)]
    origins_path = base / 'interval_business_labels_v1/origins.parquet'
    old_report_path = ROOT / 'artifacts/operational_gefs_ensemble/report.json'
    paths = sources + [origins_path, old_report_path, Path(__file__),
        ROOT / 'src/haze_forecast/forecast_snapshot_reuse.py', ROOT / 'src/haze_forecast/operational_gefs_benchmark.py',
        ROOT / 'src/haze_forecast/app.py']
    before = {str(p): sha(p) for p in paths}
    old = json.loads(old_report_path.read_bytes())
    fields = [n for n in OPERATIONAL_GEFS_FEATURES if n.endswith('f024')]
    parts = []
    for path in sources:
        assert sha(path) == old['audited_gefs_sha256'][path.name]
        raw = pd.read_parquet(path)
        safe = publication_audited_features(raw)
        complete = safe[fields].notna().all(axis=1)
        snapshots = safe.loc[complete, ['timestamp', *fields]].rename(columns={'timestamp': 'known_at'})
        snapshots['run_at'] = pd.to_datetime(raw.loc[complete, 'gefs_run_utc'], utc=True)
        snapshots['valid_at'] = pd.to_datetime(raw.loc[complete, 'gefs_forecast_valid_utc_f024'], utc=True)
        snapshots['known_at'] = pd.to_datetime(snapshots.known_at, utc=True)
        parts.append(snapshots)
    snapshots = pd.concat(parts, ignore_index=True).sort_values('known_at')
    origins = pd.read_parquet(origins_path)
    result = reuse_forecast_snapshots(snapshots, origins.timestamp, fields)
    np.testing.assert_array_equal(result.timestamp, origins.timestamp)
    available = result.known_at.notna()
    assert result.loc[available, 'known_at'].le(result.loc[available, 'timestamp']).all()
    assert result.loc[available, 'valid_at'].gt(result.loc[available, 'timestamp']).all()
    yearly = {}
    for year, indices in result.groupby(result.timestamp.dt.year).groups.items():
        frame = result.loc[indices]
        yearly[str(year)] = {'origins': len(frame), 'reusable_forecast': int(frame.known_at.notna().sum()),
            'same_timestamp': int(frame.known_at.eq(frame.timestamp).sum())}
    assert before == {str(p): sha(p) for p in paths}
    out.mkdir(parents=True)
    snapshots.to_parquet(out / 'snapshots.parquet', index=False)
    result.to_parquet(out / 'features.parquet', index=False)
    report = {'input_sha256': before, 'fields': fields, 'yearly': yearly, 'snapshots': len(snapshots),
        'inputs_unchanged': True, 'limitations': ['known_at is a conservative audited origin, not exact publication time',
            'archive timestamp proxy only; not operational certification', 'forecasts reused without interpolation',
            'f024 is relative to run; remaining lead supplied explicitly', 'no training or accuracy claim'],
        'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'yearly': yearly, 'snapshots': len(snapshots)}, indent=2))


if __name__ == '__main__':
    main()
