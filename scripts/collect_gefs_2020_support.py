"""Bounded resumable collection; every failed attempt is preserved separately."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.collect_gefs_2021_f024 import atomic_json, extract, freeze_protocol, verify_complete

ROOT = Path(__file__).resolve().parents[1]


def plan_runs(frame):
    times = pd.to_datetime(frame.timestamp)
    if times.dt.tz is None or times.isna().any() or not times.is_unique:
        raise ValueError('Require unique aware nonmissing origins')
    local = times.dt.tz_convert('Asia/Shanghai')
    mask = local.dt.year.eq(2020) & local.ge(pd.Timestamp('2020-09-23', tz='Asia/Shanghai'))
    return sorted(pd.Timestamp(v) for v in (local[mask].dt.tz_convert('UTC') - pd.Timedelta(hours=4)).dt.floor('6h').unique())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-runs', type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.max_runs <= 8:
        raise ValueError('At most eight new attempts per invocation')
    b = ROOT / 'artifacts/task01_recovery_head'
    out = b / 'gefs_2020_support_collection_20260918'
    source = b / 'gefs_2021_extension_v1/f024_training_rows.parquet'
    binary = ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    probe = b / 'gefs_2020_support_probe_20260918'
    files = [source, binary, Path(__file__).resolve(), ROOT / 'gefs_2020_collection_plan_20260918.md',
             ROOT / 'scripts/collect_gefs_2021_f024.py', ROOT / 'src/haze_forecast/grib.py',
             ROOT / 'src/haze_forecast/forecast_sources.py', probe / 'report.json']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    runs = plan_runs(pd.read_parquet(source))
    assert len(runs) == 56
    out.mkdir(exist_ok=True)
    protocol = {'input_sha256': hashes, 'runs': [r.isoformat() for r in runs],
                'max_attempts_per_run': 2, 'lead_hours': 24, 'fields_per_run': 9}
    freeze_protocol(out, protocol)
    probe_report = json.loads((probe / 'report.json').read_bytes())
    for p,h in probe_report['output_sha256'].items():
        assert hashlib.sha256((probe / p).read_bytes()).hexdigest() == h
    completed, exhausted = {}, []
    for run in runs:
        key = run.strftime('%Y%m%d%H')
        prior = probe / key
        attempts = sorted((out / key).glob('attempt-[12]'))
        for folder in [prior, *attempts]:
            if (folder / 'complete.json').exists():
                verify_complete(folder, run)
                completed[run.isoformat()] = str(folder)
                break
        if run.isoformat() not in completed and len(attempts) >= 2:
            exhausted.append(run.isoformat())
    attempted = []
    try:
        pending = [r for r in runs if r.isoformat() not in completed and r.isoformat() not in exhausted]
        for run in pending[:args.max_runs]:
            parent = out / run.strftime('%Y%m%d%H')
            parent.mkdir(exist_ok=True)
            folder = parent / f'attempt-{len(list(parent.glob("attempt-[12]"))) + 1}'
            attempted.append(run.isoformat())
            try:
                extract(run, folder, binary)
                verify_complete(folder, run)
                completed[run.isoformat()] = str(folder)
                print(f'completed {run.isoformat()} {len(completed)}/56', flush=True)
            except Exception as error:
                atomic_json(parent / f'{folder.name}-error.json', {'error': repr(error), 'run': run.isoformat()})
                print(f'failed {run.isoformat()} {error!r}', flush=True)
                if len(list(parent.glob('attempt-[12]'))) >= 2:
                    exhausted.append(run.isoformat())
    finally:
        for p,h in hashes.items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
        status = {'planned_runs': len(runs), 'completed_runs': len(completed),
                  'remaining_runs': len(runs)-len(completed), 'attempted': attempted,
                  'completed_directories': completed, 'exhausted_runs': exhausted,
                  'independently_audited_collection_runs': 0, 'model_evaluation_allowed': False}
        atomic_json(out / 'status.json', status)
        print(json.dumps({k:v for k,v in status.items() if k != 'completed_directories'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
