"""Independently decode all completed batches, then verify timestamp-only matching."""

import hashlib
import json
import shutil
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd

from haze_forecast.forecast_snapshot_reuse import reuse_forecast_snapshots
from scripts.audit_gefs_2021_f024 import audit_run
from scripts.audit_gefs_72h_matching import verify_matches

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    b = ROOT / 'artifacts/task01_recovery_head'
    collection = b / 'gefs_2020_support_collection_20260918'
    out = b / 'gefs_2020_support_audit_20260918'
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads((collection / 'protocol.json').read_bytes())
    status = json.loads((collection / 'status.json').read_bytes())
    assert status['completed_runs'] == 56 and status['remaining_runs'] == 0
    assert set(status['completed_directories']) == set(protocol['runs'])
    for p,h in protocol['input_sha256'].items():
        assert digest(Path(p)) == h
    source = b / 'gefs_2021_extension_v1/f024_training_rows.parquet'
    data = pd.read_parquet(source)
    origins = []
    runs = set()
    for t in data.timestamp:
        local = t.tz_convert('Asia/Shanghai')
        if local.year == 2020 and (local.month, local.day) >= (9, 23):
            origins.append(t)
            lagged = local.to_pydatetime().astimezone(timezone.utc) - timedelta(hours=4)
            runs.add(lagged.replace(hour=lagged.hour//6*6, minute=0, second=0, microsecond=0).isoformat())
    assert sorted(runs) == protocol['runs'] and len(origins) == 273
    binary = ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    dependencies = [source, binary, collection / 'protocol.json', collection / 'status.json',
                    Path(__file__).resolve(), ROOT / 'scripts/audit_gefs_2021_f024.py',
                    ROOT / 'scripts/audit_gefs_72h_matching.py',
                    ROOT / 'src/haze_forecast/forecast_snapshot_reuse.py']
    hashes = {str(p): digest(p) for p in dependencies}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({'input_sha256': hashes, 'runs': sorted(runs)}, indent=2), encoding='utf-8')
    verified, snapshots = [], []
    for run_text in sorted(runs):
        run = pd.Timestamp(run_text)
        original = Path(status['completed_directories'][run_text])
        # Isolated audit snapshots satisfy the existing auditor's run-directory contract.
        folder = out / run.strftime('%Y%m%d%H')
        shutil.copytree(original, folder)
        result = audit_run(folder / 'complete.json', binary, runs)
        assert digest(original / 'complete.json') == result['complete_sha256']
        record = json.loads((original / 'complete.json').read_bytes())
        for name,h in record['output_sha256'].items():
            assert digest(original / name) == h
        result['source_directory'] = str(original)
        verified.append(result)
        row = {'run_at': run, 'valid_at': pd.Timestamp(record['valid_at']),
               'known_at': pd.Timestamp(record['archive_available_at_proxy'])}
        for product, values in record['products'].items():
            prefix = {'geavg': 'mean', 'gespr': 'spread', 'gec00': 'control'}[product]
            for field, item in values['fields'].items():
                row[f'gefs_{prefix}_{field}_f024'] = item['value']
        snapshots.append(row)
        print(f'independently verified {len(verified)}/56', flush=True)
    frame = pd.DataFrame(snapshots).sort_values('known_at').reset_index(drop=True)
    fields = [n for n in frame if n.startswith('gefs_')]
    assert len(fields) == 9
    matched = reuse_forecast_snapshots(frame, pd.Series(origins), fields)
    verify_matches(frame, matched, pd.Series(origins), fields)
    assert len(matched) == 273 and matched.timestamp.is_unique
    supported = matched[fields].notna().all(axis=1)
    frame.to_parquet(out / 'snapshots.parquet', index=False)
    matched.to_parquet(out / 'features.parquet', index=False)
    for p,h in hashes.items():
        assert digest(Path(p)) == h
    report = {'verified_runs': len(verified), 'verified_grib_messages': 9*len(verified),
              'matched_origins': len(matched), 'covered_origins': int(supported.sum()),
              'missing_origins': int((~supported).sum()), 'verified': verified,
              'output_sha256': {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()},
              'limitations': ['Archive Last-Modified is a proxy, not first-publication certification.',
                              'Data verification is not accuracy improvement or deployment approval.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('verified','output_sha256')}, indent=2))


if __name__ == '__main__':
    main()
