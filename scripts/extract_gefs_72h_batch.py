"""Bounded resumable f072 collection for the frozen f024 run population."""

import argparse
import json
import subprocess
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from haze_forecast.forecast_sources import parse_gfs_index, range_for_gfs_index_entry
from haze_forecast.grib import query_grib_point
from scripts.benchmark_observed_business_aft import ROOT, sha


def extract(run, folder, binary):
    folder.mkdir(exist_ok=True)
    products = {}
    definitions = {'temperature_c': ('TMP', '2 m above ground'), 'relative_humidity_pct': ('RH', '2 m above ground'),
                   'wind_u_ms': ('UGRD', '10 m above ground'), 'wind_v_ms': ('VGRD', '10 m above ground')}
    for product, directory, suffix in [('geavg', 'pgrb2ap5', 'pgrb2a'), ('gespr', 'pgrb2ap5', 'pgrb2a'), ('gec00', 'pgrb2bp5', 'pgrb2b')]:
        url = f'https://noaa-gefs-pds.s3.amazonaws.com/gefs.{run:%Y%m%d}/{run:%H}/atmos/{directory}/{product}.t{run:%H}z.{suffix}.0p50.f072'
        response = requests.get(url + '.idx', timeout=30)
        response.raise_for_status()
        (folder / f'{product}.idx').write_bytes(response.content)
        entries = parse_gfs_index(response.text)
        fields = definitions if product != 'gec00' else {'boundary_layer_height_m': ('HPBL', 'surface')}
        records = {}
        for field, (variable, level) in fields.items():
            matches = [e for e in entries if e.variable == variable and e.level == level]
            assert len(matches) == 1
            start, end = range_for_gfs_index_entry(entries, matches[0])
            response = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=45)
            assert response.status_code == 206 and response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
            content = response.content
            assert len(content) == end-start+1 and content[:4] == b'GRIB' and content[-4:] == b'7777'
            assert content[7] == 2 and int.from_bytes(content[8:16], 'big') == len(content)
            path = folder / f'{product}_{field}.grb2'
            path.write_bytes(content)
            inventory = subprocess.run([str(binary), str(path), '-s', '-vt'], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
            assert f'd={run:%Y%m%d%H}' in inventory and ':72 hour fcst:' in inventory
            assert f':{variable}:{level}:' in inventory and f'vt={run + pd.Timedelta(hours=72):%Y%m%d%H}' in inventory
            assert ('ens mean' if product == 'geavg' else 'ens std dev' if product == 'gespr' else 'ENS=low-res ctl') in inventory
            point = query_grib_point(path, 104.0665, 30.5723, binary)
            value = point.value - 273.15 if field == 'temperature_c' and product == 'geavg' else point.value
            assert np.isfinite(value)
            if product == 'gespr' or field == 'boundary_layer_height_m':
                assert value >= 0
            if product == 'geavg' and field == 'relative_humidity_pct':
                assert 0 <= value <= 100
            records[field] = {'value': value, 'longitude': point.longitude, 'latitude': point.latitude,
                'sha256': sha(path), 'byte_range': [start, end], 'inventory': inventory,
                'last_modified': pd.Timestamp(parsedate_to_datetime(response.headers['Last-Modified'])).isoformat(),
                'etag': response.headers.get('ETag')}
        assert len({r['etag'] for r in records.values()}) == 1
        assert len({r['last_modified'] for r in records.values()}) == 1
        products[product] = {'url': url, 'fields': records}
    assert len({(f['longitude'], f['latitude']) for p in products.values() for f in p['fields'].values()}) == 1
    proxy = max(pd.Timestamp(f['last_modified']) for p in products.values() for f in p['fields'].values())
    report = {'run_at': run.isoformat(), 'valid_at': (run + pd.Timedelta(hours=72)).isoformat(),
              'archive_available_at_proxy': proxy.isoformat(), 'products': products,
              'output_sha256': {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != 'complete.json'}}
    (folder / 'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-runs', type=int, default=3)
    args = parser.parse_args()
    if args.max_runs < 0:
        raise ValueError('max-runs must be nonnegative')
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'gefs_72h_collection_v1'
    source = base / 'reusable_gefs_snapshots_v1/snapshots.parquet'
    binary = ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    runs = sorted(pd.read_parquet(source).run_at.unique())
    protocol = {'source_sha256': sha(source), 'decoder_sha256': sha(binary),
                'collector_sha256': sha(Path(__file__)), 'runs': [pd.Timestamp(t).isoformat() for t in runs],
                'limitations': ['fixed f024 run population only', 'archive timestamp proxy, not first-publication proof']}
    out.mkdir(exist_ok=True)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_bytes()) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    completed = []
    for run in runs:
        run = pd.Timestamp(run)
        folder = out / run.strftime('%Y%m%d%H')
        marker = folder / 'complete.json'
        if marker.exists():
            record = json.loads(marker.read_bytes())
            assert record['run_at'] == run.isoformat()
            assert record['output_sha256']
            for name, digest in record['output_sha256'].items():
                assert sha(folder / name) == digest
            completed.append(run)
    pending = [pd.Timestamp(r) for r in runs if r not in completed]
    attempted = []
    for run in pending[:args.max_runs]:
        folder = out / run.strftime('%Y%m%d%H')
        try:
            extract(run, folder, binary)
        except Exception as error:
            attempted.append({'run': run.isoformat(), 'status': 'failed', 'error': repr(error)})
            break
        completed.append(run)
        attempted.append({'run': run.isoformat(), 'status': 'complete'})
        print(f'completed {run.isoformat()} ({len(completed)}/{len(runs)})', flush=True)
    status = {'planned_runs': len(runs), 'completed_runs': len(completed), 'remaining_runs': len(runs)-len(completed),
              'attempted': attempted, 'formal_model_evaluation_allowed': len(completed) == len(runs)}
    (out / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
