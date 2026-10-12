"""Bounded f024 collection for all 2021 origins, without model evaluation."""

import argparse
import json
import re
import subprocess
import time
import uuid
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from haze_forecast import forecast_sources, grib
from scripts import benchmark_observed_business_aft
from scripts.benchmark_observed_business_aft import ROOT, sha

DEFINITIONS = {
    'temperature_c': ('TMP', '2 m above ground'),
    'relative_humidity_pct': ('RH', '2 m above ground'),
    'wind_u_ms': ('UGRD', '10 m above ground'),
    'wind_v_ms': ('VGRD', '10 m above ground'),
}
PRODUCT_FIELDS = {'geavg': DEFINITIONS, 'gespr': DEFINITIONS,
                  'gec00': {'boundary_layer_height_m': ('HPBL', 'surface')}}
NETWORK_ERRORS = (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                  requests.exceptions.ChunkedEncodingError)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def plan_runs(frame):
    times = pd.to_datetime(frame['timestamp'])
    require(times.dt.tz is not None, 'origin timestamps must be timezone-aware')
    times = times.dt.tz_convert('Asia/Shanghai')
    selected = times[times.dt.year == 2021].dt.tz_convert('UTC')
    return sorted((selected - pd.Timedelta('4h')).dt.floor('6h').unique())


def expected_files():
    return {f'{product}.idx' for product in PRODUCT_FIELDS} | {
        f'{product}_{field}.grb2' for product, fields in PRODUCT_FIELDS.items() for field in fields}


def freeze_protocol(out, protocol):
    path = out / 'protocol.json'
    if path.exists():
        require(json.loads(path.read_bytes()) == protocol, 'frozen protocol mismatch')
    else:
        atomic_json(path, protocol)


def verify_complete(folder, run):
    report = json.loads((folder / 'complete.json').read_bytes())
    require(report['run_at'] == run.isoformat(), 'run mismatch')
    require(report['valid_at'] == (run + pd.Timedelta('24h')).isoformat(), 'valid time mismatch')
    require(set(report['products']) == set(PRODUCT_FIELDS), 'product set mismatch')
    for product, fields in PRODUCT_FIELDS.items():
        require(set(report['products'][product]['fields']) == set(fields), 'field set mismatch')
    require(set(report['output_sha256']) == expected_files(), 'output manifest set mismatch')
    require({p.name for p in folder.iterdir()} == expected_files() | {'complete.json'},
            'output directory set mismatch')
    for name, digest in report['output_sha256'].items():
        require(sha(folder / name) == digest, f'output hash mismatch: {name}')
    return report


def with_retries(operation, log_folder, max_retries, sleep=time.sleep):
    require(max_retries >= 0, 'max-retries must be nonnegative')
    log_folder.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as error:
            retryable = isinstance(error, NETWORK_ERRORS)
            atomic_json(log_folder / f'attempt-{uuid.uuid4().hex}.json', {
                'at_utc': datetime.now(UTC).isoformat(), 'attempt': attempt + 1,
                'error': repr(error), 'retryable_network_error': retryable,
                'will_retry': retryable and attempt < max_retries})
            if not retryable or attempt == max_retries:
                raise
            sleep(min(2 ** attempt, 8))


def extract(run, folder, binary):
    folder.mkdir(exist_ok=True)
    require(not (folder / 'complete.json').exists(), 'cannot overwrite completed run')
    products = {}
    for product, fields in PRODUCT_FIELDS.items():
        directory, suffix = ('pgrb2bp5', 'pgrb2b') if product == 'gec00' else ('pgrb2ap5', 'pgrb2a')
        url = (f'https://noaa-gefs-pds.s3.amazonaws.com/gefs.{run:%Y%m%d}/{run:%H}/atmos/'
               f'{directory}/{product}.t{run:%H}z.{suffix}.0p50.f024')
        response = requests.get(url + '.idx', timeout=30)
        response.raise_for_status()
        require(response.status_code == 200, 'index HTTP status must be 200')
        (folder / f'{product}.idx').write_bytes(response.content)
        entries = forecast_sources.parse_gfs_index(response.text)
        records = {}
        for field, (variable, level) in fields.items():
            matches = [entry for entry in entries if entry.variable == variable and entry.level == level]
            require(len(matches) == 1, 'index variable/level is not unique')
            start, end = forecast_sources.range_for_gfs_index_entry(entries, matches[0])
            require(end is not None, 'open-ended byte range is not accepted')
            response = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=45)
            response.raise_for_status()
            require(response.status_code == 206, 'GRIB HTTP status must be 206')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            require(match is not None, 'invalid Content-Range')
            require(tuple(map(int, match.groups()[:2])) == (start, end)
                    and int(match.group(3)) > end, 'Content-Range mismatch')
            content = response.content
            require(len(content) >= 20 and len(content) == end - start + 1, 'GRIB length mismatch')
            require(content[:4] == b'GRIB' and content[-4:] == b'7777'
                    and content[7] == 2 and int.from_bytes(content[8:16], 'big') == len(content),
                    'invalid GRIB message')
            path = folder / f'{product}_{field}.grb2'
            path.write_bytes(content)
            inventory = subprocess.run([str(binary), str(path), '-s', '-vt'], capture_output=True,
                                       text=True, check=True, timeout=30).stdout.strip()
            require(len(inventory.splitlines()) == 1, 'expected one decoded message')
            require(f'd={run:%Y%m%d%H}:' in inventory and ':24 hour fcst:' in inventory,
                    'initialization or lead mismatch')
            require(f':{variable}:{level}:' in inventory
                    and f'vt={run + pd.Timedelta("24h"):%Y%m%d%H}' in inventory,
                    'variable/level/valid time mismatch')
            identity = {'geavg': 'ens mean', 'gespr': 'ens std dev', 'gec00': 'ENS=low-res ctl'}[product]
            require(identity in inventory, 'ensemble product identity mismatch')
            point = grib.query_grib_point(path, 104.0665, 30.5723, binary)
            require(np.isclose(point.longitude, 104.0) and np.isclose(point.latitude, 30.5),
                    'unexpected nearest grid point')
            value = point.value - 273.15 if product == 'geavg' and field == 'temperature_c' else point.value
            require(np.isfinite(value), 'nonfinite value')
            if product == 'gespr' or field == 'boundary_layer_height_m':
                require(value >= 0, 'negative spread or PBL')
            if product == 'geavg' and field == 'relative_humidity_pct':
                require(0 <= value <= 100, 'mean RH outside physical range')
            require(bool(response.headers.get('ETag')), 'ETag missing')
            modified = parsedate_to_datetime(response.headers['Last-Modified'])
            require(modified.tzinfo is not None, 'naive Last-Modified')
            records[field] = {'value': float(value), 'longitude': point.longitude,
                              'latitude': point.latitude, 'sha256': sha(path),
                              'byte_range': [start, end], 'inventory': inventory,
                              'last_modified': modified.astimezone(UTC).isoformat(),
                              'etag': response.headers['ETag']}
        require(len({r['etag'] for r in records.values()}) == 1, 'product ETag changed')
        require(len({r['last_modified'] for r in records.values()}) == 1, 'product Last-Modified changed')
        products[product] = {'url': url, 'fields': records}
    proxy = max(pd.Timestamp(record['last_modified'])
                for product in products.values() for record in product['fields'].values())
    require({p.name for p in folder.iterdir()} == expected_files(), 'unexpected output files')
    report = {'run_at': run.isoformat(), 'valid_at': (run + pd.Timedelta('24h')).isoformat(),
              'archive_available_at_proxy': proxy.isoformat(), 'products': products,
              'output_sha256': {name: sha(folder / name) for name in sorted(expected_files())}}
    atomic_json(folder / 'complete.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-runs', type=int, default=10)
    parser.add_argument('--max-retries', type=int, default=3)
    args = parser.parse_args()
    require(args.max_runs >= 0 and args.max_retries >= 0, 'limits must be nonnegative')
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'interval_business_labels_v1/origins.parquet'
    binary = ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    out = base / 'gefs_2021_f024_collection_v1'
    runs = [pd.Timestamp(run) for run in plan_runs(pd.read_parquet(source))]
    protocol = {
        'source_sha256': sha(source), 'collector_sha256': sha(Path(__file__)),
        'decoder_sha256': sha(binary),
        'helper_sha256': {str(Path(module.__file__).relative_to(ROOT)): sha(Path(module.__file__))
                          for module in (forecast_sources, grib, benchmark_observed_business_aft)},
        'runs': [run.isoformat() for run in runs],
        'selection': '2021 Shanghai origins; UTC minus 4 hours, floor 6 hours; unique sorted',
        'lead_hours': 24,
        'limitations': ['Archive Last-Modified is a proxy, not first-publication proof.',
                        'Completion and hash checks do not establish independent decoded audit or model readiness.']}
    out.mkdir(exist_ok=True)
    freeze_protocol(out, protocol)
    completed = []
    for run in runs:
        folder = out / run.strftime('%Y%m%d%H')
        if (folder / 'complete.json').exists():
            verify_complete(folder, run)
            completed.append(run)
    attempted = []
    try:
        for run in [run for run in runs if run not in completed][:args.max_runs]:
            name = run.strftime('%Y%m%d%H')
            attempted.append(run.isoformat())
            with_retries(lambda r=run, n=name: extract(r, out / n, binary),
                         out / 'failures' / name, args.max_retries)
            verify_complete(out / name, run)
            completed.append(run)
            print(f'completed {run.isoformat()} ({len(completed)}/{len(runs)})', flush=True)
    finally:
        status = {'planned_runs': len(runs), 'completed_runs': len(completed),
                  'remaining_runs': len(runs) - len(completed), 'attempted': attempted,
                  'independent_audit_required': True, 'model_evaluation_allowed': False}
        atomic_json(out / 'status.json', status)
        print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
