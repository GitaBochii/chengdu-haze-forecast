"""Prepare all frozen pilot rows, retaining missing/failed dates in the report."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import h5py
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
from PrithviWxC.dataloaders.merra2 import Merra2Dataset, SampleSpec
from experiments.prithvi_wxc_probe_20260924.inference_sample import load_inference_batch
from experiments.prithvi_wxc_probe_20260924.prepare_pilot_inventory import REVISION, selected_paths


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_sources(root, entries):
    missing, failed, verified = [], [], {}
    for entry in entries:
        path = root / entry['path']
        if not path.is_file():
            missing.append(entry['path'])
        elif path.stat().st_size != entry['size']:
            failed.append({'path': entry['path'], 'reason': 'size_mismatch'})
        else:
            digest = sha(path)
            if digest != entry['lfs']['oid']:
                failed.append({'path': entry['path'], 'reason': 'sha256_mismatch'})
            else:
                verified[entry['path']] = digest
    return {'status': 'failed' if failed else ('missing' if missing else 'ready'),
            'missing': missing, 'failed': failed, 'source_sha256': verified}


def validate_batch(batch):
    shapes = {'x': (1, 2, 160, 30, 32), 'static': (1, 10, 30, 32),
              'climate': (1, 160, 30, 32), 'input_time': (1,), 'lead_time': (1,)}
    if set(batch) != set(shapes):
        raise ValueError('exactly five inference fields required; y forbidden')
    for key, shape in shapes.items():
        value = batch[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f'invalid batch field: {key}')
    if batch['input_time'].item() != 6 or batch['lead_time'].item() != 6:
        raise ValueError('fixed six-hour input and lead times required')


def build_batch(origin, paths, target_lat, target_lon):
    data = HERE / 'real_sample'
    for path in paths:
        with h5py.File(data / path, 'r') as source:
            for key, expected in [('lat', target_lat), ('lon', target_lon)]:
                coordinate = source[key][:]
                positions = np.flatnonzero(np.isin(coordinate, expected))
                if (len(positions) != len(expected) or not np.array_equal(coordinate[positions], expected)
                        or not np.all(np.diff(positions) == 1)):
                    raise ValueError(f'{path}: regional {key} coordinates absent or inconsistent')
    cfg = yaml.safe_load((HERE / 'vendor/config.yaml').read_text())['data']
    stamp = origin.tz_convert('UTC').tz_localize(None)
    dataset = Merra2Dataset(
        time_range=(str(stamp.normalize()), str(stamp.normalize() + pd.Timedelta(hours=23))),
        lead_times=[6], input_times=[-6], data_path_surface=data / 'merra-2',
        data_path_vertical=data / 'merra-2', climatology_path_surface=data / 'climatology',
        climatology_path_vertical=data / 'climatology', surface_vars=cfg['surface_vars'],
        static_surface_vars=cfg['static_surface_vars'], vertical_vars=cfg['vertical_vars'],
        levels=cfg['levels'], positional_encoding='fourier')
    iy = int(np.flatnonzero(dataset.lats == target_lat[0])[0])
    ix = int(np.flatnonzero(dataset.lons == target_lon[0])[0])
    if not np.array_equal(dataset.lats[iy:iy+30], target_lat) or not np.array_equal(dataset.lons[ix:ix+32], target_lon):
        raise ValueError('dataset coordinate ordering differs from regional reference')
    batch = load_inference_batch(dataset, SampleSpec.get(stamp, 6, 6), slice(iy, iy+30), slice(ix, ix+32))
    validate_batch(batch)
    return batch


def main():
    manifest = HERE / 'pilot_candidate_audit.json'
    manifest_sha = sha(manifest)
    frozen = json.loads(manifest.read_text('utf-8'))
    inventory = json.loads((HERE / 'pilot_file_inventory.json').read_text('utf-8'))
    if (inventory['revision'] != REVISION or inventory['candidate_manifest_sha256'] != manifest_sha
            or len(frozen['rows']) != 25):
        raise ValueError('frozen manifest/inventory mismatch')
    entries = {item['path']: item for item in inventory['files']}
    with np.load(HERE / 'regional_raw_sample.npz', allow_pickle=False) as reference:
        lat, lon = reference['lat'].copy(), reference['lon'].copy()
    if (len(lat) != 30 or len(lon) != 32 or not np.allclose([lat.min(),lat.max(),lon.min(),lon.max()],
                                                           [23,37.5,93.75,113.125])):
        raise ValueError('regional reference coordinates changed')
    output = HERE / 'pilot_batches'
    output.mkdir(exist_ok=True)
    rows = []
    for candidate in frozen['rows']:
        row = {key: candidate[key] for key in ('timestamp', 'date', 'partition')}
        row.update(batch_path=None, batch_sha256=None, source_sha256={})
        try:
            paths = selected_paths(candidate['timestamp'])
            absent = [path for path in paths if path not in entries]
            if absent:
                row.update(status='missing', missing=absent, reason='metadata_missing')
            else:
                check = check_sources(HERE / 'real_sample', [entries[path] for path in paths])
                row.update(check)
                if check['status'] == 'ready':
                    origin = pd.Timestamp(candidate['timestamp'])
                    batch = build_batch(origin, paths, lat, lon)
                    batch_path = output / f"{candidate['date']}.pt"
                    metadata = {'origin_utc': origin.tz_convert('UTC').isoformat(),
                                'timestamp': candidate['timestamp'], 'partition': candidate['partition'],
                                'revision': REVISION, 'candidate_manifest_sha256': manifest_sha,
                                'source_sha256': check['source_sha256'],
                                'weather_times_utc': [(origin.tz_convert('UTC')-pd.Timedelta(hours=6)).isoformat(),
                                                      origin.tz_convert('UTC').isoformat()],
                                'future_actual_weather_read': False}
                    temp = batch_path.with_suffix('.pt.partial')
                    torch.save({'batch': batch, **metadata}, temp)
                    restored = torch.load(temp, weights_only=True, map_location='cpu')
                    validate_batch(restored['batch'])
                    for key in batch:
                        torch.testing.assert_close(batch[key], restored['batch'][key], rtol=0, atol=0)
                    temp.replace(batch_path)
                    row.update(metadata, status='ready', batch_path=str(batch_path.relative_to(HERE)),
                               batch_sha256=sha(batch_path), shapes={k:list(v.shape) for k,v in batch.items()})
        except Exception as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
        rows.append(row)
    if sha(manifest) != manifest_sha:
        raise ValueError('candidate manifest changed during run')
    report = {'revision': REVISION, 'candidate_manifest_sha256': manifest_sha,
              'rows': rows, 'total_rows': len(rows),
              'counts': {status: sum(row['status'] == status for row in rows)
                         for status in ('ready', 'missing', 'failed')},
              'future_actual_weather_read': False, 'partial_source_files_used': False,
              'scope': 'offline input preparation; all frozen dates retained; no model loading or training'}
    (HERE / 'pilot_batch_report.json').write_text(json.dumps(report, indent=2), 'utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
