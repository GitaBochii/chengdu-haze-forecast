"""Export physical fields from the exact audited encoder batches, not illustrations."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
PINS = {
    'pilot_candidate_audit.json': '036c3760cbc27eed1bdb15d887c836b31b19672a3c71481a4e19a0596750da81',
    'pilot_embedding_report.json': '2a4c7c9bbe3486c203448f21dd3ea078777ebcf758c3a1720ee82d0a22027943',
    'pilot_batch_report.json': 'cefb99de5b207ad8b78aa638f1c12e2ef724868fc13bb6d1b1393d8e6851fa10',
    'vendor/config.yaml': 'db25b5631dc290f68838ea65130c69e2f0f2b79a1874fd42072c1f1ed4e99554',
}
UNITS = {'T2M': 'K', 'QV2M': 'kg kg-1', 'U10M': 'm s-1', 'V10M': 'm s-1'}
FIELDS = [
    {'id': 't2m', 'label': '2 米气温', 'unit': '°C', 'note': 'T2M；原始 K 减去 273.15。'},
    {'id': 'q2m', 'label': '2 米比湿', 'unit': 'g/kg', 'note': 'QV2M；原始 kg/kg 乘以 1000，不是相对湿度。'},
    {'id': 'wind10m', 'label': '10 米风速', 'unit': 'm/s', 'note': '由实际输入的 U10M、V10M 计算 sqrt(U² + V²)。'},
]


def checked_file(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'file integrity mismatch: {path.name}')
    return path


def validate_times(timestamp, times):
    origin = pd.Timestamp(timestamp)
    if origin.tzinfo is None or len(times) != 2:
        raise ValueError('timezone-aware origin and two input times required')
    actual = [pd.Timestamp(time) for time in times]
    if actual != [origin - pd.Timedelta(hours=6), origin]:
        raise ValueError('weather must be exactly origin minus six hours and origin')


def extract_fields(x, channels):
    if x.shape != (1, 2, len(channels), 30, 32) or not np.isfinite(x).all():
        raise ValueError('invalid physical input tensor')
    parts = {name: x[0, :, channels.index(name)].astype(np.float64) for name in UNITS}
    values = {'t2m': parts['T2M'] - 273.15, 'q2m': parts['QV2M'] * 1000,
              'wind10m': np.hypot(parts['U10M'], parts['V10M'])}
    return {name: array.reshape(2, -1).tolist() for name, array in values.items()}


def export(root=HERE):
    for name, digest in PINS.items():
        checked_file(root / name, digest)
    candidates = json.loads((root / 'pilot_candidate_audit.json').read_text('utf-8'))['rows']
    batch_report = json.loads((root / 'pilot_batch_report.json').read_text('utf-8'))
    embedding_report = json.loads((root / 'pilot_embedding_report.json').read_text('utf-8'))
    batches = {row['timestamp']: row for row in batch_report['rows']}
    embeddings = {row['timestamp']: row for row in embedding_report['rows']}
    channels = yaml.safe_load((root / 'vendor/config.yaml').read_text())['data']['surface_vars']
    samples = []
    latitudes = longitudes = None
    for candidate in candidates:
        stamp = candidate['timestamp']
        row, encoded = batches[stamp], embeddings[stamp]
        if row['status'] != 'ready' or encoded['status'] != 'ready' or row['batch_sha256'] != encoded['batch_sha256']:
            raise ValueError('batch was not the encoded batch')
        batch = torch.load(checked_file(root / row['batch_path'], row['batch_sha256']), weights_only=True)
        embedding = torch.load(checked_file(root / encoded['embedding_path'], encoded['embedding_sha256']), weights_only=True)
        for key in ['timestamp', 'partition', 'candidate_manifest_sha256', 'revision']:
            if batch[key] != row[key] or embedding[key] != row[key]:
                raise ValueError(f'metadata mismatch: {key}')
        if batch['partition'] != candidate['partition'] or batch['candidate_manifest_sha256'] != PINS['pilot_candidate_audit.json']:
            raise ValueError('candidate mismatch')
        if embedding['batch_sha256'] != row['batch_sha256'] or batch['future_actual_weather_read'] is not False:
            raise ValueError('encoder lineage mismatch')
        times = batch['weather_times_utc']
        validate_times(stamp, times)
        if row['weather_times_utc'] != times or batch['source_sha256'] != row['source_sha256']:
            raise ValueError('batch source metadata mismatch')
        x = batch['batch']['x'].numpy()
        source_name = next(name for name in row['source_sha256'] if name.startswith('merra-2/MERRA2_sfc_'))
        source_path = checked_file(root / 'real_sample' / source_name, row['source_sha256'][source_name])
        # Source coordinates and all displayed cells independently prove ordering and physical units.
        with h5py.File(source_path, 'r') as source:
            lat, lon = source['lat'][:], source['lon'][:]
            iy = np.flatnonzero((lat >= 23) & (lat <= 37.5))
            ix = np.flatnonzero((lon >= 93.75) & (lon <= 113.125))
            if len(iy) != 30 or len(ix) != 32 or not np.all(np.diff(iy) == 1) or not np.all(np.diff(ix) == 1):
                raise ValueError('regional grid mismatch')
            current_lat, current_lon = lat[iy].tolist(), lon[ix].tolist()
            if latitudes is None:
                latitudes, longitudes = current_lat, current_lon
            if current_lat != latitudes or current_lon != longitudes:
                raise ValueError('grid changed between origins')
            units = source['time'].attrs['units'].decode()
            if not units.startswith('minutes since '):
                raise ValueError('unrecognized source time units')
            base = pd.Timestamp(units.removeprefix('minutes since '), tz='UTC')
            source_times = [base + pd.Timedelta(minutes=float(value)) for value in source['time'][:]]
            for name, expected_unit in UNITS.items():
                variable = source[name]
                if variable.attrs['units'].decode() != expected_unit:
                    raise ValueError(f'unit mismatch: {name}')
                for t, time in enumerate(times):
                    index = source_times.index(pd.Timestamp(time))
                    raw = variable[index, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]
                    if not np.array_equal(raw, x[0, t, channels.index(name)]):
                        raise ValueError(f'physical source and batch differ: {name}')
        fields = extract_fields(x[:, :, :len(channels)], channels)
        samples.append({'timestamp': stamp, 'times': times, 'fields': fields,
                        'provenance': {'batch_sha256': row['batch_sha256'],
                                       'embedding_sha256': encoded['embedding_sha256'],
                                       'source_sha256': {source_name: row['source_sha256'][source_name]}}})
    if len(samples) != 25 or len({sample['timestamp'] for sample in samples}) != 25:
        raise ValueError('all 25 unique audited origins required')
    return {'schema_version': 1, 'source': 'MERRA-2', 'fields': FIELDS,
            'latitudes': latitudes, 'longitudes': longitudes, 'samples': samples,
            'provenance': {'manifest_sha256': PINS, 'layout': 'latitude-major, longitude-minor',
                           'physical_source_cells_verified': 25 * 2 * 4 * 30 * 32,
                           'future_actual_weather_read': False}}


if __name__ == '__main__':
    result = export()
    output = HERE.parents[1] / 'project-showcase/weather-inputs.json'
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False), 'utf-8')
    print(json.dumps({'samples': len(result['samples']), 'bytes': output.stat().st_size,
                      'physical_source_cells_verified': result['provenance']['physical_source_cells_verified']}))
