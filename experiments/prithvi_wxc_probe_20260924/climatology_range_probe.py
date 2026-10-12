"""Bounded range-read proof using verified local HDF5 layouts, not a downloader."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import requests


HERE = Path(__file__).resolve().parent
REVISION = 'e2f62b3e19890f1a5e225bb123acb4b33df85ef8'
BASE = f'https://huggingface.co/ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M/resolve/{REVISION}/'


def slab_range(offset, shape, itemsize, level):
    if offset is None or offset < 0 or itemsize != 4:
        raise ValueError('unsupported storage offset or dtype size')
    if shape == (361, 576) and level is None:
        plane = 0
    elif shape == (14, 361, 576) and isinstance(level, int) and 0 <= level < 14:
        plane = level * 361
    else:
        raise ValueError('unsupported grid or level')
    return (offset + (plane + 226) * 576 * itemsize,
            offset + (plane + 256) * 576 * itemsize - 1)


def read_range_response(response, start, end, size):
    if response.status_code != 206:
        raise ValueError(f'range request not honored: HTTP {response.status_code}')
    if response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
        raise ValueError('unexpected Content-Range')
    if response.headers.get('Content-Encoding', 'identity') != 'identity':
        raise ValueError('encoded range response')
    data = bytearray()
    for chunk in response.iter_content(chunk_size=16384):
        if len(data) + len(chunk) > end - start + 1:
            raise ValueError('range response exceeds requested bytes')
        data.extend(chunk)
    if len(data) != end - start + 1:
        raise ValueError('truncated range response')
    return bytes(data)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    inventory_path = HERE / 'pilot_file_inventory.json'
    inventory = json.loads(inventory_path.read_text('utf-8'))
    if inventory['revision'] != REVISION:
        raise ValueError('inventory revision mismatch')
    entries = {entry['path']: entry for entry in inventory['files']}
    probes = [
        ('climatology/climate_surface_doy001_hour12.nc', [('T2M', None),
            ('QV2M', None), ('U10M', None), ('V10M', None)]),
        ('climatology/climate_vertical_doy001_hour12.nc', [('U', 0), ('T', 13)]),
    ]
    report = {'revision': REVISION, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'inventory_sha256': file_sha(inventory_path), 'status': 'started',
              'verified_files': [], 'ranges': [], 'full_history_ready': False,
              'limitation': 'Offsets derived separately from verified local full files; '
                            'unseen-file metadata discovery is not implemented.'}
    output = HERE / 'full_history/climatology_range_probes' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    try:
        with requests.Session() as session:
            # Reuse transport proxy settings, but never discover .netrc credentials.
            session.trust_env = False
            session.proxies = requests.utils.get_environ_proxies(BASE)
            for relative, variables in probes:
                entry = entries[relative]
                path = HERE / 'real_sample' / relative
                digest = file_sha(path)
                if path.stat().st_size != entry['size'] or digest != entry['lfs']['oid']:
                    raise ValueError('local reference does not match pinned inventory')
                report['verified_files'].append({'path': relative, 'sha256': digest,
                                                 'size': entry['size']})
                with h5py.File(path, 'r') as handle:
                    if not (np.array_equal(handle['lat'][226:256], np.arange(23, 38, .5))
                            and np.array_equal(handle['lon'][438:470], np.arange(93.75, 113.75, .625))):
                        raise ValueError('reference regional coordinates mismatch')
                    for name, level in variables:
                        dataset = handle[name]
                        if dataset.chunks is not None or dataset.compression or dataset.dtype.kind != 'f':
                            raise ValueError('unsupported HDF5 storage layout')
                        start, end = slab_range(dataset.id.get_offset(), dataset.shape,
                                                dataset.dtype.itemsize, level)
                        if end >= entry['size']:
                            raise ValueError('range outside verified object')
                        with session.get(BASE + relative, headers={
                            'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                            stream=True, timeout=(20, 60)) as response:
                            body = read_range_response(response, start, end, entry['size'])
                        region = np.frombuffer(body, dtype=dataset.dtype).reshape(30, 576)[:, 438:470].copy()
                        expected = (dataset[226:256, 438:470] if level is None
                                    else dataset[level, 226:256, 438:470])
                        if not np.isfinite(region).all() or not np.array_equal(region, expected):
                            raise ValueError('remote regional values differ from local reference')
                        suffix = 'surface' if level is None else f'level{level}'
                        artifact = output / f'{name}_{suffix}.npy'
                        np.save(artifact, region, allow_pickle=False)
                        report['ranges'].append({'source': relative, 'variable': name,
                            'level_index': level, 'bytes_start': start, 'bytes_end': end,
                            'body_bytes': len(body), 'body_sha256': hashlib.sha256(body).hexdigest(),
                            'cells_equal': region.size, 'artifact': artifact.name,
                            'artifact_sha256': file_sha(artifact)})
                        print(json.dumps(report['ranges'][-1]), flush=True)
        report['status'] = 'bounded_range_equality_verified'
    except Exception as error:
        report['status'] = 'failed'
        # Network exception strings can contain temporary signed URLs.
        report['error_type'] = type(error).__name__
        raise RuntimeError(f'bounded range probe failed: {type(error).__name__}; see report') from None
    finally:
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        report['verified_range_body_bytes'] = sum(row['body_bytes'] for row in report['ranges'])
        report['verified_cells'] = sum(row['cells_equal'] for row in report['ranges'])
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'status': report['status'], 'report': str(output / 'report.json'),
                          'verified_cells': report['verified_cells'],
                          'verified_range_body_bytes': report['verified_range_body_bytes']}), flush=True)


if __name__ == '__main__':
    main()
