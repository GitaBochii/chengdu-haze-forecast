"""Resumable pinned climatology region import, without global object downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import urlsplit

import h5py
import numpy as np
import requests
import yaml

from climatology_range_probe import HERE, REVISION, BASE, read_range_response, slab_range, file_sha
from remote_hdf5 import RangeFile

LAT = np.arange(23, 38, .5)
LON = np.arange(93.75, 113.75, .625)


class RangeFetcher:
    """Keep signed CDN destinations in memory only; refresh expired URLs at origin."""
    def __init__(self, session, entry):
        self.session, self.entry = session, entry
        self.origin = BASE + entry['path']
        self.destination = None
        self.requests = 0
        self.body_bytes = 0

    def __call__(self, start, end):
        for attempt in range(3):
            try:
                self.requests += 1
                with self.session.get(self.destination or self.origin, headers={
                        'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                        stream=True, timeout=(20, 90)) as response:
                    if response.status_code in (401, 403):
                        self.destination = None
                        raise ValueError('range authorization expired')
                    parsed = urlsplit(response.url)
                    host = parsed.hostname or ''
                    if (parsed.scheme != 'https' or parsed.username or parsed.password
                            or not (host == 'huggingface.co' or host == 'hf.co' or host.endswith('.hf.co'))):
                        raise ValueError('unexpected range destination')
                    body = read_range_response(response, start, end, self.entry['size'])
                    self.destination = response.url
                self.body_bytes += len(body)
                return body
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise RuntimeError('bounded range request failed after three attempts') from None
                time.sleep(2 ** attempt)


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     delete=False, suffix='.tmp') as stream:
        name = stream.name
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)


def validate_arrays(arrays, variables, levels):
    keys = {'lat', 'lon', *variables} | ({'lev'} if levels is not None else set())
    if set(arrays) != keys:
        raise ValueError('unexpected regional variables')
    if not np.array_equal(arrays['lat'], LAT) or not np.array_equal(arrays['lon'], LON):
        raise ValueError('regional coordinates mismatch')
    if levels is not None:
        if not np.array_equal(np.sort(arrays['lev']), np.sort(levels)):
            raise ValueError('level values mismatch')
    shape = (30, 32) if levels is None else (len(levels), 30, 32)
    for name in variables:
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError('invalid region shape or nonfinite values')


def cache_valid(path, entry, config_sha, variables, levels):
    try:
        record = json.loads(path.with_suffix('.json').read_text('utf-8'))
        if (record['revision'] != REVISION or record['source'] != entry['path']
                or record['source_sha256_metadata'] != entry['lfs']['oid']
                or record['source_size'] != entry['size'] or record['config_sha256'] != config_sha
                or record['artifact_sha256'] != file_sha(path)):
            return False
        with np.load(path, allow_pickle=False) as arrays:
            validate_arrays(arrays, variables, levels)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def save_region(path, arrays, entry, config_sha, method, body_bytes, requests_count):
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, suffix='.tmp', delete=False) as stream:
        temporary = stream.name
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    record = {'status': 'imported', 'revision': REVISION, 'source': entry['path'],
              'source_sha256_metadata': entry['lfs']['oid'], 'source_size': entry['size'],
              'full_source_sha256_verified': method == 'local_verified_full_object',
              'config_sha256': config_sha, 'artifact': path.name,
              'artifact_sha256': file_sha(path), 'method': method,
              'received_body_bytes': body_bytes, 'http_requests': requests_count,
              'variables': list(arrays), 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(path.with_suffix('.json'), record)
    return record


def extract(handle, variables, levels, fetch=None, size=None):
    arrays = {'lat': handle['lat'][226:256], 'lon': handle['lon'][438:470]}
    if not np.array_equal(arrays['lat'], LAT) or not np.array_equal(arrays['lon'], LON):
        raise ValueError('source coordinates mismatch')
    if levels is not None:
        arrays['lev'] = handle['lev'][:]
        if not np.array_equal(np.sort(arrays['lev']), np.sort(levels)):
            raise ValueError('source level values mismatch')
    for name in variables:
        dataset = handle[name]
        expected = (361, 576) if levels is None else (14, 361, 576)
        if dataset.shape != expected or dataset.dtype.kind != 'f' or dataset.dtype.itemsize != 4:
            raise ValueError('source grid or dtype mismatch')
        if fetch is None:
            arrays[name] = (dataset[226:256, 438:470] if levels is None
                            else dataset[:, 226:256, 438:470])
        else:
            if dataset.chunks is not None or dataset.compression:
                raise ValueError('unsupported source storage')
            slices = []
            for level in ([None] if levels is None else range(14)):
                start, end = slab_range(dataset.id.get_offset(), dataset.shape, 4, level)
                if end >= size:
                    raise ValueError('source range outside object')
                body = fetch(start, end)
                slices.append(np.frombuffer(body, dtype=dataset.dtype).reshape(30, 576)[:, 438:470].copy())
            arrays[name] = slices[0] if levels is None else np.stack(slices)
    validate_arrays(arrays, variables, levels)
    return arrays


def import_one(entry, output, config, config_sha):
    relative = Path(entry['path'])
    if relative.parent.as_posix() != 'climatology' or relative.suffix != '.nc':
        raise ValueError('invalid source path')
    vertical = relative.name.startswith('climate_vertical_')
    variables = config['vertical_vars' if vertical else 'surface_vars']
    levels = config['levels'] if vertical else None
    artifact = output / (relative.stem + '.npz')
    if cache_valid(artifact, entry, config_sha, variables, levels):
        return {'source': entry['path'], 'status': 'verified_cached'}
    local = HERE / 'real_sample' / relative
    if local.exists() and local.stat().st_size == entry['size'] and file_sha(local) == entry['lfs']['oid']:
        with h5py.File(local, 'r') as handle:
            arrays = extract(handle, variables, levels)
        return save_region(artifact, arrays, entry, config_sha, 'local_verified_full_object', 0, 0)
    with requests.Session() as session:
        session.trust_env = False
        session.proxies = requests.utils.get_environ_proxies(BASE)
        fetch = RangeFetcher(session, entry)
        with RangeFile(entry['size'], fetch) as remote:
            with h5py.File(remote, 'r') as handle:
                arrays = extract(handle, variables, levels, fetch, entry['size'])
    return save_region(artifact, arrays, entry, config_sha, 'remote_range',
                       fetch.body_bytes, fetch.requests)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16 or (args.limit is not None and args.limit < 1):
        parser.error('positive limit and 1..16 workers required')
    inventory_path = HERE / 'full_history/climatology_inventory.json'
    inventory_sha = file_sha(inventory_path)
    inventory = json.loads(inventory_path.read_text('utf-8'))
    if inventory['revision'] != REVISION or inventory['missing'] or not inventory['metadata_pagination_complete']:
        raise ValueError('complete pinned inventory required')
    config_path = HERE / 'vendor/config.yaml'
    config_sha = file_sha(config_path)
    config = yaml.safe_load(config_path.read_text('utf-8'))['data']
    entries = inventory['files'][:args.limit]
    output = HERE / 'full_history/climatology_regions'
    output.mkdir(parents=True, exist_ok=True)
    result = {'status': 'running', 'inventory_sha256': inventory_sha,
              'required_total': len(inventory['files']), 'selected_total': len(entries),
              'revision': REVISION, 'files': [], 'errors': [],
              'process_id': os.getpid(), 'started_at_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(output / 'manifest.json', result)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(import_one, e, output, config, config_sha): e for e in entries}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                row = future.result()
                result['files'].append(row)
                message = {'source': entry['path'], 'status': row['status']}
            except Exception as error:
                message = {'source': entry['path'], 'status': 'failed', 'error_type': type(error).__name__}
                result['errors'].append(message)
            result['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
            atomic_json(output / 'manifest.json', result)
            print(json.dumps({**message, 'done': len(result['files']), 'failed': len(result['errors']),
                              'selected': len(entries)}), flush=True)
    unchanged = file_sha(inventory_path) == inventory_sha and file_sha(config_path) == config_sha
    result['status'] = ('failed' if result['errors'] or not unchanged else
                        'complete' if len(entries) == len(inventory['files']) else 'partial_limit_complete')
    result['source_unchanged'] = unchanged
    atomic_json(output / 'manifest.json', result)
    print(json.dumps({'status': result['status'], 'imported': len(result['files']),
                      'failed': len(result['errors'])}), flush=True)
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
