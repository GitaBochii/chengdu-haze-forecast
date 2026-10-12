"""Inventory only the frozen pilot's remote data objects; download no tensors."""
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
REVISION = 'e2f62b3e19890f1a5e225bb123acb4b33df85ef8'
API = f'https://huggingface.co/api/models/Prithvi-WxC/prithvi.wxc.2300m.v1/tree/{REVISION}/climatology'
CANONICAL_API = f'https://huggingface.co/api/models/ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M/tree/{REVISION}/climatology'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_paths(timestamp):
    origin = pd.Timestamp(timestamp)
    if origin.tzinfo is None:
        raise ValueError('timezone-aware origin required')
    origin = origin.tz_convert('UTC')
    if (origin.hour, origin.minute, origin.second, origin.microsecond, origin.nanosecond) != (6, 0, 0, 0, 0):
        raise ValueError('pilot origin must be UTC 06:00')
    target = origin + pd.Timedelta(hours=6)
    doy = min(target.dayofyear, 365)
    date = origin.strftime('%Y%m%d')
    return [f'merra-2/MERRA2_sfc_{date}.nc', f'merra-2/MERRA_pres_{date}.nc',
            f'climatology/climate_surface_doy{doy:03}_hour12.nc',
            f'climatology/climate_vertical_doy{doy:03}_hour12.nc']


def validate_entry(entry):
    lfs = entry.get('lfs', {})
    if (not re.fullmatch('[0-9a-f]{64}', str(lfs.get('oid', '')))
            or type(entry.get('size')) is not int or entry['size'] <= 0
            or lfs.get('size') != entry['size']):
        raise ValueError('missing SHA256 or inconsistent object size')


def main():
    frozen_path = HERE / 'pilot_candidate_audit.json'
    manifest_sha = sha(frozen_path)
    frozen = json.loads(frozen_path.read_text('utf-8'))
    if len(frozen['rows']) != 25:
        raise ValueError('unexpected frozen candidate count')
    mapping = [{'origin': row['timestamp'], 'partition': row['partition'],
                'paths': selected_paths(row['timestamp'])} for row in frozen['rows']]
    requested = sorted({path for item in mapping for path in item['paths']})
    weather = json.loads((HERE / 'hf_merra_inventory.json').read_text('utf-8'))
    if weather['revision'] != REVISION:
        raise ValueError('weather revision mismatch')
    entries = {entry['path']: entry for entry in weather['entries']}
    pages, url = [], API
    with requests.Session() as session:
        for _ in range(20):
            if not url:
                break
            if not any(url == base or url.startswith(base + '?') for base in (API, CANONICAL_API)):
                raise ValueError('unexpected pagination destination')
            response = session.get(url, timeout=(15, 40))
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list):
                raise ValueError('unexpected HF metadata payload')
            pages.append({'url': url, 'entries': len(page)})
            for entry in page:
                path = entry['path']
                if path in entries and entries[path] != entry:
                    raise ValueError('conflicting metadata entries')
                entries[path] = entry
            url = response.links.get('next', {}).get('url')
        if url:
            raise ValueError('metadata pagination incomplete')
    missing = [path for path in requested if path not in entries]
    selected = []
    for path in requested:
        if path not in entries:
            continue
        entry = entries[path]
        validate_entry(entry)
        cached = HERE / 'real_sample' / path
        cache_status = 'missing'
        if cached.exists():
            if cached.stat().st_size != entry['size']:
                cache_status = 'size_mismatch'
            elif sha(cached) != entry['lfs']['oid']:
                cache_status = 'sha256_mismatch'
            else:
                cache_status = 'verified_complete'
        selected.append({'path': path, 'size': entry['size'], 'lfs': entry['lfs'],
                         'revision': REVISION, 'cache_status': cache_status,
                         'cache_path': str(cached.relative_to(HERE)), 'upstream_entry': entry})
    if sha(frozen_path) != manifest_sha:
        raise ValueError('frozen candidate manifest changed during audit')
    source = HERE / 'vendor/PrithviWxC/dataloaders/merra2.py'
    if 'min(self.target.dayofyear, 365)' not in source.read_text('utf-8'):
        raise ValueError('pinned official calendar convention changed')
    result = {'status': 'complete_metadata_inventory' if not missing else 'missing_selected_objects',
              'revision': REVISION, 'candidate_manifest_sha256': manifest_sha,
              'calendar_rule': 'SampleSpec.climatology_index: min(target.dayofyear,365), target.hour',
              'official_calendar_source_sha256': sha(source), 'metadata_pages': pages,
              'selected_days': len(mapping), 'requested_unique_objects': len(requested),
              'weather_objects': sum(path.startswith('merra-2/') for path in requested),
              'climatology_objects': sum(path.startswith('climatology/') for path in requested),
              'missing': missing, 'missing_paths': missing, 'total_available_bytes': sum(e['size'] for e in selected),
              'verified_cache_objects': sum(e['cache_status'] == 'verified_complete' for e in selected),
              'remaining_bytes': sum(e['size'] for e in selected if e['cache_status'] != 'verified_complete'),
              'origins': mapping, 'files': selected,
              'limitations': ['metadata existence is not decoded-channel validation',
                              'no replacement dates and no large downloads',
                              'upstream SHA256 metadata is verified against local content only for complete cache objects']}
    (HERE / 'pilot_file_inventory.json').write_text(json.dumps(result, indent=2), 'utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('origins', 'files')}, indent=2))


if __name__ == '__main__':
    main()
