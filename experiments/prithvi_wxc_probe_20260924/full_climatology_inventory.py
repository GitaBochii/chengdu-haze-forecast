"""Inventory pinned climatology metadata for every support-aligned origin."""
from datetime import datetime, timezone
import json
import os
import tempfile

import pandas as pd
import requests

from prepare_pilot_inventory import HERE, REVISION, CANONICAL_API, sha, validate_entry


def required_paths(frame):
    paths = set()
    for value in frame['support_latest_utc']:
        latest = pd.Timestamp(value)
        if pd.isna(latest) or latest.tzinfo is None:
            raise ValueError('timezone-aware support timestamp required')
        latest = latest.tz_convert('UTC')
        if latest != latest.floor('6h'):
            raise ValueError('six-hour support timestamp required')
        target = latest + pd.Timedelta(hours=6)
        doy = min(target.dayofyear, 365)
        for kind in ('surface', 'vertical'):
            paths.add(f'climatology/climate_{kind}_doy{doy:03}_hour{target.hour:02}.nc')
    if not paths:
        raise ValueError('empty source inventory')
    return sorted(paths)


def collect_metadata(session, max_pages=20):
    entries, pages, visited = {}, [], set()
    url = CANONICAL_API
    for _ in range(max_pages):
        if not url:
            break
        if not (url == CANONICAL_API or url.startswith(CANONICAL_API + '?')):
            raise ValueError('unexpected pagination destination')
        if url in visited:
            raise ValueError('duplicate pagination URL')
        visited.add(url)
        response = session.get(url, timeout=(20, 90), allow_redirects=False)
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list):
            raise ValueError('unexpected metadata payload')
        pages.append({'page': len(pages) + 1, 'entries': len(page)})
        for entry in page:
            path = entry['path']
            if path in entries:
                raise ValueError('duplicate metadata entry')
            entries[path] = entry
        url = response.links.get('next', {}).get('url')
    if url:
        raise ValueError('metadata pagination incomplete')
    return entries, pages


def select_entries(paths, entries):
    selected, missing = [], []
    for path in paths:
        if path not in entries:
            missing.append(path)
            continue
        entry = entries[path]
        validate_entry(entry)
        selected.append({'path': path, 'size': entry['size'],
                         'lfs': {'oid': entry['lfs']['oid'], 'size': entry['lfs']['size']},
                         'revision': REVISION})
    return selected, missing


def main():
    source = HERE / 'full_history/support_aligned_plan/candidate_hours.parquet'
    source_sha = sha(source)
    calendar = HERE / 'vendor/PrithviWxC/dataloaders/merra2.py'
    calendar_sha = sha(calendar)
    if 'min(self.target.dayofyear, 365)' not in calendar.read_text('utf-8'):
        raise ValueError('pinned official calendar convention changed')
    frame = pd.read_parquet(source)
    paths = required_paths(frame)
    with requests.Session() as session:
        session.trust_env = False
        session.proxies = requests.utils.get_environ_proxies(CANONICAL_API)
        entries, pages = collect_metadata(session)
    selected, missing = select_entries(paths, entries)
    if sha(source) != source_sha or sha(calendar) != calendar_sha:
        raise ValueError('source changed during metadata query')
    result = {
        'status': 'complete_metadata_inventory' if not missing else 'missing_selected_objects',
        'queried_at_utc': datetime.now(timezone.utc).isoformat(),
        'revision': REVISION, 'metadata_endpoint': CANONICAL_API,
        'source': str(source.relative_to(HERE)), 'source_sha256': source_sha,
        'source_rows': len(frame), 'source_unchanged_after_query': True,
        'official_calendar_source_sha256': calendar_sha,
        'calendar_rule': 'target = support_latest_utc + 6h; min(target.dayofyear,365), target.hour',
        'metadata_pages': pages, 'metadata_pagination_complete': True,
        'all_catalog_entries': len(entries), 'required_files': len(paths),
        'metadata_found': len(selected), 'missing': missing,
        'full_global_objects_bytes': sum(entry['size'] for entry in selected),
        'files': selected, 'data_downloaded': False,
        'limitations': ['Metadata existence does not prove local import or decoded channel validity.',
                        'No global objects are downloaded; regional acquisition remains separate.']}
    output = HERE / 'full_history/climatology_inventory.json'
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output.parent,
                                     suffix='.tmp', delete=False) as stream:
        temporary = stream.name
        json.dump(result, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
