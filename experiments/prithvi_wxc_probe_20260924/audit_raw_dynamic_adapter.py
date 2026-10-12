"""Validate assembled 160-channel inputs against pinned prepared references."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from raw_dynamic_adapter import assemble, HERE, LAT, LON, PRODUCTS
from scripts.import_merra_history import atomic_json
from scripts.download_merra_regional_sample import file_hash


def main():
    base = HERE / 'full_history'
    watched = [base / 'dynamic_import/manifest.json', HERE / 'pilot_batch_report.json',
               HERE / 'vendor/config.yaml', HERE / 'raw_dynamic_adapter.py']
    hashes = {str(path): file_hash(path) for path in watched}
    manifest = json.loads(watched[0].read_text('utf-8'))
    pilot = json.loads(watched[1].read_text('utf-8'))
    config = yaml.safe_load(watched[2].read_text('utf-8'))['data']
    references = {}
    for row in pilot['rows']:
        for relative, digest in row.get('source_sha256', {}).items():
            if relative in references and references[relative] != digest:
                raise ValueError('Conflicting frozen hashes')
            references[relative] = digest

    def verify(path, digest):
        if str(path) not in hashes:
            if file_hash(path) != digest:
                raise ValueError('Input checksum mismatch: ' + str(path))
            hashes[str(path)] = digest

    rows, skipped = [], []
    labels = config['surface_vars'] + [f'{v}@{int(level)}' for v in config['vertical_vars'] for level in config['levels']]
    for relative, digest in sorted(references.items()):
        if not relative.startswith('merra-2/MERRA2_sfc_'):
            continue
        vertical_relative = relative.replace('MERRA2_sfc_', 'MERRA_pres_')
        if vertical_relative not in references:
            raise ValueError('Frozen vertical counterpart absent')
        surface_path, vertical_path = HERE / 'real_sample' / relative, HERE / 'real_sample' / vertical_relative
        verify(surface_path, digest)
        verify(vertical_path, references[vertical_relative])
        with ExitStack() as stack:
            surface = stack.enter_context(xr.open_dataset(surface_path, engine='h5netcdf'))
            vertical = stack.enter_context(xr.open_dataset(vertical_path, engine='h5netcdf'))
            day = surface.time.values[0].astype('datetime64[D]')
            products = {}
            for product in PRODUCTS:
                pieces = []
                for date in (str(day - np.timedelta64(1, 'D')), str(day)):
                    key = product + '/' + date
                    if key not in manifest['files']:
                        continue
                    path = base / 'dynamic_import' / product / (date + '.nc4')
                    verify(path, manifest['files'][key]['sha256'])
                    pieces.append(stack.enter_context(xr.open_dataset(path, engine='h5netcdf')))
                if pieces:
                    products[product] = xr.concat(pieces, dim='time').sortby('time')
            for time in surface.time.values:
                missing = []
                for product in PRODUCTS:
                    targets = [time - np.timedelta64(30, 'm'), time + np.timedelta64(30, 'm')] if product.startswith('M2T') else [time]
                    if product not in products or not all(t in products[product].time.values for t in targets):
                        missing.append(product)
                if missing:
                    skipped.append({'time': str(time), 'missing_overlap': missing})
                    continue
                actual = assemble(products, time, config)
                s = surface.sel(time=time, lat=LAT, lon=LON)
                v = vertical.sel(time=time, lat=LAT, lon=LON, lev=config['levels'])
                expected = np.concatenate([
                    np.stack([s[name].transpose('lat', 'lon').values for name in config['surface_vars']]),
                    np.stack([v[name].transpose('lev', 'lat', 'lon').values for name in config['vertical_vars']]).reshape(140, 30, 32)
                ]).astype('float32')
                if not np.isfinite(expected).all():
                    raise ValueError('Nonfinite reference')
                counts = (actual == expected).sum(axis=(1, 2))
                rows.append({'time': str(time), 'equal': bool(np.array_equal(actual, expected)),
                             'equal_cells': int(counts.sum()), 'cells': int(actual.size),
                             'mismatched_channels': [labels[i] for i in np.flatnonzero(counts != 960)],
                             'max_abs_difference': float(np.abs(actual.astype('float64') - expected).max())})
    if any(file_hash(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('Source changed during audit')
    result = {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'channel_labels': labels,
              'source_sha256': hashes, 'rows': rows, 'skipped': skipped,
              'checked_times': len(rows), 'exact_times': sum(row['equal'] for row in rows),
              'compared_cells': sum(row['cells'] for row in rows),
              'full_history_preprocessing_verified': False, 'training_performed': False,
              'scope': 'Raw physical dynamics only. Static, climate, normalization and encoder batching are not validated here.'}
    atomic_json(base / 'raw_dynamic_adapter_audit.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('rows', 'skipped', 'source_sha256', 'channel_labels')}, indent=2))
    print('mismatches', json.dumps([row for row in rows if not row['equal']]))
    print('skipped', len(skipped))


if __name__ == '__main__':
    main()
