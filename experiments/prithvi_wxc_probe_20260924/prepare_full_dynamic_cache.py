"""Build all planned physical dynamic cubes; preserve boundary uncertainty."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from raw_dynamic_adapter import HERE, assemble, PRODUCTS
from audit_full_import import parquet, derive_requirements
from scripts.import_merra_history import atomic_json, validate
from scripts.download_merra_regional_sample import file_hash


def utc(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('Aware UTC-convertible timestamp required')
    return stamp.astimezone(timezone.utc)


def boundary_flag(value):
    stamp = utc(value)
    return (stamp.month, stamp.day, stamp.hour, stamp.minute, stamp.second, stamp.microsecond) == (1, 1, 0, 0, 0, 0)


def cache_name(value):
    stamp = utc(value)
    if stamp.minute or stamp.second or stamp.microsecond or stamp.hour % 6:
        raise ValueError('Six-hour cadence required')
    return stamp.strftime('%Y%m%dT%H%M.npz')


def validate_cube(value):
    if value.shape != (160, 30, 32) or value.dtype != np.float32 or not np.isfinite(value).all() or np.any(np.abs(value) >= 1e14):
        raise ValueError('Invalid dynamic cube')


def main():
    base = HERE / 'full_history'
    out = base / 'dynamic_cubes_centered_v1'
    out.mkdir(exist_ok=True)
    source = base / 'support_aligned_plan/candidate_hours.parquet'
    watched = [source, base / 'support_aligned_plan/required_weather_times.parquet',
               base / 'dynamic_import/manifest.json', HERE / 'vendor/config.yaml',
               HERE / 'raw_dynamic_adapter.py', Path(__file__)]
    hashes = {str(path): file_hash(path) for path in watched}
    candidates = parquet(source)
    requirements = derive_requirements(candidates)
    times = [row['weather_time_utc'] for row in requirements['times']]
    if times != [utc(r['weather_time_utc']).isoformat() for r in parquet(watched[1])]:
        raise ValueError('Required time plan mismatch')
    manifest = json.loads(watched[2].read_text('utf-8'))
    config = yaml.safe_load(watched[3].read_text())['data']
    records, raw_hashes = [], {}
    mapping = []
    for row in candidates:
        pair = [row['support_earlier_utc'], row['support_latest_utc']]
        mapping.append({'origin': row['timestamp'], 'partition': row['partition'],
                        'files': [cache_name(t) for t in pair],
                        'year_start_boundary_uncertain': any(boundary_flag(t) for t in pair)})
    report = {'status': 'running', 'policy': 'centered_float32_no_clamp_v1',
              'source_sha256': hashes, 'raw_source_sha256': raw_hashes, 'files': records,
              'required_times': len(times), 'candidates': mapping,
              'boundary_flagged_candidates': sum(r['year_start_boundary_uncertain'] for r in mapping),
              'training_allowed': False, 'full_history_preprocessing_verified': False,
              'scope': 'Physical dynamic cache only; uncertainty flags are retained, not exclusions.'}
    atomic_json(out / 'manifest.json', report)
    groups = {}
    for time in times:
        groups.setdefault(time[:10], []).append(time)
    for day, group in groups.items():
        with ExitStack() as stack:
            products, dependencies = {}, []
            for product in PRODUCTS:
                centers = [np.datetime64(utc(t).replace(tzinfo=None), 'ns') for t in group]
                targets = [t + np.timedelta64(m, 'm') for t in centers for m in ((-30, 30) if product.startswith('M2T') else (0,))]
                pieces = []
                for date in sorted({str(t.astype('datetime64[D]')) for t in targets}):
                    key = product + '/' + date
                    entry = manifest['files'][key]
                    path = base / 'dynamic_import' / product / (date + '.nc4')
                    if str(path) not in raw_hashes:
                        digest = file_hash(path)
                        if digest != entry['sha256'] or path.stat().st_size != entry['bytes']:
                            raise ValueError('Raw checksum mismatch: ' + key)
                        validate(path, entry['job'])
                        raw_hashes[str(path)] = digest
                    pieces.append(stack.enter_context(xr.open_dataset(path, engine='h5netcdf')))
                    dependencies.append(key)
                products[product] = xr.concat(pieces, dim='time').sortby('time')
            for time in group:
                stamp = np.datetime64(utc(time).replace(tzinfo=None), 'ns')
                cube = assemble(products, stamp, config)
                validate_cube(cube)
                path = out / cache_name(time)
                temporary = path.with_suffix('.npz.partial')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, x=cube)
                with np.load(temporary, allow_pickle=False) as restored:
                    validate_cube(restored['x'])
                    if not np.array_equal(restored['x'], cube):
                        raise ValueError('Serialization changed values')
                temporary.replace(path)
                records.append({'time': time, 'path': path.name, 'sha256': file_hash(path),
                                'source_jobs': dependencies.copy(),
                                'year_start_boundary_uncertain': boundary_flag(time)})
        if len(records) % 20 < len(group):
            atomic_json(out / 'manifest.json', report)
            print(json.dumps({'prepared': len(records), 'required': len(times)}), flush=True)
    if any(file_hash(Path(path)) != digest for path, digest in {**hashes, **raw_hashes}.items()):
        raise ValueError('Source changed during build')
    complete = {record['path'] for record in records}
    if len(records) != len(times) or not all(set(row['files']) <= complete for row in mapping):
        raise ValueError('Incomplete candidate mapping')
    report['status'] = 'dynamic_cache_complete_not_training_ready'
    report['fully_mapped_candidates'] = len(mapping)
    atomic_json(out / 'manifest.json', report)
    print(json.dumps({'status': report['status'], 'cubes': len(records), 'candidates': len(mapping),
                      'boundary_flagged_candidates': report['boundary_flagged_candidates']}), flush=True)


if __name__ == '__main__':
    main()
