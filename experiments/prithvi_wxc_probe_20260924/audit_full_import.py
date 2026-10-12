"""Read-only data audit; the sole write is the requested audit report."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import xarray as xr
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts import import_merra_history as dynamic
from scripts import import_merra_static as static
from import_climatology_regions import cache_valid, REVISION, file_sha


def timestamp(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('Naive timestamp')
    return value.astimezone(timezone.utc)


def derive_requirements(candidates):
    origins, times, climate, mappings = set(), set(), set(), []
    for row in candidates:
        origin = timestamp(row['timestamp'])
        if origin in origins:
            raise ValueError('Duplicate origin')
        origins.add(origin)
        shifted = origin - timedelta(hours=1)
        latest = shifted.replace(hour=shifted.hour // 6 * 6, minute=0, second=0, microsecond=0)
        earlier = latest - timedelta(hours=6)
        for key, expected in [('support_latest_utc', latest), ('support_earlier_utc', earlier),
                              ('support_end_utc', latest + timedelta(hours=1))]:
            if timestamp(row[key]) != expected:
                raise ValueError('Candidate support rule mismatch: ' + key)
        times.update([earlier, latest])
        target = latest + timedelta(hours=6)
        names = [f'climatology/climate_{kind}_doy{min(target.timetuple().tm_yday,365):03d}_hour{target.hour:02d}.nc'
                 for kind in ('surface', 'vertical')]
        climate.update(names)
        keys = set()
        for instant in (earlier, latest):
            for product in dynamic.PRODUCTS:
                sources = (instant - timedelta(minutes=30), instant + timedelta(minutes=30)) if product in dynamic.AVERAGED else (instant,)
                keys.update(product + '/' + source.date().isoformat() for source in sources)
        mappings.append({'origin': origin.isoformat(), 'dynamic': sorted(keys), 'climate': names})
    if not origins:
        raise ValueError('Empty candidates')
    rows = [{'weather_time_utc': t.isoformat(),
             'source_left_utc': (t-timedelta(minutes=30)).isoformat(),
             'source_right_utc': (t+timedelta(minutes=30)).isoformat(),
             'source_right_support_end_utc': (t+timedelta(hours=1)).isoformat()} for t in sorted(times)]
    return {'times': rows, 'climate': climate, 'mappings': mappings}


def completed(metadata_only, errors, dynamic_valid, dynamic_total, climate_valid, climate_total):
    return not metadata_only and not errors and dynamic_valid == dynamic_total and climate_valid == climate_total


def read_json(path):
    return json.loads(path.read_text('utf-8'))


def parquet(path):
    code = "import pandas as pd,sys; print(pd.read_parquet(sys.argv[1]).to_json(orient='records',date_format='iso'))"
    return json.loads(subprocess.check_output([str(ROOT/'.venv_air_models/Scripts/python.exe'),
                                               '-X', 'utf8', '-c', code, str(path)]))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(metadata_only=False):
    base = HERE/'full_history'
    plan = base/'support_aligned_plan'
    errors, valid_dynamic, valid_climate = [], set(), set()
    watched = [base/'candidate_hours.parquet', base/'coverage.json', plan/'candidate_hours.parquet',
               plan/'required_weather_times.parquet', plan/'coverage.json', plan/'catalog/coverage.json',
               base/'climatology_inventory.json', HERE/'vendor/config.yaml', HERE/'real_sample_manifest.json',
               HERE/'pilot_batch_report.json', ROOT/'data/processed/regional_air_models_full_history/city_hourly.parquet']
    initial = {str(path): file_sha(path) for path in watched}
    def check(label, operation):
        try:
            operation()
            return True
        except Exception as exc:
            errors.append({'item': label, 'error': str(exc), 'error_type': type(exc).__name__})
            return False
    candidates = parquet(plan/'candidate_hours.parquet')
    requirements = derive_requirements(candidates)
    actual_times = parquet(plan/'required_weather_times.parquet')
    normalized = [{k: timestamp(r[k]).isoformat() for k in requirements['times'][0]} for r in actual_times]
    check('required_time_derivation', lambda: require(normalized == requirements['times'], 'Required times differ from candidates'))
    original = parquet(base/'candidate_hours.parquet')
    check('candidate_preservation', lambda: require(len(original) == len(candidates) and
          all(all(b.get(k) == v for k, v in a.items()) for a,b in zip(original,candidates)), 'Original candidates changed'))
    check('source_hash_chain', lambda: require(
        read_json(base/'coverage.json')['source_sha256'] == initial[str(watched[-1])] and
        read_json(plan/'coverage.json')['source_sha256'] == initial[str(base/'candidate_hours.parquet')] and
        read_json(plan/'catalog/coverage.json')['required_times_sha256'] == initial[str(plan/'required_weather_times.parquet')],
        'Source hash chain mismatch'))
    inventory = read_json(base/'climatology_inventory.json')
    calendar = HERE/'vendor/PrithviWxC/dataloaders/merra2.py'
    initial[str(calendar)] = file_sha(calendar)
    entries = {r['path']: r for r in inventory['files']}
    check('climate_inventory', lambda: require(inventory['revision'] == REVISION and
        inventory['source_sha256'] == initial[str(plan/'candidate_hours.parquet')] and
        inventory['metadata_pagination_complete'] and not inventory['missing'] and
        inventory['official_calendar_source_sha256'] == initial[str(calendar)] and
        len(entries) == len(inventory['files']) and set(entries) == requirements['climate'], 'Climatology inventory mismatch'))
    config = yaml.safe_load((HERE/'vendor/config.yaml').read_text('utf-8'))['data']
    check('model_channels', lambda: require(len(config['surface_vars']) == 20 and len(config['vertical_vars']) == 10 and
        len(set(config['levels'])) == 14 and set(config['levels']) <= set(range(34,73)), 'Unexpected model channels'))
    jobs = dynamic.time_jobs(requirements['times'])
    manifest = read_json(base/'dynamic_import/manifest.json')
    catalogs = {p: read_json(plan/'catalog'/f'{p}.json') for p in dynamic.PRODUCTS}
    for job in jobs:
        key = job['product'] + '/' + job['date']
        def verify_dynamic():
            matches = [g for g in catalogs[job['product']]['granules'] if g['time_start'].startswith(job['date']+'T')]
            require(len(matches) == 1, 'Missing or ambiguous daily source')
            expected = dict(job, source_url=dynamic.access.opendap_url(matches[0]['links']), constraint=dynamic.constraint(job))
            record = manifest['files'][key]
            require(record['job'] == expected, 'Source job mismatch')
            if not metadata_only:
                path = base/'dynamic_import'/job['product']/(job['date']+'.nc4')
                require(path.stat().st_size == record['bytes'] and file_sha(path) == record['sha256'], 'Dynamic checksum mismatch')
                dynamic.validate(path, job)
        if check(key, verify_dynamic) and not metadata_only:
            valid_dynamic.add(key)
    for name in sorted(requirements['climate']):
        def verify_climate():
            entry = entries[name]
            require(len(entry['lfs']['oid']) == 64 and all(c in '0123456789abcdef' for c in entry['lfs']['oid']) and
                    entry['size'] == entry['lfs']['size'], 'Invalid upstream digest metadata')
            if not metadata_only:
                path = base/'climatology_regions'/(Path(name).stem+'.npz')
                vertical = 'climate_vertical_' in name
                require(cache_valid(path, entry, initial[str(HERE/'vendor/config.yaml')],
                    config['vertical_vars' if vertical else 'surface_vars'], config['levels'] if vertical else None),
                    'Climatology sidecar, checksum or numerical validation failed')
                require(read_json(path.with_suffix('.json'))['status'] == 'imported', 'Invalid sidecar status')
        if check(name, verify_climate) and not metadata_only:
            valid_climate.add(name)
    frozen = read_json(HERE/'real_sample_manifest.json')
    def verify_static_scalers():
        require(frozen['revision'] == REVISION, 'Frozen revision mismatch')
        names = ['climatology/'+n+'.nc' for n in ('musigma_surface','musigma_vertical','anomaly_variance_surface','anomaly_variance_vertical')]
        records = {r['path']:r for r in frozen['downloads']}
        for name in names:
            record = records[name]
            require(record['upstream_sha256_verified'], 'Scaler upstream digest unverified')
            if not metadata_only:
                path = HERE/'real_sample'/name
                require(path.stat().st_size == record['bytes'] and file_sha(path) == record['sha256'], 'Scaler hash mismatch')
        report = read_json(base/'static_import/report.json')
        if metadata_only:
            return
        pieces = {}
        for product in static.PRODUCTS:
            record = report['products'][product]
            path = base/'static_import'/(product+'.nc4')
            require(record['source_url'] == static.source_url(product) and record['constraint'] == static.constraint(product), 'Static provenance mismatch')
            require(file_sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], 'Static checksum mismatch')
            with xr.open_dataset(path,engine='h5netcdf') as ds:
                validation = static.validate(ds, product)
                require(validation['all_months_equal'], 'Static monthly inconsistency')
                pieces[product] = ds.load()
        reference = HERE/'real_sample/merra-2/MERRA2_sfc_20200101.nc'
        frozen_hashes = {r['source_sha256']['merra-2/MERRA2_sfc_20200101.nc'] for r in read_json(HERE/'pilot_batch_report.json')['rows']
                         if 'merra-2/MERRA2_sfc_20200101.nc' in r.get('source_sha256', {})}
        require(frozen_hashes == {file_sha(reference)}, 'Frozen static reference hash mismatch')
        with xr.open_dataset(reference,engine='h5netcdf') as ref:
            require(static.compare_reference(pieces['M2C0NXCTM'],ref)['all_equal'], 'Static reference mismatch')
        for name in static.PRODUCTS['M2C0NXASM']['variables']:
            require(np.all(pieces['M2C0NXASM'][name].values == pieces['M2C0NXCTM'][name].values), 'Static products disagree')
    check('static_and_scalers', verify_static_scalers)
    mapped = sum(set(row['dynamic']) <= valid_dynamic and set(row['climate']) <= valid_climate for row in requirements['mappings'])
    check('sources_unchanged', lambda: require(all(file_sha(Path(p)) == sha for p,sha in initial.items()), 'Source changed during audit'))
    return {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'metadata_only': metadata_only,
            'source_sha256': initial, 'candidate_rows': len(candidates), 'fully_mapped_candidates': mapped,
            'required_weather_times': len(requirements['times']), 'dynamic_required': len(jobs),
            'dynamic_verified': len(valid_dynamic), 'climate_required': len(requirements['climate']),
            'climate_verified': len(valid_climate), 'errors': errors,
            'data_import_complete': completed(metadata_only, errors, len(valid_dynamic),len(jobs),len(valid_climate),len(requirements['climate'])) and mapped == len(candidates),
            'preprocessing_complete': False, 'training_complete': False,
            'limitations': ['Regional range artifacts have local checksums; global climate source SHA is metadata, not a full-object recomputation.',
                            'Import completeness does not prove real-time availability or preprocessing compatibility.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE/'full_history/import_audit.json')
    args = parser.parse_args()
    result = audit(args.metadata_only)
    dynamic.atomic_json(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','errors','limitations')}, indent=2))
    print('errors:', len(result['errors']))
    if result['errors'] or (not args.metadata_only and not result['data_import_complete']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
