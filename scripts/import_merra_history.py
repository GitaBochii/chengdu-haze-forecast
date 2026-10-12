"""Resumable raw regional MERRA-2 import; never interpolates averaged fields."""
import argparse
import contextlib
from datetime import datetime, timezone
from functools import reduce
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_merra_regional_sample import (
    BASE, LAT, LON, LEVELS, SURFACE, VERTICAL, download_subset, file_hash,
)
from scripts.download_merra_averaged_sample import PRODUCTS as AVERAGED
from scripts import verify_merra_access as access
from scripts.download_satellite_sample import credential_form, login_from_form

PRODUCTS = {'M2I1NXASM': SURFACE, 'M2I3NVASM': VERTICAL, **AVERAGED}
PLAN = BASE / 'support_aligned_plan'
OUT = BASE / 'dynamic_import'
# Observed in regional_samples/20260926T084925166866Z and
# boundary_suites/20260926T094023388995Z, not inferred from variable names.
EXPECTED_UNITS = dict(
    U10M='m s-1', V10M='m s-1', T2M='K', QV2M='kg kg-1', PS='Pa',
    SLP='Pa', TS='K', TQI='kg m-2', TQL='kg m-2', TQV='kg m-2',
    CLOUD='1', H='m', OMEGA='Pa s-1', PL='Pa', QI='kg kg-1',
    QL='kg kg-1', QV='kg kg-1', T='K', U='m s-1', V='m s-1',
    GWETROOT='1', LAI='1', EFLUX='W m-2', HFLUX='W m-2', Z0M='m',
    LWGEM='W m-2', LWGAB='W m-2', LWTUP='W m-2', SWGNT='W m-2', SWTNT='W m-2')


def atomic_json(path, value):
    value['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def time_jobs(rows):
    grouped = {}
    for row in rows:
        for product in PRODUCTS:
            columns = ('source_left_utc', 'source_right_utc') if product in AVERAGED else ('weather_time_utc',)
            for column in columns:
                time = datetime.fromisoformat(row[column].replace('Z', '+00:00'))
                if time.tzinfo is None:
                    raise ValueError('Source timestamps must be timezone-aware')
                time = time.astimezone(timezone.utc)
                average = product in AVERAGED
                if time.second or time.microsecond or time.minute != (30 if average else 0):
                    raise ValueError('Invalid source cadence')
                if product == 'M2I3NVASM' and time.hour % 3:
                    raise ValueError('Invalid native vertical cadence')
                index = time.hour // 3 if product == 'M2I3NVASM' else time.hour
                grouped.setdefault((product, time.date().isoformat()), set()).add(index)
    if not grouped:
        raise ValueError('Empty required times')
    return [dict(product=p, date=d, indices=sorted(indices))
            for (p, d), indices in sorted(grouped.items(), key=lambda item:(item[0][1],item[0][0]))]


def returned_indices(job):
    values = job['indices']
    stride = reduce(math.gcd, [b-a for a, b in zip(values, values[1:])], 0) or 1
    return list(range(values[0], values[-1]+1, stride))


def constraint(job):
    indices = returned_indices(job)
    stride = indices[1]-indices[0] if len(indices)>1 else 1
    time = f'[{indices[0]}:{stride}:{indices[-1]}]'
    vertical = job['product']=='M2I3NVASM'
    parts = ['/time'+time, '/lat[226:1:255]', '/lon[438:1:469]']
    if vertical:
        parts.append('/lev[33:1:71]')
    parts.extend('/'+name+time+('[33:1:71]' if vertical else '')+
                 '[226:1:255][438:1:469]' for name in PRODUCTS[job['product']])
    return ';'.join(parts)


def validate(path, job):
    vertical = job['product']=='M2I3NVASM'
    indices = returned_indices(job)
    minutes = np.array(indices)*(180 if vertical else 60)+(30 if job['product'] in AVERAGED else 0)
    expected = np.datetime64(job['date'], 'm')+minutes.astype('timedelta64[m]')
    with xr.open_dataset(path, engine='h5netcdf') as ds:
        for name, values in [('time', expected), ('lat', LAT), ('lon', LON)]:
            if name not in ds.coords or not np.array_equal(ds[name].values, values):
                raise ValueError('Unexpected coordinates: '+name)
        if vertical and not np.array_equal(ds.lev.values, np.arange(34,73)):
            raise ValueError('Unexpected native model levels')
        dims = ('time','lev','lat','lon') if vertical else ('time','lat','lon')
        units = {}
        for name in PRODUCTS[job['product']]:
            value = ds[name].transpose(*dims)
            unit = value.attrs.get('units')
            unit = unit.decode() if isinstance(unit, bytes) else unit
            if unit != EXPECTED_UNITS[name] or not np.isfinite(value.values).all():
                raise ValueError('Missing units or finite values: '+name)
            units[name] = unit.decode() if isinstance(unit, bytes) else str(unit)
    return units


def reusable(path, job, record):
    try:
        if not record or record.get('job') != job or not path.is_file():
            return False
        if path.stat().st_size != record['bytes'] or file_hash(path) != record['sha256']:
            return False
        validate(path, job)
        return True
    except (KeyError, ValueError, OSError):
        return False


def load_jobs():
    # The satellite runtime has HDF5/Earthdata; the model runtime owns parquet.
    source = PLAN/'required_weather_times.parquet'
    digest = file_hash(source)
    code = "import pandas as pd,sys; print(pd.read_parquet(sys.argv[1]).to_json(orient='records',date_format='iso'))"
    raw = subprocess.check_output([str(ROOT/'.venv_air_models/Scripts/python.exe'),
                                   '-X', 'utf8', '-c', code, str(source)])
    jobs = time_jobs(json.loads(raw))
    coverage = json.loads((PLAN/'catalog/coverage.json').read_text('utf-8'))
    if coverage.get('required_times_sha256') != digest or not coverage['all_required_dates_have_metadata']:
        raise ValueError('Catalog does not cover current required-time plan')
    catalogs = {p: json.loads((PLAN/'catalog'/f'{p}.json').read_text('utf-8')) for p in PRODUCTS}
    for job in jobs:
        matches = [g for g in catalogs[job['product']]['granules'] if g['time_start'].startswith(job['date']+'T')]
        if len(matches)!=1:
            raise ValueError('Missing or ambiguous daily source')
        job['source_url'] = access.opendap_url(matches[0]['links'])
        job['constraint'] = constraint(job)
    if file_hash(source)!=digest:
        raise ValueError('Required plan changed during read')
    return digest, jobs


def retry(operation, sleep=time.sleep):
    for attempt in range(3):
        try:
            return operation()
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status in (401,403) or attempt == 2:
                raise
            sleep(2**attempt)


class CheckedSession:
    """Preserve unauthorized response codes hidden by the bounded sample helper."""
    def __init__(self, session):
        self.session = session

    def get(self, *args, **kwargs):
        response = self.session.get(*args, **kwargs)
        if response.status_code in (401,403):
            response.close()
            raise requests.HTTPError('Authorization rejected', response=response)
        return response


def run_import(session, jobs, manifest, output=OUT, limit=None):
    downloaded = 0
    verified = 0
    for job in jobs:
        key = job['product']+'/'+job['date']
        target = output/job['product']/(job['date']+'.nc4')
        old = manifest['files'].get(key)
        if reusable(target, job, old):
            verified += 1
            continue
        if limit is not None and downloaded >= limit:
            manifest.update(status='batch_limit_reached', raw_dynamic_complete=False)
            atomic_json(output/'manifest.json', manifest)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.partial')
        manifest.update(status='downloading', current=key)
        atomic_json(output/'manifest.json', manifest)
        try:
            result = retry(lambda: download_subset(CheckedSession(session), job['source_url'], job['constraint'], temporary))
            units = validate(temporary, job)
            os.replace(temporary, target)
            manifest['files'][key] = dict(job=job, **result, units=units,
                                          verified_at_utc=datetime.now(timezone.utc).isoformat())
            manifest['failures'].pop(key, None)
            downloaded += 1
            verified += 1
            manifest['verified_jobs_this_run'] = verified
            atomic_json(output/'manifest.json', manifest)
            print(json.dumps(dict(verified_files=verified, expected_files=len(jobs),
                                  product=job['product'], date=job['date'])), flush=True)
        except Exception as exc:
            manifest['files'].pop(key, None)
            manifest['failures'][key] = {'error_type':type(exc).__name__}
            manifest.update(status='failed', raw_dynamic_complete=False)
            atomic_json(output/'manifest.json', manifest)
            raise
        finally:
            temporary.unlink(missing_ok=True)
    if verified != len(jobs):
        raise ValueError('Not every required job verified')
    manifest.update(status='complete', raw_dynamic_complete=True, current=None,
                    verified_jobs_this_run=verified)
    atomic_json(output/'manifest.json', manifest)


def run_authenticated(session, jobs, manifest, limit=None):
    from scripts.import_merra_static import run_import as import_static
    manifest.update(status='importing_model_constants', static_fields_included=False)
    atomic_json(OUT/'manifest.json', manifest)
    report = retry(lambda: import_static(session, BASE/'static_import'))
    manifest['static_constants_ready'] = bool(report['ready_as_frozen_model_constants'])
    manifest['static_report'] = str(BASE/'static_import/report.json')
    atomic_json(OUT/'manifest.json', manifest)
    run_import(session, jobs, manifest, output=OUT, limit=limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int, help='Maximum new downloads; existing files are verified and skipped')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error('--limit must be non-negative')
    digest, jobs = load_jobs()
    summary = dict(required_time_sha256=digest, daily_requests=len(jobs),
                   products={p:sum(j['product']==p for j in jobs) for p in PRODUCTS},
                   model_levels=LEVELS, downloaded_native_levels=list(range(34,73)),
                   uncompressed_float32_bytes=sum(len(returned_indices(j))*30*32*4*
                       len(PRODUCTS[j['product']])*(39 if j['product']=='M2I3NVASM' else 1) for j in jobs),
                   raw_dynamic_complete=False, preprocessing_complete=False,
                   static_fields_included=False,
                   static_source_note='FRACI requires M2C0NXCTM; FRLAND/FROCEAN/PHIS require constant collection, not M2I1NXASM',
                   requested_new_download_limit=args.limit)
    if args.dry_run:
        print(json.dumps(summary, indent=2)); return 0
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT/'manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8')) if manifest_path.exists() else {'files':{},'failures':{}}
    if manifest.get('required_time_sha256', digest)!=digest:
        raise ValueError('Manifest belongs to a different plan')
    manifest.update(summary, process_id=os.getpid(),status='waiting_for_local_login')
    atomic_json(manifest_path, manifest)
    import earthaccess
    credentials = credential_form()
    if credentials is None:
        manifest['status']='cancelled'; atomic_json(manifest_path, manifest); return 1
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            auth = login_from_form(earthaccess, *credentials)
    finally:
        del credentials
    if not auth.authenticated:
        manifest['status']='authentication_failed'; atomic_json(manifest_path, manifest); return 1
    session = auth.get_session()
    try:
        run_authenticated(session, jobs, manifest, limit=args.limit)
    finally:
        session.close()
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        # Never echo Earthdata exception text, which can contain authentication URLs.
        path = OUT/'manifest.json'
        if path.exists():
            value = json.loads(path.read_text('utf-8'))
            if value.get('process_id') == os.getpid():
                value.update(status='failed', error_type=type(exc).__name__, raw_dynamic_complete=False)
                atomic_json(path, value)
        print('Import failed: '+type(exc).__name__, file=sys.stderr)
        sys.exit(1)
