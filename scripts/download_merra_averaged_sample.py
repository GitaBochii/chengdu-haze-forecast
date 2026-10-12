"""Check a bounded ten-channel time-averaging hypothesis against frozen HF data."""
from datetime import datetime, timezone
from pathlib import Path
import contextlib
import io
import json
import sys

import numpy as np
import requests
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import verify_merra_access as access
from scripts.download_merra_regional_sample import (
    BASE, HERE, LAT, LON, TIMES, download_subset, file_hash,
)
from scripts.download_satellite_sample import credential_form, login_from_form

PRODUCTS={
    'M2T1NXLND':['GWETROOT','LAI'],
    'M2T1NXFLX':['EFLUX','HFLUX','Z0M'],
    'M2T1NXRAD':['LWGEM','LWGAB','LWTUP','SWGNT','SWTNT'],
}
DATES=('2019-12-31','2020-01-01')


def source_times(times=TIMES):
    centers=np.asarray(times,dtype='datetime64[m]')
    return np.stack((centers-np.timedelta64(30,'m'),
                     centers+np.timedelta64(30,'m')),axis=1).reshape(-1)


def constraint(product,date,reference_date='2020-01-01'):
    previous=str(np.datetime64(reference_date,'D')-np.timedelta64(1,'D'))
    if date not in (previous,reference_date):
        raise ValueError('Unexpected sample day')
    time='[23:1:23]' if date==previous else '[0:1:6]'
    parts=['/time'+time,'/lat[226:1:255]','/lon[438:1:469]']
    parts.extend('/'+name+time+'[226:1:255][438:1:469]' for name in PRODUCTS[product])
    return ';'.join(parts)


def validate_daily(dataset, date, names,reference_date='2020-01-01'):
    previous=str(np.datetime64(reference_date,'D')-np.timedelta64(1,'D'))
    if date not in (previous,reference_date):
        raise ValueError('Unexpected sample day')
    start=np.datetime64(date,'m')
    minutes=[23*60+30] if date==previous else list(range(30,7*60,60))
    expected=start+np.array(minutes,dtype='timedelta64[m]')
    for key,values in [('time',expected),('lat',LAT),('lon',LON)]:
        if key not in dataset.coords or not np.array_equal(dataset[key].values,values):
            raise ValueError('Unexpected daily coordinates: '+key)
    for name in names:
        array=dataset[name].transpose('time','lat','lon').values
        if not np.isfinite(array).all():
            raise ValueError('Non-finite daily input: '+name)


def compare_averaged(downloaded,reference,names,times=TIMES):
    times=np.asarray(times,dtype='datetime64[m]')
    if times.shape!=(2,) or np.isnat(times).any() or not np.all(np.diff(times)>np.timedelta64(0,'m')):
        raise ValueError('Expected two ordered finite target times')
    for key,expected in [('lat',LAT),('lon',LON)]:
        if key not in downloaded.coords or not np.array_equal(downloaded[key].values,expected):
            raise ValueError('Unexpected spatial coordinates: '+key)
    if 'time' not in downloaded.coords or not downloaded.indexes['time'].is_unique:
        raise ValueError('Missing or duplicate source times')
    try:
        raw=downloaded.sel(time=source_times(times))
        ref=reference.sel(time=times,lat=LAT,lon=LON)
    except KeyError as exc:
        raise ValueError('Required exact source or reference time unavailable') from exc
    results={}
    for name in names:
        def unit(value):
            return value.decode() if isinstance(value,bytes) else value
        units=unit(raw[name].attrs.get('units'))
        reference_units=unit(ref[name].attrs.get('units'))
        if not units or (reference_units is not None and units!=reference_units):
            raise ValueError('Unit mismatch: '+name)
        values=raw[name].transpose('time','lat','lon').values
        expected=ref[name].transpose('time','lat','lon').values
        if (values.shape!=(4,len(LAT),len(LON)) or
                expected.shape!=(2,len(LAT),len(LON)) or
                not np.isfinite(values).all() or not np.isfinite(expected).all()):
            raise ValueError('Invalid numerical input: '+name)
        pairs=values.reshape(2,2,len(LAT),len(LON))
        native=pairs.mean(axis=1)
        double=pairs.astype('float64').mean(axis=1)
        # Diagnostic hypothesis only: do not replace the centered-mean result.
        boundary=native.copy()
        midnight=times==times.astype('datetime64[D]')
        boundary[midnight]=pairs[midnight,1]
        def difference(average):
            delta=np.abs(average.astype('float64')-expected.astype('float64'))
            return dict(equal=bool(np.array_equal(average,expected)),
                        max_abs_difference=float(delta.max()),
                        mean_abs_difference=float(delta.mean()),
                        max_abs_difference_by_time=delta.max(axis=(1,2)).tolist(),
                        equal_cells_by_time=(average==expected).sum(axis=(1,2)).tolist())
        results[name]=dict(unit=units,reference_unit=reference_units,
                           unit_check='verified' if reference_units else 'reference_units_missing',
                           cells=int(expected.size),
                           source_dtype=str(values.dtype),reference_dtype=str(expected.dtype),
                           native_precision=difference(native),float64_precision=difference(double),
                           daily_boundary_candidate=difference(boundary))
    return dict(all_native_equal=all(v['native_precision']['equal'] for v in results.values()),
                all_float64_equal=all(v['float64_precision']['equal'] for v in results.values()),
                all_units_verified=all(v['unit_check']=='verified' for v in results.values()),
                all_daily_boundary_candidate_equal=all(v['daily_boundary_candidate']['equal'] for v in results.values()),
                variables=results)


def comparison_report(products,downloaded_requests=0):
    return dict(date='2020-01-01',times_utc=['00:00','06:00'],products=products,
                source_times_utc=[str(t)+'Z' for t in source_times()],
                averaging_hypothesis='Arithmetic mean of adjacent nominal half-hour centers',
                alternative_hypothesis='At midnight only, use current-day first half-hour value; diagnostic candidate, not a validated all-history rule',
                inferred_support_end_utc=['2020-01-01T01:00:00Z','2020-01-01T07:00:00Z'],
                support_interval_evidence='Inferred from one-hour centered averages, not independently proven by metadata',
                deployment_boundary='MERRA-2 reanalysis; not real-time data. This hypothesis uses support after prediction origin.',
                downloaded_requests=downloaded_requests,expected_download_requests=6,
                checked_dynamic_channels=sum(len(PRODUCTS[p]) for p in products),
                sample_comparison_complete=set(products)==set(PRODUCTS),
                total_dynamic_channels=160,full_preprocessing_verified=False,
                full_history_downloaded=False)


def exact_granule(catalog,date):
    entries=[row for row in catalog['granules'] if row['time_start'].startswith(date+'T')]
    if not entries:
        collection_ids={row['collection_concept_id'] for row in catalog['granules']}
        if len(collection_ids)!=1:
            raise ValueError('Ambiguous public collection identifier')
        response=requests.get('https://cmr.earthdata.nasa.gov/search/granules.json',
            params={'collection_concept_id':next(iter(collection_ids)),
                    'temporal':date+'T00:00:00Z,'+date+'T23:59:59Z','page_size':10},timeout=60)
        try:
            response.raise_for_status()
            entries=[row for row in response.json()['feed']['entry']
                     if row['time_start'].startswith(date+'T')]
        finally:
            response.close()
    if len(entries)!=1:
        raise ValueError('Expected exactly one sample granule for date')
    access.opendap_url(entries[0]['links'])
    return entries[0]


def main():
    import earthaccess
    access.OUT=BASE/'averaged_sample_status.json'
    access.status('preparing_reference')
    batch=json.loads((HERE/'pilot_batch_report.json').read_text('utf-8'))
    relative='merra-2/MERRA2_sfc_20200101.nc'
    reference=HERE/'real_sample'/relative
    digest=file_hash(reference)
    pins={row['source_sha256'][relative] for row in batch['rows']
          if relative in row.get('source_sha256',{})}
    if pins!={digest}:
        raise ValueError('Frozen reference hash mismatch')
    jobs={}
    for product in PRODUCTS:
        catalog=json.loads((BASE/'catalog'/f'{product}.json').read_text('utf-8'))
        jobs[product]=[(date,exact_granule(catalog,date)) for date in DATES]
    access.status('waiting_for_local_login',purpose='averaged_numeric_sample')
    credentials=credential_form()
    if credentials is None:
        access.status('cancelled_by_user')
        return 1
    access.status('authenticating')
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        try:
            auth=login_from_form(earthaccess,*credentials)
        finally:
            del credentials
    if not auth.authenticated:
        access.status('authentication_failed')
        return 1
    run=BASE/'averaged_samples'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True,exist_ok=False)
    products={}
    downloads=[]
    session=auth.get_session()
    def persist(state):
        report=comparison_report(products,len(downloads))
        report.update(downloads=downloads,reference_sha256=digest,status=state)
        access.OUT=run/'comparison.json'
        access.status(state,**{key:value for key,value in report.items() if key!='status'})
        access.OUT=BASE/'averaged_sample_status.json'
        access.status(state,run_directory=str(run),downloaded_requests=len(downloads),
                      checked_dynamic_channels=report['checked_dynamic_channels'],
                      full_preprocessing_verified=False,full_history_downloaded=False)
    try:
        persist('downloading_averaged_sample')
        for product,names in PRODUCTS.items():
            pieces=[]
            for date,granule in jobs[product]:
                url=access.opendap_url(granule['links'])
                destination=run/f'{product}_{date}.nc4'
                downloaded=download_subset(session,url,constraint(product,date),destination)
                downloads.append(dict(product=product,date=date,source_url=url,**downloaded))
                persist('validating_download')
                with xr.open_dataset(destination,engine='h5netcdf') as dataset:
                    validate_daily(dataset,date,names)
                    pieces.append(dataset.load())
            joined=xr.concat(pieces,dim='time')
            with xr.open_dataset(reference,engine='h5netcdf') as ref:
                comparison=compare_averaged(joined,ref,names)
            products[product]=dict(comparison=comparison)
            persist('comparing_averaged_sample')
        equal=all(p['comparison']['all_native_equal'] for p in products.values())
        units_verified=all(p['comparison']['all_units_verified'] for p in products.values())
        state=('averaged_sample_native_exact_match' if units_verified else
               'averaged_sample_numeric_match_units_unverified') if equal else 'averaged_sample_differs'
        persist(state)
        return 0 if equal else 2
    except Exception as exc:
        persist('failed')
        access.status('failed',error_type=type(exc).__name__,run_directory=str(run),
                      downloaded_requests=len(downloads),
                      checked_dynamic_channels=sum(len(PRODUCTS[p]) for p in products))
        return 1
    finally:
        session.close()


def run_cli():
    access.OUT=BASE/'averaged_sample_status.json'
    try:
        return main()
    except Exception as exc:
        access.OUT=BASE/'averaged_sample_status.json'
        access.status('failed',error_type=type(exc).__name__)
        return 1


if __name__=='__main__':
    sys.exit(run_cli())
