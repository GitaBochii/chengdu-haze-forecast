"""Download a bounded real MERRA-2 regional sample, then compare frozen inputs."""
from pathlib import Path
from datetime import datetime,timezone
import contextlib
import hashlib
import io
import json
import sys

import numpy as np
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import verify_merra_access as access
from scripts.download_satellite_sample import credential_form,login_from_form

HERE=ROOT/'experiments/prithvi_wxc_probe_20260924'
BASE=HERE/'full_history'
LEVELS=[34,39,41,43,44,45,48,51,53,56,63,68,71,72]
SURFACE=['U10M','V10M','T2M','QV2M','PS','SLP','TS','TQI','TQL','TQV']
VERTICAL=['CLOUD','H','OMEGA','PL','QI','QL','QV','T','U','V']
TIMES=np.array(['2020-01-01T00','2020-01-01T06'],dtype='datetime64[h]')
LAT=np.arange(23,38,.5)
LON=np.arange(93.75,113.126,.625)


def constraint(vertical):
    time='[0:2:2]' if vertical else '[0:6:6]'
    grid='[226:1:255][438:1:469]'
    parts=['/time'+time,'/lat[226:1:255]','/lon[438:1:469]']
    if vertical:
        parts.append('/lev[0:1:71]')
    parts.extend('/'+name+time+('[0:1:71]' if vertical else '')+grid
                 for name in (VERTICAL if vertical else SURFACE))
    return ';'.join(parts)


def download_subset(session,url,ce,destination,max_bytes=32*1024*1024):
    url=access.opendap_url([{'href':url}])
    response=session.get(url+'.dap.nc4',params={'dap4.ce':ce},
                         timeout=(30,180),stream=True,allow_redirects=False)
    try:
        if response.status_code!=200:
            raise ValueError('Numeric subset request did not return HTTP 200')
        data=bytearray()
        for chunk in response.iter_content(65536):
            data.extend(chunk)
            if len(data)>max_bytes:
                raise ValueError('Numeric subset exceeded byte limit')
        if data[:8]!=b'\x89HDF\r\n\x1a\n':
            raise ValueError('Numeric subset is not an HDF5 NetCDF4 response')
        # Opening with the existing structured reader also rejects invalid HDF5.
        with xr.open_dataset(io.BytesIO(data),engine='h5netcdf') as ds:
            if not ds.data_vars:
                raise ValueError('Numeric subset has no data variables')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(data)
        return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    finally:
        response.close()


def compare_inputs(downloaded,reference,names,vertical):
    for coord,expected in [('time',TIMES),('lat',LAT),('lon',LON)]:
        if coord not in downloaded.coords or not np.array_equal(downloaded[coord].values,expected):
            raise ValueError('Unexpected subset coordinates: '+coord)
    if vertical and not np.array_equal(downloaded.lev.values,np.arange(1,73)):
        raise ValueError('Expected all 72 native levels, not pressure levels')
    selection=dict(time=TIMES,lat=LAT,lon=LON)
    if vertical:
        selection['lev']=LEVELS
    raw=downloaded.sel(**selection)
    ref=reference.sel(**selection)
    dims=('time','lev','lat','lon') if vertical else ('time','lat','lon')
    results={}
    for name in names:
        def unit(value):
            return value.decode() if isinstance(value,bytes) else value
        raw_unit=unit(raw[name].attrs.get('units'))
        if not raw_unit or raw_unit!=unit(ref[name].attrs.get('units')):
            raise ValueError('Unit mismatch: '+name)
        a,b=raw[name].transpose(*dims).values,ref[name].transpose(*dims).values
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Missing or incompatible numerical values: '+name)
        delta=np.abs(a.astype('float64')-b.astype('float64'))
        results[name]=dict(unit=raw_unit,cells=int(a.size),equal=bool(np.array_equal(a,b)),
                           max_abs_difference=float(delta.max()),
                           mean_abs_difference=float(delta.mean()))
    return dict(all_equal=all(v['equal'] for v in results.values()),variables=results)


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def comparison_report(products):
    counts={'M2I1NXASM':10,'M2I3NVASM':140}
    return dict(date='2020-01-01',times_utc=['00:00','06:00'],products=products,
                checked_dynamic_channels=sum(counts[p] for p in products),
                sample_comparison_complete=set(products)==set(counts),
                total_dynamic_channels=160,full_preprocessing_verified=False,
                full_history_downloaded=False)


def main():
    import earthaccess
    access.OUT=BASE/'regional_sample_status.json'
    access.status('preparing_reference')
    report=json.loads((HERE/'pilot_batch_report.json').read_text('utf-8'))
    jobs=[]
    for product,filename,vertical in [
        ('M2I1NXASM','MERRA2_sfc_20200101.nc',False),
        ('M2I3NVASM','MERRA_pres_20200101.nc',True)]:
        catalog=json.loads((BASE/'catalog'/f'{product}.json').read_text('utf-8'))
        entries=[g for g in catalog['granules'] if g['time_start'].startswith('2020-01-01T')]
        if len(entries)!=1:
            raise ValueError('Expected one exact sample granule')
        url=access.opendap_url(entries[0]['links'])
        relative='merra-2/'+filename
        pins={row['source_sha256'][relative] for row in report['rows']
              if relative in row.get('source_sha256',{})}
        reference=HERE/'real_sample'/relative
        digest=file_hash(reference)
        if pins!={digest}:
            raise ValueError('Frozen reference hash mismatch')
        jobs.append((product,url,reference,vertical,digest))
    access.status('waiting_for_local_login',purpose='regional_numeric_sample')
    credentials=credential_form()
    if credentials is None:
        access.status('cancelled_by_user');return
    access.status('authenticating')
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        try:
            auth=login_from_form(earthaccess,*credentials)
        finally:
            del credentials
    if not auth.authenticated:
        access.status('authentication_failed');return
    run=BASE/'regional_samples'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True,exist_ok=False)
    products={}
    session=auth.get_session()
    try:
        for product,url,reference,vertical,digest in jobs:
            access.status('downloading_regional_sample',product=product,run_directory=str(run))
            target=run/f'{product}.nc4'
            downloaded=download_subset(session,url,constraint(vertical),target)
            with xr.open_dataset(target,engine='h5netcdf') as ds, xr.open_dataset(reference,engine='h5netcdf') as ref:
                comparison=compare_inputs(ds,ref,VERTICAL if vertical else SURFACE,vertical)
            products[product]=dict(download=downloaded,source_url=url,
                reference_sha256=digest,comparison=comparison)
            (run/'comparison.json').write_text(json.dumps(
                comparison_report(products),indent=2,allow_nan=False),'utf-8')
    finally:
        session.close()
    equal=all(p['comparison']['all_equal'] for p in products.values())
    access.status('regional_sample_exact_match' if equal else 'regional_sample_differs',
                  data_downloaded=True,run_directory=str(run),checked_dynamic_channels=150,
                  full_preprocessing_verified=False,full_history_downloaded=False)


def run_cli():
    access.OUT=BASE/'regional_sample_status.json'
    try:
        main()
    except Exception as exc:
        access.status('failed',error_type=type(exc).__name__)
        return 1
    return 0


if __name__=='__main__':
    sys.exit(run_cli())
