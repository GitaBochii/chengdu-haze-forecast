"""Bounded, single-session comparison of year-start and ordinary-day averages."""
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import download_merra_averaged_sample as averaged
from scripts.download_merra_regional_sample import BASE,HERE,file_hash,download_subset
from scripts.download_satellite_sample import credential_form,login_from_form

LEGACY_RUN='20260926T085628139252Z'


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(value,stream,indent=2,allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def job_specs():
    return [('2020-01-01',p,d) for p in averaged.PRODUCTS
            for d in ('2019-12-31','2020-01-01')]+[
                ('2020-01-02','M2T1NXLND',d) for d in ('2020-01-01','2020-01-02')]


def cache_key(product,date,reference_date):
    return (product,date,averaged.constraint(product,date,reference_date=reference_date))


def read_piece(path,date,reference_date,names):
    with xr.open_dataset(path,engine='h5netcdf') as ds:
        averaged.validate_daily(ds,date,names,reference_date=reference_date)
        return ds.load()


def load_cached(record,url,date,reference_date,names):
    path=Path(record['path']).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Cache path outside workspace')
    if record['source_url']!=url:
        raise ValueError('Cache URL mismatch')
    if path.stat().st_size!=record['bytes'] or file_hash(path)!=record['sha256']:
        raise ValueError('Cache original digest or byte count mismatch')
    return read_piece(path,date,reference_date,names)


def prepare():
    batch=json.loads((HERE/'pilot_batch_report.json').read_text('utf-8'))
    refs={}
    for day in ('2020-01-01','2020-01-02'):
        relative='merra-2/MERRA2_sfc_'+day.replace('-','')+'.nc'
        path=HERE/'real_sample'/relative
        pins={r['source_sha256'][relative] for r in batch['rows']
              if relative in r.get('source_sha256',{})}
        digest=file_hash(path)
        if pins!={digest}: raise ValueError('Frozen reference hash mismatch')
        refs[day]={'path':str(path),'sha256':digest}
    jobs={}
    for reference_date,product,date in job_specs():
        key=cache_key(product,date,reference_date)
        catalog=json.loads((BASE/'catalog'/f'{product}.json').read_text('utf-8'))
        granule=averaged.exact_granule(catalog,date)
        jobs[key]=averaged.access.opendap_url(granule['links'])
    legacy=BASE/'averaged_samples'/LEGACY_RUN
    report=json.loads((legacy/'comparison.json').read_text('utf-8'))
    cache={}
    # This fixed legacy run predates constraint logging; infer only its known
    # Jan-1 slices and validate their actual coordinates before accepting them.
    for date in ('2019-12-31','2020-01-01'):
        matches=[r for r in report['downloads']
                 if r['product']=='M2T1NXLND' and r['date']==date]
        if len(matches)!=1: raise ValueError('Missing unique legacy provenance')
        record=dict(matches[0],path=str(legacy/f'M2T1NXLND_{date}.nc4'),
                    provenance_report=str(legacy/'comparison.json'),
                    provenance_report_sha256=file_hash(legacy/'comparison.json'))
        key=cache_key('M2T1NXLND',date,'2020-01-01')
        load_cached(record,jobs[key],date,'2020-01-01',averaged.PRODUCTS['M2T1NXLND'])
        cache[key]=record
    return refs,jobs,cache


def validate_units(pieces,names):
    for name in names:
        values=[piece[name].attrs.get('units') for piece in pieces]
        values=[v.decode() if isinstance(v,bytes) else v for v in values]
        if not all(values) or len(set(values))!=1:
            raise ValueError('Cross-day units missing or inconsistent: '+name)


def execute(session,run,refs,jobs,cache,report,persist):
    groups={}
    for reference_date,product,date in job_specs():
        key=cache_key(product,date,reference_date)
        url=jobs[key]
        names=averaged.PRODUCTS[product]
        if key in cache:
            record=cache[key]
            piece=load_cached(record,url,date,reference_date,names)
            report['reused_cached_requests']+=1
            report['downloads'].append(dict(record,constraint=key[2],reused=True,
                                            cache_path=record['path'],reference_date=reference_date))
            persist('cache_verified')
        else:
            path=run/f'{product}_{date}_for_{reference_date}.nc4'
            report['attempted_download_requests']+=1
            persist('downloading')
            downloaded=download_subset(session,url,key[2],path,max_bytes=32*1024*1024)
            report['successful_download_requests']+=1
            report['downloads'].append(dict(product=product,date=date,source_url=url,
                constraint=key[2],path=str(path),reference_date=reference_date,reused=False,**downloaded))
            persist('validating_download')
            piece=read_piece(path,date,reference_date,names)
        group=groups.setdefault((reference_date,product),[])
        group.append(piece)
        if len(group)==2:
            validate_units(group,names)
            times=np.array([reference_date+'T00',reference_date+'T06'],dtype='datetime64[h]')
            with xr.open_dataset(refs[reference_date]['path'],engine='h5netcdf') as ref:
                comparison=averaged.compare_averaged(xr.concat(group,dim='time'),ref,names,times=times)
            report['comparisons'].append(dict(reference_date=reference_date,product=product,
                source_times_utc=[str(t)+'Z' for t in averaged.source_times(times=times)],
                comparison=comparison))
            report['checked_channels_by_date']={day:sum(len(averaged.PRODUCTS[c['product']])
                for c in report['comparisons'] if c['reference_date']==day)
                for day in ('2020-01-01','2020-01-02')}
            persist('compared')


def main(client=None):
    run=BASE/'boundary_suites'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True,exist_ok=False)
    report=dict(run_directory=str(run),process_id=os.getpid(),attempted_download_requests=0,
        successful_download_requests=0,expected_new_download_requests=6,reused_cached_requests=0,
        downloads=[],comparisons=[],checked_channels_by_date={},
        full_preprocessing_verified=False,full_history_downloaded=False,
        diagnostic_scope='Jan 1 ten channels; Jan 2 two land channels. Neither hypothesis establishes an all-history boundary rule.',
        deployment_boundary='Offline reanalysis; centered averages can include support after prediction origin.')
    def persist(state):
        report.update(status=state,updated_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(run/'comparison.json',report)
        atomic_json(BASE/'boundary_suite_status.json',report)
    session=None
    try:
        persist('preparing')
        refs,jobs,cache=prepare()
        report['references']=refs
        persist('waiting_for_local_login')
        credentials=credential_form()
        if credentials is None:
            persist('cancelled_by_user');return 1
        if client is None:
            import earthaccess
            client=earthaccess
        persist('authenticating')
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            try:
                auth=login_from_form(client,*credentials)
            finally:
                del credentials
        if not auth.authenticated:
            persist('authentication_failed');return 1
        session=auth.get_session()
        execute(session,run,refs,jobs,cache,report,persist)
        persist('bounded_comparison_complete')
        return 0
    except Exception as exc:
        report['error_type']=type(exc).__name__
        persist('failed')
        return 1
    finally:
        if session is not None: session.close()


if __name__=='__main__':
    sys.exit(main())
