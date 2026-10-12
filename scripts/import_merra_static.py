"""Import official regional constants using an existing Earthdata session."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr

from scripts.download_merra_regional_sample import LAT, LON, HERE, download_subset, file_hash

UNITS={'FRACI':'1','FRLAND':'1','FROCEAN':'1','PHIS':'m+2 s-2'}
PRODUCTS={
    'M2C0NXCTM':{'collection':'C1278787764-GES_DISC','stem':'ctm','times':12,
                  'variables':list(UNITS)},
    'M2C0NXASM':{'collection':'C1276812819-GES_DISC','stem':'asm','times':1,
                  'variables':['FRLAND','FROCEAN','PHIS']},
}


def source_url(product):
    spec=PRODUCTS[product]
    return ('https://opendap.earthdata.nasa.gov/collections/'+spec['collection']+
            '/granules/'+product+'.5.12.4%3AMERRA2_101.const_2d_'+spec['stem']+
            '_Nx.00000000.nc4')


def constraint(product):
    spec=PRODUCTS[product]; time=f"[0:1:{spec['times']-1}]"
    parts=['/time'+time,'/lat[226:1:255]','/lon[438:1:469]']
    parts.extend('/'+name+time+'[226:1:255][438:1:469]' for name in spec['variables'])
    return ';'.join(parts)


def validate(ds, product):
    spec=PRODUCTS[product]
    for name, expected in [('lat',LAT),('lon',LON)]:
        if name not in ds.coords or not np.array_equal(ds[name].values,expected):
            raise ValueError('Invalid static coordinate: '+name)
    if ds.sizes.get('time')!=spec['times'] or 'time' not in ds.coords:
        raise ValueError('Invalid static time dimension')
    if len(np.unique(ds.time.values))!=spec['times']:
        raise ValueError('Duplicate static time coordinates')
    variables={}
    for name in spec['variables']:
        if name not in ds or ds[name].dims!=('time','lat','lon'):
            raise ValueError('Invalid static dimensions: '+name)
        unit=ds[name].attrs.get('units')
        if isinstance(unit,bytes): unit=unit.decode()
        if unit!=UNITS[name]: raise ValueError('Invalid static units: '+name)
        values=ds[name].values
        if not np.isfinite(values).all() or np.any(np.abs(values)>=1e14):
            raise ValueError('Missing static values: '+name)
        if name!='PHIS' and (values.min()<0 or values.max()>1):
            raise ValueError('Invalid static fraction: '+name)
        equal=all(np.array_equal(values[0],a) for a in values)
        variables[name]={'unit':unit,'all_months_equal':equal,'cells':int(values.size)}
    return {'variables':variables,'all_months_equal':all(v['all_months_equal'] for v in variables.values()),
            'time_coordinates':[str(t) for t in ds.time.values]}


def compare_reference(ds, reference):
    ref=reference.sel(lat=LAT,lon=LON)
    results={}
    for name in UNITS:
        unit=ref[name].attrs.get('units')
        if isinstance(unit,bytes): unit=unit.decode()
        if unit!=UNITS[name]: raise ValueError('Reference unit mismatch: '+name)
        a=ds[name].values; b=ref[name].transpose('lat','lon').values
        if not np.isfinite(b).all() or np.any(np.abs(b)>=1e14):
            raise ValueError('Invalid reference values: '+name)
        delta=np.abs(a.astype('float64')-b.astype('float64'))
        results[name]={'equal':bool(np.all(a==b)), 'max_abs_difference':float(delta.max()),
                       'compared_cells':int(a.size)}
    return {'all_equal':all(v['equal'] for v in results.values()),'variables':results}


def write_report(path, report):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                     prefix='.static-',suffix='.tmp',delete=False) as stream:
        temporary=stream.name
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def run_import(session, output):
    """Download/validate two small subsets; do not login or close caller session."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    report={'started_at_utc':datetime.now(timezone.utc).isoformat(),'status':'importing',
            'products':{},'full_history_applicability_verified':False,
            'scope':'Official model constants, not date-specific observations; CMR nominal end 2020-12-31.'}
    report_path=output/'report.json'
    previous=json.loads(report_path.read_text('utf-8')) if report_path.exists() else {}
    pieces={}
    try:
        for product in PRODUCTS:
            target=output/(product+'.nc4'); url=source_url(product); ce=constraint(product)
            cached=previous.get('products',{}).get(product,{})
            reuse=(target.exists() and cached.get('source_url')==url and cached.get('constraint')==ce
                   and cached.get('sha256')==file_hash(target)
                   and cached.get('bytes')==target.stat().st_size)
            download=({'bytes':target.stat().st_size,'sha256':file_hash(target)} if reuse else
                      download_subset(session,url,ce,target,max_bytes=4*1024*1024))
            with xr.open_dataset(target,engine='h5netcdf') as ds:
                validation=validate(ds,product); pieces[product]=ds.load()
            report['products'][product]={**download,'source_url':url,'constraint':ce,
                                        'reused':reuse,'validation':validation}
            write_report(report_path,report)
        reference=HERE/'real_sample/merra-2/MERRA2_sfc_20200101.nc'
        digest=file_hash(reference)
        pinned=json.loads((HERE/'pilot_batch_report.json').read_text('utf-8'))
        hashes={r['source_sha256']['merra-2/MERRA2_sfc_20200101.nc'] for r in pinned['rows']
                if 'merra-2/MERRA2_sfc_20200101.nc' in r.get('source_sha256',{})}
        if hashes!={digest}: raise ValueError('Frozen static reference hash mismatch')
        with xr.open_dataset(reference,engine='h5netcdf') as ref:
            report['reference_comparison']=compare_reference(pieces['M2C0NXCTM'],ref)
        report['reference_sha256']=digest
        report['asm_ctm_equal']={name:bool(np.all(pieces['M2C0NXCTM'][name].values==
                                    pieces['M2C0NXASM'][name].values))
                                for name in PRODUCTS['M2C0NXASM']['variables']}
        consistent=(all(report['asm_ctm_equal'].values()) and
                    report['products']['M2C0NXCTM']['validation']['all_months_equal'])
        report['status']='imported_constants_consistent' if consistent else 'imported_requires_review'
        report['ready_as_frozen_model_constants']=consistent and report['reference_comparison']['all_equal']
        report['data_downloaded']=True
        write_report(report_path,report)
        return report
    except Exception as exc:
        report['status']='failed'; report['error_type']=type(exc).__name__
        write_report(report_path,report)
        raise
