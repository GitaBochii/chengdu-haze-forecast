import numpy as np
import pytest
import xarray as xr
from scripts import import_merra_static as static


def fixture(product='M2C0NXCTM'):
    n=12 if product=='M2C0NXCTM' else 1
    ds=xr.Dataset(coords={'time':np.arange(n), 'lat':static.LAT,'lon':static.LON})
    for name in static.PRODUCTS[product]['variables']:
        ds[name]=(('time','lat','lon'),np.ones((n,30,32),dtype='float32'))
        ds[name].attrs['units']=static.UNITS[name]
    return ds


def test_valid_static_and_changed_month():
    ds=fixture()
    assert static.validate(ds,'M2C0NXCTM')['all_months_equal']
    ds.FRACI.values[3,0,0]=0.5
    assert not static.validate(ds,'M2C0NXCTM')['all_months_equal']


@pytest.mark.parametrize('bad',['unit','lat','fill','nan'])
def test_reject_invalid(bad):
    ds=fixture()
    if bad=='unit': ds.PHIS.attrs['units']='m'
    if bad=='lat': ds=ds.assign_coords(lat=ds.lat+1)
    if bad=='fill': ds.PHIS.values[0,0,0]=1e15
    if bad=='nan': ds.FRACI.values[0,0,0]=np.nan
    with pytest.raises(ValueError): static.validate(ds,'M2C0NXCTM')


def test_comparison_reports_difference_not_failure():
    ds=fixture(); ref=ds.isel(time=0,drop=True).copy(deep=True)
    ref.PHIS.values[0,0]=2
    result=static.compare_reference(ds,ref)
    assert not result['all_equal']
    assert result['variables']['PHIS']['max_abs_difference']==1


def test_constraint_keeps_all_ctm_months():
    assert '/FRACI[0:1:11][226:1:255][438:1:469]' in static.constraint('M2C0NXCTM')
    assert '/PHIS[0:1:0][226:1:255][438:1:469]' in static.constraint('M2C0NXASM')


def test_partial_download_failure_is_recorded(tmp_path,monkeypatch):
    import json
    calls=[]
    def download(session,url,ce,path,max_bytes):
        calls.append(url)
        if len(calls)==2: raise RuntimeError('must-not-leak-private-error')
        ds=fixture()
        ds.to_netcdf(path,engine='h5netcdf')
        return {'bytes':path.stat().st_size,'sha256':static.file_hash(path)}
    monkeypatch.setattr(static,'download_subset',download)
    with pytest.raises(RuntimeError): static.run_import(object(),tmp_path)
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['status']=='failed'
    assert set(report['products'])=={'M2C0NXCTM'}
    assert report['error_type']=='RuntimeError'
    assert 'must-not-leak' not in (tmp_path/'report.json').read_text()


def test_resume_rechecks_file_hash(tmp_path,monkeypatch):
    import json
    calls=[]
    def download(session,url,ce,path,max_bytes):
        calls.append(url)
        if 'M2C0NXASM' in url: raise RuntimeError('stop before reference')
        fixture().to_netcdf(path,engine='h5netcdf')
        return {'bytes':path.stat().st_size,'sha256':static.file_hash(path)}
    monkeypatch.setattr(static,'download_subset',download)
    with pytest.raises(RuntimeError): static.run_import(object(),tmp_path)
    with pytest.raises(RuntimeError): static.run_import(object(),tmp_path)
    assert len(calls)==3
    (tmp_path/'M2C0NXCTM.nc4').write_bytes(b'tampered')
    with pytest.raises(RuntimeError): static.run_import(object(),tmp_path)
    assert len(calls)==5
