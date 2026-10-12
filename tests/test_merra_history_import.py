import json
import numpy as np
import pytest
import xarray as xr
from scripts import import_merra_history as m


def records():
    return [dict(weather_time_utc='2020-01-01T00:00:00Z',
                 source_left_utc='2019-12-31T23:30:00Z',
                 source_right_utc='2020-01-01T00:30:00Z')]


def test_jobs_preserve_midnight_raw_support():
    jobs=m.time_jobs(records())
    assert len(jobs)==8
    assert next(j for j in jobs if j['product']=='M2I3NVASM')['indices']==[0]
    assert next(j for j in jobs if j['product']=='M2T1NXLND' and j['date']=='2019-12-31')['indices']==[23]
    assert next(j for j in jobs if j['product']=='M2T1NXLND' and j['date']=='2020-01-01')['indices']==[0]


def test_invalid_source_cadence_rejected():
    rows=records();rows[0]['source_left_utc']='2019-12-31T23:00:00Z'
    with pytest.raises(ValueError):m.time_jobs(rows)


def test_span_is_explicit_and_constraints():
    job=dict(product='M2I1NXASM',date='2020-01-01',indices=[0,6,18])
    assert m.returned_indices(job)==list(range(0,19,6))
    assert '/T2M[0:6:18][226:1:255][438:1:469]' in m.constraint(job)


def test_validation_and_resume(tmp_path):
    job=dict(product='M2T1NXLND',date='2020-01-01',indices=[0])
    ds=xr.Dataset({name:(('time','lat','lon'),np.ones((1,30,32)),{'units':'1'})
                   for name in m.PRODUCTS[job['product']]},
                  coords={'time':[np.datetime64('2020-01-01T00:30')],'lat':m.LAT,'lon':m.LON})
    path=tmp_path/'data.nc4';ds.to_netcdf(path,engine='h5netcdf')
    m.validate(path,job)
    record=dict(bytes=path.stat().st_size,sha256=m.file_hash(path),job=job)
    assert m.reusable(path,job,record)
    assert not m.reusable(path,{**job,'indices':[1]},record)
    path.write_bytes(b'bad')
    assert not m.reusable(path,job,record)


def test_atomic_json_and_failed_download(tmp_path):
    m.atomic_json(tmp_path/'manifest.json',{'status':'incomplete'})
    assert json.loads((tmp_path/'manifest.json').read_text())['status']=='incomplete'
    assert not list(tmp_path.glob('*.tmp'))


def test_failure_is_checkpointed_not_completed(tmp_path, monkeypatch):
    job=dict(product='M2T1NXLND',date='2020-01-01',indices=[0],
             source_url='https://example.invalid',constraint='test')
    manifest={'files':{},'failures':{}}
    def fail(*args):
        raise ValueError('do not persist server text')
    monkeypatch.setattr(m,'download_subset',fail)
    with pytest.raises(ValueError):m.run_import(None,[job],manifest,tmp_path)
    saved=json.loads((tmp_path/'manifest.json').read_text())
    assert saved['status']=='failed' and not saved['raw_dynamic_complete']
    assert saved['files']=={}
    assert 'do not persist' not in json.dumps(saved)


def test_resume_verifies_and_skips_network(tmp_path, monkeypatch):
    job=dict(product='M2T1NXLND',date='2020-01-01',indices=[0],
             source_url='https://example.invalid',constraint='test')
    manifest={'files':{},'failures':{}}
    calls=[]
    def download(session,url,ce,path):
        calls.append(path)
        ds=xr.Dataset({n:(('time','lat','lon'),np.ones((1,30,32)),{'units':'1'})
                       for n in m.PRODUCTS[job['product']]},coords={
                       'time':[np.datetime64('2020-01-01T00:30')],'lat':m.LAT,'lon':m.LON})
        ds.to_netcdf(path,engine='h5netcdf')
        return dict(bytes=path.stat().st_size,sha256=m.file_hash(path))
    monkeypatch.setattr(m,'download_subset',download)
    m.run_import(None,[job],manifest,tmp_path)
    m.run_import(None,[job],manifest,tmp_path)
    assert len(calls)==1 and manifest['raw_dynamic_complete']


def test_limit_never_marks_all_history_complete(tmp_path, monkeypatch):
    job=dict(product='M2T1NXLND',date='2020-01-01',indices=[0],
             source_url='https://example.invalid',constraint='test')
    manifest={'files':{},'failures':{}}
    m.run_import(None,[job],manifest,tmp_path,limit=0)
    assert manifest['status']=='batch_limit_reached'
    assert not manifest['raw_dynamic_complete']


def test_expected_units_reject_wrong_nonempty_unit(tmp_path):
    job=dict(product='M2T1NXLND',date='2020-01-01',indices=[0])
    ds=xr.Dataset({n:(('time','lat','lon'),np.ones((1,30,32)),{'units':'K'})
                  for n in m.PRODUCTS[job['product']]},coords={
                  'time':[np.datetime64('2020-01-01T00:30')],'lat':m.LAT,'lon':m.LON})
    p=tmp_path/'bad.nc4';ds.to_netcdf(p,engine='h5netcdf')
    with pytest.raises(ValueError):m.validate(p,job)


def test_retry_recovers_transient_error(monkeypatch):
    calls=[]
    def op():
        calls.append(1)
        if len(calls)<3:raise OSError('transient')
        return 'ok'
    assert m.retry(op,sleep=lambda seconds:None)=='ok'
    assert len(calls)==3


def test_authorization_failure_is_not_retried():
    class Response:
        status_code=403
        def close(self):pass
    class Session:
        calls=0
        def get(self,*args,**kwargs):
            self.calls+=1
            return Response()
    session=Session()
    with pytest.raises(m.requests.HTTPError):
        m.retry(lambda:m.CheckedSession(session).get('ignored'),sleep=lambda x:None)
    assert session.calls==1


def test_same_session_imports_constants_before_dynamic(tmp_path,monkeypatch):
    from scripts import import_merra_static
    calls=[]
    session=object()
    def static(s,out):
        assert s is session
        calls.append('static')
        return {'ready_as_frozen_model_constants':True}
    def dynamic(s,*args,**kwargs):
        assert s is session
        calls.append('dynamic')
    monkeypatch.setattr(import_merra_static,'run_import',static)
    monkeypatch.setattr(m,'run_import',dynamic)
    monkeypatch.setattr(m,'OUT',tmp_path)
    m.run_authenticated(session,[],{'files':{}},5)
    assert calls==['static','dynamic']
