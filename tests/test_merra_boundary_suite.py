import json
from pathlib import Path
import pytest
import xarray as xr
from scripts import download_merra_boundary_suite as suite


def test_distinct_slices():
    first=suite.cache_key('M2T1NXLND','2020-01-01','2020-01-01')
    previous=suite.cache_key('M2T1NXLND','2020-01-01','2020-01-02')
    assert first!=previous


def test_cache_checks_original_hash_and_bytes(tmp_path,monkeypatch):
    path=tmp_path/'sample.nc4';path.write_bytes(b'original')
    record={'path':str(path),'source_url':'official','bytes':8,
            'sha256':suite.file_hash(path)}
    monkeypatch.setattr(suite,'ROOT',tmp_path)
    monkeypatch.setattr(suite,'read_piece',lambda *args: 'validated')
    assert suite.load_cached(record,'official','2020-01-01','2020-01-01',[])=='validated'
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='digest'):
        suite.load_cached(record,'official','2020-01-01','2020-01-01',[])
    with pytest.raises(ValueError,match='URL'):
        suite.load_cached(record,'different','2020-01-01','2020-01-01',[])


def test_cache_rejects_outside_workspace(tmp_path,monkeypatch):
    monkeypatch.setattr(suite,'ROOT',tmp_path/'workspace')
    with pytest.raises(ValueError,match='workspace'):
        suite.load_cached({'path':str(tmp_path/'outside')},'url','a','a',[])


def test_plan_has_six_new_requests():
    jobs=suite.job_specs()
    cached={suite.cache_key('M2T1NXLND',d,'2020-01-01') for d in ('2019-12-31','2020-01-01')}
    assert len(jobs)==8
    assert sum(suite.cache_key(p,d,r) not in cached for r,p,d in jobs)==6


@pytest.mark.parametrize('fail_on,attempted,successful',[(None,6,6),(3,3,2)])
def test_single_login_and_partial_failure_counts(tmp_path,monkeypatch,fail_on,attempted,successful):
    calls={'login':0,'download':0}
    class Session:
        def close(self): pass
    class Auth:
        authenticated=True
        def get_session(self): return Session()
    def login(*args):
        calls['login']+=1
        return Auth()
    monkeypatch.setattr(suite,'BASE',tmp_path)
    monkeypatch.setattr(suite,'credential_form',lambda:('fake','fake'))
    monkeypatch.setattr(suite,'login_from_form',login)
    jobs={suite.cache_key(p,d,r):'url' for r,p,d in suite.job_specs()}
    cache={suite.cache_key('M2T1NXLND',d,'2020-01-01'):{'path':'cache'}
           for d in ('2019-12-31','2020-01-01')}
    refs={d:{'path':'reference'} for d in ('2020-01-01','2020-01-02')}
    monkeypatch.setattr(suite,'prepare',lambda: (refs,jobs,cache))
    piece=xr.Dataset({name:('time',[1.]) for names in suite.averaged.PRODUCTS.values()
                      for name in names},coords={'time':[0]})
    for name in piece: piece[name].attrs['units']='test'
    monkeypatch.setattr(suite,'load_cached',lambda *args:piece.copy())
    monkeypatch.setattr(suite,'read_piece',lambda *args:piece.copy())
    monkeypatch.setattr(suite.xr,'open_dataset',lambda *args,**kwargs:piece.copy())
    monkeypatch.setattr(suite.averaged,'compare_averaged',lambda *args,**kwargs:{'test':True})
    def download(session,url,ce,path,max_bytes):
        assert max_bytes==32*1024*1024
        calls['download']+=1
        if calls['download']==fail_on: raise ValueError('test')
        return {'bytes':1,'sha256':'fake'}
    monkeypatch.setattr(suite,'download_subset',download)
    suite.main(client=object())
    state=json.loads((tmp_path/'boundary_suite_status.json').read_text())
    assert calls['login']==1
    assert state['attempted_download_requests']==attempted
    assert state['successful_download_requests']==successful
    assert state['reused_cached_requests']==2
    if fail_on is None:
        assert state['checked_channels_by_date']=={'2020-01-01':10,'2020-01-02':2}


def test_cross_day_units_required():
    a=xr.Dataset({'LAI':('time',[1.])})
    b=a.copy(deep=True)
    a.LAI.attrs['units']='1'
    with pytest.raises(ValueError,match='units'):
        suite.validate_units([a,b],['LAI'])
    b.LAI.attrs['units']='m'
    with pytest.raises(ValueError,match='units'):
        suite.validate_units([a,b],['LAI'])


def test_atomic_publication_uses_unique_temporary_files(tmp_path,monkeypatch):
    target=tmp_path/'status.json'
    sources=[]
    real_replace=suite.os.replace
    def replace(source,destination):
        assert source.parent==destination.parent
        assert json.loads(source.read_text('utf-8'))['value'] in (1,2)
        sources.append(source)
        real_replace(source,destination)
    monkeypatch.setattr(suite.os,'replace',replace)
    suite.atomic_json(target,{'value':1})
    suite.atomic_json(target,{'value':2})
    assert len(set(sources))==2
    assert all(not p.exists() for p in sources)
    assert json.loads(target.read_text('utf-8'))=={'value':2}


def test_atomic_failure_preserves_previous_and_unrelated_temp(tmp_path,monkeypatch):
    target=tmp_path/'status.json'
    suite.atomic_json(target,{'value':1})
    other=tmp_path/'status.json.other.tmp'
    other.write_text('other','utf-8')
    def fail(*args): raise OSError('test')
    monkeypatch.setattr(suite.os,'replace',fail)
    with pytest.raises(OSError):
        suite.atomic_json(target,{'value':2})
    assert json.loads(target.read_text('utf-8'))=={'value':1}
    assert list(tmp_path.glob('*.tmp'))==[other]
