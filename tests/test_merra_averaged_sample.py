import numpy as np
import pytest
import xarray as xr
import json
from types import SimpleNamespace

from scripts import download_merra_averaged_sample as sample


def datasets(dtype='float32'):
    times=sample.source_times()
    values=np.arange(4,dtype=dtype)[:,None,None]*np.ones((4,30,32),dtype=dtype)
    raw=xr.Dataset({'LAI':(('time','lat','lon'),values,{'units':'1'})},
                   coords={'time':times,'lat':sample.LAT,'lon':sample.LON})
    expected=(values[[0,2]]+values[[1,3]])/2
    ref=xr.Dataset({'LAI':(('time','lat','lon'),expected,{'units':'1'})},
                   coords={'time':sample.TIMES,'lat':sample.LAT,'lon':sample.LON})
    return raw,ref


def test_times_cross_previous_day_and_constraint():
    assert str(sample.source_times()[0])=='2019-12-31T23:30'
    assert str(sample.source_times()[-1])=='2020-01-01T06:30'
    assert '/LAI[23:1:23]' in sample.constraint('M2T1NXLND','2019-12-31')
    assert '/LAI[0:1:6]' in sample.constraint('M2T1NXLND','2020-01-01')
    with pytest.raises(ValueError):
        sample.constraint('M2T1NXLND','2020-01-02')


def test_ordinary_day_uses_its_own_dates_and_reference_times():
    times=sample.TIMES+np.timedelta64(1,'D')
    assert str(sample.source_times(times)[0])=='2020-01-01T23:30'
    assert '/LAI[23:1:23]' in sample.constraint('M2T1NXLND','2020-01-01','2020-01-02')
    assert '/LAI[0:1:6]' in sample.constraint('M2T1NXLND','2020-01-02','2020-01-02')
    raw,ref=datasets()
    raw=raw.assign_coords(time=raw.time+np.timedelta64(1,'D'))
    ref=ref.assign_coords(time=ref.time+np.timedelta64(1,'D'))
    sample.validate_daily(raw.isel(time=[0]),'2020-01-01',['LAI'],'2020-01-02')
    assert sample.compare_averaged(raw,ref,['LAI'],times)['all_native_equal']
    with pytest.raises(ValueError):
        sample.compare_averaged(raw,ref,['LAI'])


def test_exact_average():
    raw,ref=datasets()
    result=sample.compare_averaged(raw,ref,['LAI'])
    assert result['all_native_equal']
    assert result['variables']['LAI']['cells']==1920


@pytest.mark.parametrize('failure',['time','unit','nan'])
def test_invalid_inputs_rejected(failure):
    raw,ref=datasets()
    if failure=='time':
        raw=raw.assign_coords(time=raw.time+np.timedelta64(1,'h'))
    elif failure=='unit':
        raw.LAI.attrs['units']='kg'
    else:
        raw.LAI.values[0,0,0]=np.nan
    with pytest.raises(ValueError):
        sample.compare_averaged(raw,ref,['LAI'])


def test_native_float32_and_float64_mean_distinguished():
    raw,ref=datasets()
    raw.LAI.values[0]=np.float32(1)
    raw.LAI.values[1]=np.nextafter(np.float32(1),np.float32(2))
    ref.LAI.values[0]=np.float32(1)
    result=sample.compare_averaged(raw,ref,['LAI'])['variables']['LAI']
    assert result['native_precision']['equal']
    assert not result['float64_precision']['equal']
    assert result['float64_precision']['max_abs_difference']>0


def test_missing_reference_units_allow_diagnostics_not_unit_verification():
    raw,ref=datasets()
    ref.LAI.attrs.pop('units')
    result=sample.compare_averaged(raw,ref,['LAI'])
    assert result['all_native_equal']
    assert result['all_units_verified'] is False
    assert result['variables']['LAI']['unit_check']=='reference_units_missing'


def test_midnight_boundary_candidate_does_not_replace_centered_hypothesis():
    raw,ref=datasets()
    ref.LAI.values[0]=raw.LAI.values[1]
    result=sample.compare_averaged(raw,ref,['LAI'])
    assert result['all_native_equal'] is False
    assert result['all_daily_boundary_candidate_equal'] is True
    row=result['variables']['LAI']
    assert row['native_precision']['equal_cells_by_time']==[0,960]
    assert row['daily_boundary_candidate']['equal_cells_by_time']==[960,960]


def test_partial_report_counts_only_completed_comparisons():
    report=sample.comparison_report({'M2T1NXLND':{}},downloaded_requests=3)
    assert report['checked_dynamic_channels']==2
    assert report['downloaded_requests']==3
    assert not report['sample_comparison_complete']
    assert not report['full_preprocessing_verified']
    assert report['total_dynamic_channels']==160


def test_cli_exception_nonzero_and_sanitized(monkeypatch):
    saved=[]
    monkeypatch.setattr(sample,'main',lambda: (_ for _ in ()).throw(ValueError('secret')))
    monkeypatch.setattr(sample.access,'status',lambda state,**kw:saved.append((state,kw)))
    assert sample.run_cli()==1
    assert 'secret' not in str(saved)


def test_daily_coordinates_reject_mislabelled_hour():
    raw,_=datasets()
    previous=raw.isel(time=[0])
    sample.validate_daily(previous,'2019-12-31',['LAI'])
    with pytest.raises(ValueError):
        sample.validate_daily(previous.assign_coords(time=previous.time+np.timedelta64(1,'h')),
                              '2019-12-31',['LAI'])


def test_interrupted_download_preserves_actual_counts(tmp_path,monkeypatch):
    base=tmp_path/'full_history'
    (base/'catalog').mkdir(parents=True)
    (tmp_path/'pilot_batch_report.json').write_text(json.dumps({'rows':[
        {'source_sha256':{'merra-2/MERRA2_sfc_20200101.nc':'pinned'}}]}))
    for product in sample.PRODUCTS:
        (base/'catalog'/f'{product}.json').write_text('{}')
    monkeypatch.setattr(sample,'HERE',tmp_path)
    monkeypatch.setattr(sample,'BASE',base)
    monkeypatch.setattr(sample,'file_hash',lambda _: 'pinned')
    monkeypatch.setattr(sample,'exact_granule',lambda *_: {'links':[{
        'href':'https://opendap.earthdata.nasa.gov/collections/test/granules/sample'}]})
    monkeypatch.setattr(sample,'credential_form',lambda: ('test','not-real'))
    closed=[]
    monkeypatch.setattr(sample,'login_from_form',lambda *_: SimpleNamespace(
        authenticated=True,get_session=lambda:SimpleNamespace(close=lambda:closed.append(True))))
    calls=[]
    def download(*_):
        calls.append(True)
        if len(calls)==4:
            raise ValueError('private service response must not be logged')
        return {'bytes':10,'sha256':'fake'}
    monkeypatch.setattr(sample,'download_subset',download)
    monkeypatch.setattr(sample.xr,'open_dataset',lambda *_args,**_kw: xr.Dataset())
    monkeypatch.setattr(sample,'validate_daily',lambda *_: None)
    monkeypatch.setattr(sample,'compare_averaged',lambda *_: {'all_native_equal':True})
    saved=[]
    monkeypatch.setattr(sample.access,'status',lambda state,**kw:saved.append((state,kw)))
    assert sample.run_cli()==1
    assert saved[-1][1]['downloaded_requests']==3
    assert saved[-1][1]['checked_dynamic_channels']==2
    assert 'private service' not in str(saved)
    assert closed==[True]
