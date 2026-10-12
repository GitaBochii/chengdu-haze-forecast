import io
import numpy as np
import pytest
import requests
import xarray as xr
from scripts import download_merra_regional_sample as sample


def datasets():
    coords=dict(time=np.array(['2020-01-01T00','2020-01-01T06'],dtype='datetime64[h]'),
                lev=np.arange(1,73),lat=np.arange(23,38,.5),
                lon=np.arange(93.75,113.126,.625))
    values=np.broadcast_to(np.arange(1,73)[None,:,None,None],(2,72,30,32)).astype('float32')
    ds=xr.Dataset({'T':(('time','lev','lat','lon'),values,{'units':'K'})},coords=coords)
    ref=ds.sel(lev=list(reversed(sample.LEVELS))).copy(deep=True)
    return ds,ref


def test_compares_native_levels_by_value_not_position():
    ds,ref=datasets()
    result=sample.compare_inputs(ds,ref,['T'],True)
    assert result['all_equal'] is True
    assert result['variables']['T']['max_abs_difference']==0
    assert result['variables']['T']['cells']==2*14*30*32


@pytest.mark.parametrize('failure',['time','unit','grid','missing'])
def test_rejects_incompatible_input(failure):
    ds,ref=datasets()
    if failure=='time': ds=ds.assign_coords(time=ds.time+np.timedelta64(1,'h'))
    if failure=='unit': ds.T.attrs['units']='C'
    if failure=='grid': ds=ds.assign_coords(lon=ds.lon+.625)
    if failure=='missing': ds.T.values[0,33,0,0]=np.nan
    with pytest.raises(ValueError):
        sample.compare_inputs(ds,ref,['T'],True)


def test_value_mismatch_is_reported_not_silently_corrected():
    ds,ref=datasets()
    ds.T.values[0,33,0,0]+=1
    result=sample.compare_inputs(ds,ref,['T'],True)
    assert result['all_equal'] is False
    assert result['variables']['T']['max_abs_difference']==1


def test_subset_constraints_use_distinct_hourly_and_three_hourly_indices():
    assert '/time[0:6:6]' in sample.constraint(False)
    assert '/time[0:2:2]' in sample.constraint(True)
    assert '/T[0:2:2][0:1:71][226:1:255][438:1:469]' in sample.constraint(True)


def test_response_rejects_html_and_oversize(tmp_path):
    class Session:
        def __init__(self,body): self.body=body
        def get(self,*args,**kwargs):
            r=requests.Response()
            r.status_code=200
            r.raw=io.BytesIO(self.body)
            return r
    target=tmp_path/'sample.nc4'
    url='https://opendap.earthdata.nasa.gov/collections/a/granules/b'
    with pytest.raises(ValueError):
        sample.download_subset(Session(b'<html>login</html>'),url,'/time',target)
    with pytest.raises(ValueError):
        sample.download_subset(Session(b'\x89HDF\r\n\x1a\n'+b'x'*100),url,'/time',target,max_bytes=32)
    assert not target.exists()


def test_partial_report_counts_only_completed_comparisons():
    report=sample.comparison_report({'M2I1NXASM':{'comparison':{'all_equal':True}}})
    assert report['checked_dynamic_channels']==10
    assert report['sample_comparison_complete'] is False
    report=sample.comparison_report({'M2I1NXASM':{},'M2I3NVASM':{}})
    assert report['checked_dynamic_channels']==150
    assert report['sample_comparison_complete'] is True
    assert report['full_preprocessing_verified'] is False


def test_cli_error_returns_nonzero_without_recording_exception_text(monkeypatch,tmp_path):
    def fail():
        raise ValueError('secret text')
    monkeypatch.setattr(sample,'main',fail)
    monkeypatch.setattr(sample,'BASE',tmp_path)
    assert sample.run_cli()==1
    text=(tmp_path/'regional_sample_status.json').read_text()
    assert 'ValueError' in text
    assert 'secret text' not in text
