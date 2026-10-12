import pytest
import io
import json
import requests
from scripts import verify_merra_access
from scripts.verify_merra_access import opendap_url


def test_uses_only_official_opendap_service():
    assert opendap_url([{'href':'https://opendap.earthdata.nasa.gov/collections/a/granules/b'}]).endswith('/b')
    with pytest.raises(ValueError):
        opendap_url([{'href':'https://example.com/steal'}])
    with pytest.raises(ValueError):
        opendap_url([{'href':'https://opendap.earthdata.nasa.gov.evil.test/collections/a'}])


def error_response(body, code=403, headers=None):
    response=requests.Response()
    response.status_code=code
    response.raw=io.BytesIO(body)
    response.headers.update(headers or {})
    return response


def test_error_diagnostics_never_preserve_body_or_redirect_secrets():
    response=error_response(b'Application not authorized; password=secret-body',headers={
        'Content-Type':'text/html; private=secret-header',
        'Location':'https://urs.earthdata.nasa.gov/oauth/authorize?token=secret-url',
        'Set-Cookie':'secret-cookie'})
    result=verify_merra_access.response_diagnostics(response)
    assert result['http_status']==403
    assert result['authorization_message_detected'] is True
    assert result['content_kind']=='html'
    assert result['redirect_destination']=='earthdata_login'
    assert 'secret' not in json.dumps(result)


def test_unknown_403_does_not_claim_missing_application_authorization():
    result=verify_merra_access.response_diagnostics(error_response(b'Forbidden'))
    assert result['authorization_message_detected'] is False
    assert result['redirect_destination']=='none'


def test_error_diagnostics_read_is_bounded():
    response=error_response(b'x'*100000)
    result=verify_merra_access.response_diagnostics(response)
    assert response.raw.tell()<=16384
    assert result['sample_bytes']==16384


def test_dap_error_classification_keeps_only_fixed_markers():
    response=error_response(b'Error { code = 1002; message = "BES upstream AccessDenied https://example.invalid/?secret=hidden"; };')
    result=verify_merra_access.response_diagnostics(response)
    assert result['dap_error_code']==1002
    assert result['error_markers']==['access_denied','backend']
    assert 'hidden' not in json.dumps(result)


@pytest.mark.parametrize('body,protocol,expected',[
    (b'Dataset { Float32 T[time = 8]; } sample;', 'dds', True),
    (b'<html>Dataset Float32</html>', 'dds', False),
    (b'<Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#"><Float32 name="T"/></Dataset>', 'dmr', True),
    (b'<Dataset><Float32 name="T"/></Dataset>', 'dmr', False),
    (b'Error { code = 403; };', 'dds', False),
])
def test_metadata_validation(body,protocol,expected):
    assert verify_merra_access.is_dataset_metadata(body,protocol) is expected


def test_probe_matrix_does_not_count_service_control_as_merra_success(tmp_path):
    class Session:
        def __init__(self):
            self.calls=[]

        def get(self,url,**kwargs):
            self.calls.append(url)
            assert kwargs['allow_redirects'] is False
            if '/hyrax/' in url:
                return error_response(b'Dataset { Float32 T[time = 8]; } control;',200)
            return error_response(b'Error { code = 403; message = "AccessDenied"; };')

    session=Session()
    result=verify_merra_access.probe_matrix(session,
        'https://opendap.earthdata.nasa.gov/collections/a/granules/b',tmp_path)
    assert len(session.calls)==3
    assert result['service_control_verified'] is True
    assert result['merra_metadata_verified'] is False
    assert not list(tmp_path.glob('M2I3NVASM*'))


def test_probe_matrix_saves_only_verified_merra_metadata(tmp_path):
    class Session:
        def get(self,url,**kwargs):
            if url.endswith('.dmr'):
                return error_response(b'<Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#"><Float32 name="T"/></Dataset>',200)
            return error_response(b'<html>login required</html>',200)

    result=verify_merra_access.probe_matrix(Session(),
        'https://opendap.earthdata.nasa.gov/collections/a/granules/b',tmp_path)
    assert result['merra_metadata_verified'] is True
    assert (tmp_path/'M2I3NVASM_20200101.dmr').exists()
    assert not (tmp_path/'M2I3NVASM_20200101.dds').exists()


def test_status_publishes_complete_json_atomically(tmp_path,monkeypatch):
    import os
    path=tmp_path/'status.json'
    path.write_text('{"status":"old"}')
    monkeypatch.setattr(verify_merra_access,'OUT',path)
    real_replace=os.replace
    replacements=[]

    def checked_replace(source,target):
        assert json.loads(path.read_text())['status']=='old'
        assert json.loads(source.read_text())['status']=='new'
        replacements.append((source,target))
        real_replace(source,target)

    monkeypatch.setattr(os,'replace',checked_replace)
    verify_merra_access.status('new',authenticated=True)
    assert len(replacements)==1
    saved=json.loads(path.read_text())
    assert saved['status']=='new'
    assert saved['process_id']==os.getpid()
    assert saved['updated_at_utc']
    assert list(tmp_path.iterdir())==[path]
