"""User-entered, non-persistent NASA login and a bounded MERRA-2 metadata probe."""
from pathlib import Path
from urllib.parse import urlparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.download_satellite_sample import credential_form,login_from_form

OUT=ROOT/'experiments/prithvi_wxc_probe_20260924/full_history/access_check.json'


def opendap_url(links):
    for item in links:
        url=item.get('href','')
        parsed=urlparse(url)
        if (parsed.scheme=='https' and parsed.hostname=='opendap.earthdata.nasa.gov'
                and parsed.path.startswith('/collections/') and not parsed.username
                and not parsed.query):
            return url
    raise ValueError('No approved NASA OPeNDAP link in catalog')


def status(state,**values):
    OUT.parent.mkdir(parents=True,exist_ok=True)
    record=dict(status=state,**values,process_id=os.getpid(),
                updated_at_utc=datetime.now(timezone.utc).isoformat())
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',
                dir=OUT.parent,prefix='.access_check_',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(record,stream,ensure_ascii=False,indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,OUT)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def response_diagnostics(response):
    # Persist only fixed categories and counts, never server text or auth URLs.
    data=bytearray()
    for chunk in response.iter_content(4096):
        data.extend(chunk[:16384-len(data)])
        if len(data)>=16384:
            break
    body=data.decode('utf-8',errors='replace').lower()
    media=response.headers.get('Content-Type','').split(';',1)[0].strip().lower()
    content_kind={'text/html':'html','application/json':'json',
                  'text/plain':'text'}.get(media,'other')
    location=response.headers.get('Location','')
    try:
        host=urlparse(location).hostname
    except ValueError:
        host=None
    destination=({'urs.earthdata.nasa.gov':'earthdata_login',
                  'opendap.earthdata.nasa.gov':'opendap'}.get(host,'other')
                 if location else 'none')
    code=re.search(r'\bcode\s*=\s*(\d{1,5})\s*;',body)
    markers={'access_denied':('accessdenied','access denied'),
             'backend':('bes','upstream'), 'storage':('s3','amazonaws'),
             'invalid_token':('invalid token','invalid_token'),
             'expired_token':('token expired','expired token'),
             'application':('hyrax in the cloud','authorized applications')}
    return dict(http_status=response.status_code,content_kind=content_kind,
                redirect_destination=destination,sample_bytes=len(data),
                dap_error_code=int(code.group(1)) if code else None,
                error_markers=[key for key,words in markers.items()
                               if any(word in body for word in words)],
                authorization_message_detected=any(marker in body for marker in (
                    'application not authorized','application is not authorized',
                    'approve this application','authorize this application')))


def is_dataset_metadata(data,protocol):
    if protocol=='dds':
        text=data.decode('utf-8',errors='replace').strip()
        return (text.startswith('Dataset {') and 'Float32' in text
                and text.endswith(';') and '<html' not in text.lower())
    if protocol=='dmr':
        try:
            root=ET.fromstring(data)
        except ET.ParseError:
            return False
        ns='{http://xml.opendap.org/ns/DAP/4.0#}'
        return root.tag==ns+'Dataset' and any(
            element.tag in (ns+'Float32',ns+'Float64') for element in root.iter())
    return False


def probe_matrix(session,url,destination):
    import requests
    url=opendap_url([{'href':url}])
    probes=[('service_control','dds',
             'https://opendap.earthdata.nasa.gov/hyrax/data/nc/fnoc1.nc.dds'),
            ('merra_dap2','dds',url+'.dds'),('merra_dap4','dmr',url+'.dmr')]
    results=[]
    for name,protocol,target in probes:
        record=dict(probe=name,metadata_verified=False)
        response=None
        try:
            response=session.get(target,timeout=90,stream=True,allow_redirects=False)
            if response.status_code!=200:
                record.update(response_diagnostics(response))
            else:
                data=bytearray()
                for chunk in response.iter_content(65536):
                    data.extend(chunk)
                    if len(data)>2*1024*1024:
                        raise ValueError('Metadata response exceeds safety limit')
                valid=is_dataset_metadata(data,protocol)
                record.update(http_status=200,bytes=len(data),metadata_verified=valid)
                if valid and name!='service_control':
                    destination.mkdir(parents=True,exist_ok=True)
                    (destination/f'M2I3NVASM_20200101.{protocol}').write_bytes(data)
        except (requests.RequestException,ValueError) as exc:
            record['error_type']=type(exc).__name__
        finally:
            if response is not None:
                response.close()
        results.append(record)
    return dict(probes=results,service_control_verified=results[0]['metadata_verified'],
                merra_metadata_verified=any(r['metadata_verified'] for r in results[1:]))


def main():
    import requests
    import earthaccess
    status('reading_public_catalog')
    for attempt in range(3):
        try:
            response=requests.get('https://cmr.earthdata.nasa.gov/search/granules.json',params={
                'collection_concept_id':'C1276812900-GES_DISC',
                'temporal':'2020-01-01T00:00:00Z,2020-01-01T23:59:59Z','page_size':1},timeout=45)
            break
        except (requests.ConnectionError,requests.Timeout):
            if attempt==2:
                raise
    response.raise_for_status()
    item=response.json()['feed']['entry'][0]
    url=opendap_url(item['links'])
    status('waiting_for_local_login',product='M2I3NVASM',date='2020-01-01')
    credentials=credential_form()
    if credentials is None:
        status('cancelled_by_user');return
    status('authenticating')
    # Third-party login diagnostics may contain account data; never serialize them.
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        try:
            auth=login_from_form(earthaccess,*credentials)
        finally:
            del credentials
    if not auth.authenticated:
        status('authentication_failed');return
    status('checking_authorized_opendap',authenticated=True)
    session=earthaccess.get_requests_https_session()
    try:
        results=probe_matrix(session,url,OUT.parent)
    finally:
        session.close()
    state=('authorized_metadata_verified' if results['merra_metadata_verified']
           else 'data_access_not_confirmed')
    status(state,authenticated=True,product='M2I3NVASM',date='2020-01-01',
           data_downloaded=False,**results)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('failed',error_type=type(exc).__name__)
