import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from haze_forecast.showcase_server import ROOT, create_server


class ResearchStub:
    def status(self):
        return {'available':True}
    def cases(self):
        return {'records':[]}
    def replay(self,timestamp,seed=24,arm='spatial_fusion'):
        return {'timestamp':timestamp,'seed':seed,'arm':arm,'mode':'multimodal_archive'}
    def reinfer(self,timestamp,seed=24,arm='spatial_fusion'):
        return {**self.replay(timestamp,seed,arm),'mode':'multimodal_reinference'}


def test_research_routes_and_same_origin_boundary(monkeypatch):
    monkeypatch.setattr('haze_forecast.showcase_server.mimetypes.guess_type',lambda _:('text/plain',None))
    server=create_server(0,SimpleNamespace(root=ROOT,multimodal=ResearchStub()))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        assert json.load(urlopen(base+'/api/multimodal/status'))['available']
        assert json.load(urlopen(base+'/api/multimodal/cases'))=={'records':[]}
        result=json.load(urlopen(base+'/api/multimodal/replay?timestamp=sample&seed=42&arm=baseline'))
        assert result['seed']==42 and result['arm']=='baseline'
        script=urlopen(base+'/multimodal.js')
        assert script.status==200
        assert script.headers.get_content_type()=='application/javascript'
        for seed in ('nope','24.0','true','1'):
            with pytest.raises(HTTPError) as error:
                urlopen(base+'/api/multimodal/replay?seed='+seed)
            assert error.value.code==400
        payload=json.dumps({'timestamp':'sample','seed':24,'arm':'spatial_fusion'}).encode()
        good=Request(base+'/api/multimodal/reinfer',data=payload,headers={'Content-Type':'application/json','Origin':base})
        assert json.load(urlopen(good))['mode']=='multimodal_reinference'
        bad=Request(base+'/api/multimodal/reinfer',data=payload,headers={'Content-Type':'application/json','Origin':'https://other.example'})
        with pytest.raises(HTTPError) as error:
            urlopen(bad)
        assert error.value.code==403
    finally:
        server.shutdown();server.server_close();thread.join(3)
