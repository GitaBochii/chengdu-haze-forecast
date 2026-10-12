"""Compare isolated source and extracted-runtime service behavior."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROBE = r'''
import hashlib,json,pathlib,threading
from urllib.error import HTTPError
from urllib.request import urlopen
import pandas as pd
import haze_forecast.showcase_server as module
root=pathlib.Path.cwd().resolve()
assert pathlib.Path(module.__file__).resolve().is_relative_to(root/'src')
assert module.ROOT.resolve()==root
service=module.ShowcaseService()
status=service.status()
values={field['name']:field['default'] for field in service.schema()['fields']}
manual=service.predict({'mode':'manual','timestamp':'2026-09-11T12:00','values':values})
frame=pd.DataFrame([values]*25)
frame.insert(0,'timestamp',pd.date_range('2026-09-10',periods=25,freq='h'))
sequence=service.predict({'mode':'sequence','csv':frame.to_csv(index=False)})
date=service.dates()['dates'][0]
record=service.history(date)['records'][0]
replay=service.replay(record['id'])
assert manual['eligible'] and sequence['eligible']
assert manual['prediction_hours']>0 and sequence['prediction_hours']>0
assert len(replay['trend'])>1 and len(sequence['trend'])==25
server=module.create_server(port=0,service=service)
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
base='http://127.0.0.1:'+str(server.server_port)
try:
    assert json.load(urlopen(base+'/api/status'))['ready']
    static={}
    for path in sorted(module.STATIC_FILES):
        response=urlopen(base+'/'+path)
        assert response.status==200
        body=response.read()
        if path=='workbench.html':
            body=body.replace(b'id="tab-multimodal" hidden',b'id="tab-multimodal"')
        if path=='workbench.js':
            body=body.replace(b"document.querySelectorAll('[role=tab]:not([hidden])')",b"document.querySelectorAll('[role=tab]')")
        static[path]=hashlib.sha256(body).hexdigest()
    for path in ['/artifacts/cnemc_2025_deployment_candidate/aft_model.joblib','/../pyproject.toml']:
        try:urlopen(base+path)
        except HTTPError as error:assert error.code==404
        else:raise AssertionError('Private or traversal path exposed')
finally:
    server.shutdown();server.server_close();thread.join()
print(json.dumps({'status':status,'manual':manual,'sequence':sequence,'replay':replay,'canonical_static_sha256':static},ensure_ascii=False,allow_nan=False))
'''


def probe(root: Path) -> dict:
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(root / 'src')
    environment['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(
        [sys.executable, '-X', 'utf8', '-c', PROBE], cwd=root, env=environment,
        capture_output=True, text=True, encoding='utf-8', timeout=120, check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime_root', type=Path)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    runtime = args.runtime_root.resolve()
    if source == runtime:
        raise ValueError('Runtime verification requires a separate extracted directory')
    original = probe(source)
    extracted = probe(runtime)
    assert extracted == original, 'Extracted runtime differs from original service'
    html = (runtime / 'project-showcase' / 'workbench.html').read_text(encoding='utf-8')
    assert html.count('id="tab-multimodal" hidden') == 1
    script = (runtime / 'project-showcase' / 'workbench.js').read_text(encoding='utf-8')
    assert script.count("document.querySelectorAll('[role=tab]:not([hidden])')") == 1
    report = {
        'status': 'passed_isolated_source_and_runtime_equality',
        'source_root': str(source), 'runtime_root': str(runtime),
        'interpreter': sys.executable,
        'environment_boundary': 'Uses the existing interpreter/dependencies, but imports and data are asserted to be inside each tested root; not a fresh environment installation test.',
        'history_records': extracted['status']['history_records'],
        'manual_prediction_hours': extracted['manual']['prediction_hours'],
        'sequence_prediction_hours': extracted['sequence']['prediction_hours'],
        'static_files': len(extracted['canonical_static_sha256']),
        'checked': ['status', 'real_history_replay', 'manual_prediction', '25_hour_sequence', 'all_static_assets', 'private_asset_and_traversal_rejection', 'exact_prediction_response_equality', 'static_equality_except_declared_multimodal_tab_hiding_and_keyboard_filter'],
        'not_checked': ['fresh_dependency_installation', 'live_online_provider', 'browser_layout', 'multimodal_training_or_inference'],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
