"""Check three label-critical missing hours against raw CSV and its current mirror."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]


def inspect(content, hour):
    frame = pd.read_csv(io.BytesIO(content))
    assert {'date','hour','type'}.issubset(frame.columns)
    count = int(frame.hour.eq(hour).sum())
    independent = sum(int(row['hour']) == hour for row in csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
    assert count == independent
    return {'rows': len(frame), 'present_hours': sorted(map(int, frame.hour.unique())),
            'target_hour_rows': count, 'sha256': hashlib.sha256(content).hexdigest()}


def main():
    out = ROOT/'artifacts/task01_recovery_head/recovery_gap_source_audit_20260918'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    records=[]
    for day,hour in [('20230107',5),('20230114',21),('20230202',6)]:
        original=ROOT/f'data/raw/cnemc_2023_daily/china_sites_{day}.csv'
        content=original.read_bytes()
        local=inspect(content,hour)
        url=f'https://quotsoft.net/air/data/china_sites_{day}.csv'
        row={'day':day,'target_hour':hour,'local_path':str(original),'local':local,'url':url}
        try:
            response=requests.get(url,timeout=40)
            row['http_status']=response.status_code
            response.raise_for_status()
            (out/f'current_{day}.csv').write_bytes(response.content)
            row['current']=inspect(response.content,hour)
            row['same_bytes']=content==response.content
            row['last_modified']=response.headers.get('Last-Modified')
        except Exception as error:
            row['error']=repr(error)
        assert hashlib.sha256(original.read_bytes()).hexdigest()==local['sha256']
        records.append(row)
        print(json.dumps(row),flush=True)
    report={'records':records,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Three label-critical hours only. Mirror is not first-party publication proof. No original file, labels, model or scores changed.'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
