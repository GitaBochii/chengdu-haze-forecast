"""Collect fixed one-hour-lag source days, reusing only hash-verified archives."""

import hashlib
import json
from pathlib import Path
import urllib.request

import pandas as pd

from haze_forecast.neighbor_city_data import parse_city_day


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    base = root/'artifacts/task01_recovery_head'
    old = base/'neighbor_city_collection_20260918'
    out = base/'neighbor_city_recent_collection_20260918'
    out.mkdir(exist_ok=True)
    source = base/'gefs_2021_extension_v1/f024_training_rows.parquet'
    origins = pd.read_parquet(source, columns=['timestamp']).timestamp
    days = sorted((origins-pd.Timedelta(hours=1)).dt.strftime('%Y%m%d').unique())
    assert len(days)==482 and max(days)<'20230101'
    paths=[source,Path(__file__).resolve(),root/'src/haze_forecast/neighbor_city_data.py',
           root/'neighbor_city_recent_plan_20260918.md',old/'attempts.jsonl']
    protocol={'dates':days,'input_sha256':{str(p):sha(p) for p in paths}}
    path=out/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())==protocol
    else:
        path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=[]
    for r in [json.loads(x) for x in (old/'attempts.jsonl').read_text().splitlines()]:
        if r['ok'] and r['date'] in days:
            path=old/f"{r['date']}.csv"
            assert sha(path)==r['sha256']
            parse_city_day(path.read_bytes(),r['date'])
            records.append({**r,'path':str(path),'reused':True})
    ledger=out/'attempts.jsonl'
    attempts=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    for r in attempts:
        if r['ok']:
            assert sha(Path(r['path']))==r['sha256']
            parse_city_day(Path(r['path']).read_bytes(),r['date'])
            records.append(r)
    done={r['date'] for r in records}
    pending=[day for day in days if day not in done and sum(r['date']==day for r in attempts)<2]
    for day in pending[:64]:
        record={'date':day,'ok':False,'reused':False,'url':f'https://quotsoft.net/air/data/china_cities_{day}.csv'}
        try:
            with urllib.request.urlopen(record['url'],timeout=20) as response:
                raw=response.read(); record['status']=response.status
                record['last_modified']=response.headers.get('Last-Modified')
            parse_city_day(raw,day)
            path=out/f'{day}.csv'
            with path.open('xb') as f:
                f.write(raw)
            record.update(ok=True,path=str(path),sha256=sha(path))
            records.append(record); done.add(day)
        except Exception as exc:
            record['error']=f'{type(exc).__name__}: {exc}'
        with ledger.open('a',encoding='utf-8') as f:
            f.write(json.dumps(record)+'\n')
        attempts.append(record)
        print(day,record['ok'],len(done),'/',len(days),flush=True)
    summary={'required_days':len(days),'parsed_days':len(done),'remaining_days':len(days)-len(done),
             'reused_days':sum(r['reused'] for r in records),
             'exhausted_dates':[day for day in days if day not in done and sum(r['date']==day for r in attempts)>=2],
             'failed_attempts':sum(not r['ok'] for r in attempts)}
    assert all(sha(Path(p))==h for p,h in protocol['input_sha256'].items())
    (out/'sources.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (out/'progress.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
