"""Bounded resumable collection of fixed pre-2023 historical input dates."""

import hashlib
import json
from pathlib import Path
import urllib.request

import pandas as pd

from haze_forecast.neighbor_city_data import parse_city_day, required_days


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'artifacts/task01_recovery_head/gefs_2021_extension_v1/f024_training_rows.parquet'
    out = root/'artifacts/task01_recovery_head/neighbor_city_collection_20260918'
    out.mkdir(exist_ok=True)
    dates = required_days(pd.read_parquet(source, columns=['timestamp']).timestamp)
    assert len(dates) == 478 and max(dates) < '20230101'
    paths = [source, Path(__file__).resolve(), root/'src/haze_forecast/neighbor_city_data.py',
             root/'neighbor_city_collection_plan_20260918.md']
    protocol = {'dates': dates, 'input_sha256': {str(p): sha(p) for p in paths}}
    config = out/'protocol.json'
    if config.exists():
        assert json.loads(config.read_text(encoding='utf-8')) == protocol
    else:
        config.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    ledger = out/'attempts.jsonl'
    attempts = [json.loads(x) for x in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
    done = {}
    for r in attempts:
        if r.get('ok'):
            path = out/f"{r['date']}.csv"
            assert sha(path) == r['sha256']
            parse_city_day(path.read_bytes(), r['date'])
            done[r['date']] = r
    pending = [d for d in dates if d not in done and sum(r['date'] == d for r in attempts) < 2]
    for date in pending[:64]:
        record = {'date': date, 'url': f'https://quotsoft.net/air/data/china_cities_{date}.csv', 'ok': False}
        try:
            with urllib.request.urlopen(record['url'], timeout=20) as response:
                raw = response.read()
                record['status'] = response.status
                record['last_modified'] = response.headers.get('Last-Modified')
            parsed = parse_city_day(raw, date)
            record['nonmissing_cells'] = int(parsed.drop(columns='timestamp').notna().sum().sum())
            with (out/f'{date}.csv').open('xb') as target:
                target.write(raw)
            record.update(ok=True, sha256=hashlib.sha256(raw).hexdigest())
            done[date] = record
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        with ledger.open('a', encoding='utf-8') as target:
            target.write(json.dumps(record)+'\n')
        attempts.append(record)
        print(date, record['ok'], len(done), '/', len(dates), flush=True)
    summary = {'required_days': len(dates), 'parsed_days': len(done), 'remaining_days': len(dates)-len(done),
               'failed_attempts': sum(not r['ok'] for r in attempts),
               'exhausted_dates': [d for d in dates if d not in done and sum(r['date']==d for r in attempts)>=2]}
    assert all(sha(Path(p)) == h for p,h in protocol['input_sha256'].items())
    (out/'progress.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
