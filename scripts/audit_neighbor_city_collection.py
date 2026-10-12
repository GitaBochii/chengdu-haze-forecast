"""Independent csv audit of completed files; safe while the collector is idle."""

import csv
from datetime import timedelta
import hashlib
import io
import json
import math
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'artifacts/task01_recovery_head/neighbor_city_collection_20260918'
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    origins = pd.read_parquet(root/'artifacts/task01_recovery_head/gefs_2021_extension_v1/f024_training_rows.parquet', columns=['timestamp']).timestamp
    days = sorted({(x.to_pydatetime()-timedelta(hours=48)).strftime('%Y%m%d') for x in origins})
    assert days == protocol['dates']
    attempts = [json.loads(x) for x in (out/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    successful = [r for r in attempts if r['ok']]
    assert len({r['date'] for r in successful}) == len(successful)
    cities = ['成都', '德阳', '眉山', '资阳', '雅安', '绵阳', '遂宁', '乐山']
    complete = partial = cells = 0
    coverage = {}
    for record in successful:
        raw = (out/f"{record['date']}.csv").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record['sha256']
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        keys = set()
        counts = {f'{c}_{p}': 0 for c in cities for p in ['AQI', 'PM2.5', 'PM10']}
        for row in rows:
            assert row['date'] == record['date']
            hour = float(row['hour'])
            assert 0 <= hour <= 23 and hour.is_integer()
            key = (row['date'], int(hour), row['type'])
            assert key not in keys
            keys.add(key)
            if row['type'] not in ['AQI', 'PM2.5', 'PM10']:
                continue
            for city in cities:
                value = row.get(city, '').strip()
                if value and value.lower() not in ['nan', 'na', 'null', 'n/a']:
                    number = float(value)
                    assert math.isfinite(number) and number >= 0
                    counts[f"{city}_{row['type']}"] += 1
        n = sum(counts.values())
        assert n == record['nonmissing_cells']
        complete += n == 576
        partial += n != 576
        cells += n
        coverage[record['date']] = counts
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protocol['input_sha256'].items())
    report = {'required_dates': len(days), 'independently_audited_dates': len(successful),
              'complete_24h_all_city_pollutant_dates': complete, 'partial_dates': partial,
              'nonmissing_cells': cells, 'coverage': coverage,
              'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination = out/f'audit_{len(successful):03d}.json'
    with destination.open('x', encoding='utf-8') as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != 'coverage'}))


if __name__ == '__main__':
    main()
