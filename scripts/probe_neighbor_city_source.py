"""Bounded source-only probe; never reads targets or fits a model."""

import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'artifacts/task01_recovery_head/neighbor_city_source_probe_20260918'
    out.mkdir(exist_ok=False)
    # Geographic candidates fixed before inspecting any prediction errors.
    cities = ['成都', '德阳', '眉山', '资阳', '雅安', '绵阳', '遂宁', '乐山']
    dates = ['20200115', '20210115', '20220115']
    results = []
    for date in dates:
        url = f'https://quotsoft.net/air/data/china_cities_{date}.csv'
        record = {'date': date, 'url': url}
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                raw = response.read()
                record['status'] = response.status
                record['last_modified'] = response.headers.get('Last-Modified')
            (out / f'{date}.csv').write_bytes(raw)
            table = pd.read_csv(io.BytesIO(raw))
            independent = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
            assert len(independent) == len(table)
            assert set(table.date.astype(str)) == {date}
            assert not table.duplicated(['date', 'hour', 'type']).any()
            assert table.hour.between(0, 23).all()
            record['sha256'] = hashlib.sha256(raw).hexdigest()
            record['rows'] = len(table)
            record['coverage'] = {}
            for city in cities:
                counts = {}
                for pollutant in ['AQI', 'PM2.5', 'PM10']:
                    values = pd.to_numeric(table.loc[table.type.eq(pollutant), city], errors='raise')
                    count = int(values.notna().sum())
                    other = sum(r['type'] == pollutant and bool(r[city].strip()) for r in independent)
                    assert count == other
                    counts[pollutant] = count
                record['coverage'][city] = counts
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        results.append(record)
        print(json.dumps(record, ensure_ascii=True), flush=True)
    report = {'cities': cities, 'dates': dates, 'results': results,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Three historical days only. No full-history or real-time availability claim.'}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
