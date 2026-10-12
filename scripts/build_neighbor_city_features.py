"""Match exact 48-hour-old city observations without filling missing values."""

import csv
from datetime import timedelta
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.neighbor_city_data import CITIES, POLLUTANTS, parse_city_day


def main():
    root = Path(__file__).resolve().parents[1]
    collection = root/'artifacts/task01_recovery_head/neighbor_city_collection_20260918'
    out = root/'artifacts/task01_recovery_head/neighbor_city_features_20260918'
    if out.exists():
        raise FileExistsError(out)
    progress = json.loads((collection/'progress.json').read_text(encoding='utf-8'))
    assert progress['remaining_days'] == len(progress['exhausted_dates'])
    origins = pd.read_parquet(root/'artifacts/task01_recovery_head/gefs_2021_extension_v1/f024_training_rows.parquet', columns=['timestamp'])
    records = [json.loads(x) for x in (collection/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    frames = []
    lookup = {}
    hashes = {}
    for record in records:
        if not record['ok']:
            continue
        path = collection/f"{record['date']}.csv"
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        assert hashes[path.name] == record['sha256']
        frames.append(parse_city_day(raw, record['date']))
        for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
            if r['type'] in POLLUTANTS:
                for city in CITIES:
                    text = r.get(city, '').strip()
                    value = float(text) if text and text.lower() not in ['nan','na','null','n/a'] else np.nan
                    lookup[(r['date'], int(r['hour']), city, r['type'])] = value
    hourly = pd.concat(frames, ignore_index=True).set_index('timestamp').sort_index()
    assert hourly.index.is_unique
    source = origins.timestamp-pd.Timedelta(hours=48)
    features = hourly.reindex(pd.DatetimeIndex(source)).reset_index(drop=True)
    names = list(features)
    # Standard-library date arithmetic and raw csv values independently rebuild every cell.
    direct = []
    for origin in origins.timestamp:
        at = origin.to_pydatetime()-timedelta(hours=48)
        direct.append([lookup.get((at.strftime('%Y%m%d'), at.hour, city, pollutant), np.nan)
                       for city in CITIES for pollutant in POLLUTANTS])
    np.testing.assert_allclose(features.to_numpy(), direct, rtol=0, atol=0, equal_nan=True)
    features.columns = [f'city48_{i:02d}' for i in range(len(names))]
    data = pd.concat([origins.reset_index(drop=True), source.rename('source_timestamp').reset_index(drop=True), features], axis=1)
    assert (data.source_timestamp < data.timestamp).all()
    coverage = {}
    for year in sorted(data.timestamp.dt.year.unique()):
        subset = data.loc[data.timestamp.dt.year.eq(year), features.columns[3:]]
        coverage[str(year)] = {'origins': len(subset), 'all_21_neighbor_values': int(subset.notna().all(axis=1).sum()),
                               'any_neighbor_value': int(subset.notna().any(axis=1).sum())}
    out.mkdir()
    data.to_parquet(out/'features.parquet', index=False)
    report = {'coverage': coverage, 'column_mapping': dict(zip(features.columns, names)),
              'source_sha256': hashes, 'rows': len(data), 'independently_checked_cells': len(data)*len(names),
              'missing_dates': progress['exhausted_dates'],
              'feature_sha256': hashlib.sha256((out/'features.parquet').read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': '48-hour lag is a research proxy, not proof of historical publication availability.'}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(coverage), flush=True)


if __name__ == '__main__':
    main()
