"""Build recent offline inputs with independent raw-cell and timestamp checks."""

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
    root=Path(__file__).resolve().parents[1]
    base=root/'artifacts/task01_recovery_head'
    source=base/'neighbor_city_recent_collection_20260918'
    out=base/'neighbor_city_recent_features_20260918'
    if out.exists():
        raise FileExistsError(out)
    progress=json.loads((source/'progress.json').read_text())
    assert progress['remaining_days']==len(progress['exhausted_dates'])
    origins=pd.read_parquet(base/'gefs_2021_extension_v1/f024_training_rows.parquet',columns=['timestamp'])
    protocol=json.loads((source/'protocol.json').read_text())
    needed=sorted({(t.to_pydatetime()-timedelta(hours=1)).strftime('%Y%m%d') for t in origins.timestamp})
    assert needed==protocol['dates']
    records=json.loads((source/'sources.json').read_text())
    assert len({r['date'] for r in records})==len(records)==progress['parsed_days']
    frames=[]; lookup={}
    for record in records:
        raw=Path(record['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==record['sha256']
        frames.append(parse_city_day(raw,record['date']))
        keys=set()
        for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
            assert row['date']==record['date']
            hour=float(row['hour']); assert 0<=hour<=23 and hour.is_integer()
            key=(row['date'],int(hour),row['type']); assert key not in keys; keys.add(key)
            if row['type'] in POLLUTANTS:
                for city in CITIES:
                    text=row.get(city,'').strip()
                    value=float(text) if text and text.lower() not in ['nan','na','null','n/a'] else np.nan
                    assert np.isnan(value) or (np.isfinite(value) and value>=0)
                    lookup[(*key,city)]=value
    hourly=pd.concat(frames,ignore_index=True).set_index('timestamp')
    assert hourly.index.is_unique
    at=origins.timestamp-pd.Timedelta(hours=1)
    extra=hourly.reindex(pd.DatetimeIndex(at)).reset_index(drop=True)
    direct=[]
    for origin in origins.timestamp:
        t=origin.to_pydatetime()-timedelta(hours=1)
        direct.append([lookup.get((t.strftime('%Y%m%d'),t.hour,p,c),np.nan) for c in CITIES for p in POLLUTANTS])
    np.testing.assert_allclose(extra.to_numpy(),direct,rtol=0,atol=0,equal_nan=True)
    mapping=dict(zip([f'city01_{i:02d}' for i in range(24)],extra.columns))
    extra.columns=list(mapping)
    data=pd.concat([origins,at.rename('source_timestamp'),extra],axis=1)
    coverage={}
    for year in sorted(origins.timestamp.dt.year.unique()):
        group=extra.loc[origins.timestamp.dt.year.eq(year),list(mapping)[3:]]
        coverage[str(year)]={'origins':len(group),'complete_neighbors':int(group.notna().all(axis=1).sum()),
                             'any_neighbor':int(group.notna().any(axis=1).sum())}
    out.mkdir()
    data.to_parquet(out/'features.parquet',index=False)
    report={'column_mapping':mapping,'coverage':coverage,'raw_files_verified':len(records),
            'independent_cell_checks':len(data)*24,'missing_dates':progress['exhausted_dates'],
            'feature_sha256':hashlib.sha256((out/'features.parquet').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'sources_manifest_sha256':hashlib.sha256((source/'sources.json').read_bytes()).hexdigest()}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
