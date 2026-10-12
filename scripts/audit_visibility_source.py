"""Audit visibility reporting regimes without fitting or scoring forecasts."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'artifacts/task01_recovery_head/visibility_source_20260918'
    if out.exists():
        raise FileExistsError(out)
    annual, hashes, frames = {}, {}, []
    for year in range(2014, 2026):
        p = ROOT / f'data/raw/noaa_isd/{year}/56294099999.csv'
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        raw = pd.read_csv(p, usecols=['DATE', 'VIS'], dtype=str)
        parts = raw.VIS.str.split(',', expand=True)
        x = pd.to_numeric(parts[0], errors='coerce')
        valid = x.between(0, 160000) & parts[1].isin(['1', '5'])
        qc5 = parts[1].eq('5')
        observed = pd.to_datetime(raw.DATE, utc=True).dt.tz_convert('Asia/Shanghai')
        frame = pd.DataFrame({'timestamp': observed.dt.ceil('h'), 'observed_at': observed,
                              'visibility_m': x.where(valid), 'quality': parts[1]})
        frame = frame.sort_values('observed_at', kind='stable').drop_duplicates('timestamp', keep='last')
        frames.append(frame)
        annual[str(year)] = {'raw_rows': len(raw), 'hourly_rows': len(frame),
            'valid_hours': int(frame.visibility_m.notna().sum()),
            'qc5_raw_rows': int(qc5.sum()), 'qc5_not_9900_rows': int((qc5 & x.ne(9900)).sum()),
            'top_raw_values': {str(k): int(n) for k,n in x[valid].value_counts().head(4).items()},
            'hours_below_5000': int(frame.visibility_m.lt(5000).sum())}
    hourly = pd.concat(frames).sort_values('observed_at', kind='stable').drop_duplicates('timestamp', keep='last')
    assert hourly.timestamp.is_unique and hourly.observed_at.le(hourly.timestamp).all()
    coverage = {}
    lookup = hourly.set_index('timestamp').visibility_m
    for name, file in [('training','f024_training_rows.parquet'),('development','inference_rows.parquet')]:
        p = ROOT / 'artifacts/task01_recovery_head/gefs_2021_extension_v1' / file
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        d = pd.read_parquet(p)
        times = pd.DatetimeIndex(d.timestamp)
        past = lookup.reindex(times-pd.Timedelta(hours=3))
        coverage[name] = {'origins': len(d), 'lag3_valid': int(past.notna().sum()),
                          'lag3_below_5000': int(past.lt(5000).sum())}
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for p,h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    out.mkdir()
    hourly.to_parquet(out/'hourly.parquet',index=False)
    result = {'annual':annual,'coverage':coverage,'input_sha256':hashes,
              'hourly_sha256':hashlib.sha256((out/'hourly.parquet').read_bytes()).hexdigest(),
              'limits':['9900 and 9999 are not the ISD missing sentinel 999999.',
                        'QC5 concentration at 9900 is an observed encoding association, not a proven sensor cause.',
                        'No real-time publication evidence, no candidate score computed.']}
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'coverage':coverage,'all_qc5_equal_9900':all(v['qc5_not_9900_rows']==0 for v in annual.values())},indent=2))


if __name__ == '__main__':
    main()
