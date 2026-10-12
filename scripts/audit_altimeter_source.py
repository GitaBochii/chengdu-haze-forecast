"""Audit an unused MA1 observation without relabelling it as sea-level pressure."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'artifacts/task01_recovery_head/altimeter_source_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [ROOT / f'data/raw/noaa_isd/{y}/56294099999.csv' for y in range(2014, 2026)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    parts, annual = [], {}
    for year, path in zip(range(2014, 2026), paths):
        raw = pd.read_csv(path, usecols=lambda n: n in ['DATE', 'MA1', 'STATION'], dtype=str)
        assert raw.STATION.eq('56294099999').all()
        observed = pd.to_datetime(raw.DATE, utc=True, errors='raise').dt.tz_convert('Asia/Shanghai')
        field = raw.get('MA1', pd.Series(pd.NA, index=raw.index, dtype='string')).str.split(',', expand=True)
        if field.shape[1] < 2:
            value = pd.Series(float('nan'), index=raw.index)
        else:
            numeric = pd.to_numeric(field[0], errors='coerce')
            value = (numeric / 10).where(numeric.between(8635, 10904) & field[1].isin(['1', '5']))
        d = pd.DataFrame({'observed_at': observed, 'timestamp': observed.dt.ceil('h'),
                          'altimeter_setting_hpa': value})
        d = d.sort_values('observed_at', kind='stable').drop_duplicates('timestamp', keep='last')
        assert d.observed_at.le(d.timestamp).all()
        assert (d.timestamp-d.observed_at).lt(pd.Timedelta(hours=1)).all()
        annual[str(year)] = {'hours_with_report': len(d), 'valid_altimeter_hours': int(d.altimeter_setting_hpa.notna().sum()),
                             'minimum_hpa': float(d.altimeter_setting_hpa.min()),
                             'maximum_hpa': float(d.altimeter_setting_hpa.max())}
        parts.append(d)
    hourly = pd.concat(parts).sort_values('observed_at', kind='stable').drop_duplicates('timestamp', keep='last')
    coverage = {}
    for name, relative in [('training', 'gefs_2021_extension_v1/f024_training_rows.parquet'),
                           ('development', 'gefs_2021_extension_v1/inference_rows.parquet')]:
        path = ROOT / 'artifacts/task01_recovery_head' / relative
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        origins = pd.read_parquet(path)
        clock = pd.DatetimeIndex(origins.timestamp)
        lookup = hourly.set_index('timestamp').altimeter_setting_hpa
        coverage[name] = {'rows': len(origins), 'current_valid': int(lookup.reindex(clock).notna().sum()),
                          'three_hour_lag_valid': int(lookup.reindex(clock-pd.Timedelta(hours=3)).notna().sum())}
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for p, h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    out.mkdir()
    hourly.to_parquet(out / 'hourly.parquet', index=False)
    result = {'input_sha256': hashes, 'annual': annual, 'origin_coverage': coverage,
              'source_spec': 'https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf',
              'rule': 'MA1 first value / 10; QC 1 or 5; range 863.5-1090.4 hPa; '
                      'latest report per ceiling hour, no filling; not SLP; '
                      'observation clock does not prove publication availability.',
              'hourly_sha256': hashlib.sha256((out / 'hourly.parquet').read_bytes()).hexdigest()}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'annual': annual, 'origin_coverage': coverage}, indent=2))


if __name__ == '__main__':
    main()
