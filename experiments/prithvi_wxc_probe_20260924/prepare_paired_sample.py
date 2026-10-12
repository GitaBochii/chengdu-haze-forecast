"""Align one real weather cube with observed pollution history and recovery bounds."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.interval_recovery_labels import interval_recovery_bounds

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / 'data/processed/regional_air_models_full_history'
POLLUTANTS = ['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3_8h']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    official = json.loads((ROOT / 'official_batch_report.json').read_text('utf-8'))
    regional = json.loads((ROOT / 'regional_sample_report.json').read_text('utf-8'))
    with np.load(ROOT / 'regional_raw_sample.npz', allow_pickle=False) as cube:
        raw_times = cube['times'].tolist()
        shape = list(cube['x'].shape)
    if (raw_times != regional['times_from_file'] or shape != [2, 160, 30, 32]
            or official['shapes']['x'] != [1, *shape]
            or not official['raw_cube_matches_official_exactly']):
        raise ValueError('weather evidence is inconsistent')
    times = pd.DatetimeIndex(pd.to_datetime(raw_times, utc=True))
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError('weather times must be increasing and unique')
    if (times[1] - times[0]).total_seconds() / 3600 != official['input_time_hours']:
        raise ValueError('weather input interval mismatch')
    origin = times[-1].tz_convert('Asia/Shanghai')
    if origin != pd.Timestamp('2020-01-01T14:00:00+08:00'):
        raise ValueError('unexpected sample origin; review protocol before changing dates')
    start = origin - pd.Timedelta(hours=23)
    expected = pd.date_range(start, origin, freq='h')
    city = pd.read_parquet(DATA / 'city_hourly.parquet')
    stations = pd.read_parquet(DATA / 'station_hourly.parquet')
    selected = []
    for kind, frame in [('city', city), ('station', stations)]:
        history = frame.loc[frame.timestamp.between(start, origin),
                            ['timestamp', 'node_id', 'latitude', 'longitude', *POLLUTANTS]].copy()
        for _, group in history.groupby('node_id'):
            if not pd.DatetimeIndex(group.timestamp.sort_values()).equals(expected):
                raise ValueError('history must contain exactly 24 observed hourly positions per node')
        if history.empty:
            raise ValueError('missing history')
        history.insert(0, 'record_type', kind)
        selected.append(history)
    history = pd.concat(selected, ignore_index=True)
    if history.timestamp.max() > origin:
        raise ValueError('future observations entered history')
    bounds = interval_recovery_bounds(city, pd.DatetimeIndex([origin])).iloc[0]
    if bounds.reason != 'exact' or bounds.exact_hours != 46:
        raise ValueError('independent sample target differs from audited 46 hours')
    # Supervision remains separate; no future values or target columns enter inputs.
    output = ROOT / 'paired_pollution_history.parquet'
    history.to_parquet(output, index=False)
    restored = pd.read_parquet(output)
    pd.testing.assert_frame_equal(restored, history)
    report = {
        'status': 'one_real_time_aligned_sample_verified', 'samples': 1,
        'weather_times_utc': [t.isoformat() for t in times],
        'origin_utc': times[-1].isoformat(), 'origin_local': origin.isoformat(),
        'history_start': start.isoformat(), 'history_end': origin.isoformat(),
        'history_hours': 24, 'history_rows': len(history),
        'station_nodes': int(history.loc[history.record_type.eq('station'), 'node_id'].nunique()),
        'city_rows': int(history.record_type.eq('city').sum()),
        'missing_pollutant_cells': {k: int(v) for k, v in history[POLLUTANTS].isna().sum().items()},
        'missing_policy': 'preserved; no imputation performed',
        'target': {'definition': 'start of first future six consecutive hours with observed AQI <= 100',
                   'lower_bound_hours': float(bounds.lower_bound_hours),
                   'upper_bound_hours': float(bounds.upper_bound_hours),
                   'exact_hours': float(bounds.exact_hours), 'reason': bounds.reason,
                   'label_available_at': bounds.label_available_at.isoformat()},
        'input_columns': list(history.columns), 'history_sha256': sha(output),
        'source_sha256': {str(p.relative_to(ROOT.parents[1])): sha(p) for p in
                         [DATA / 'city_hourly.parquet', DATA / 'station_hourly.parquet',
                          ROOT / 'regional_raw_sample.npz', ROOT / 'official_batch_report.json']},
        'boundaries': ['MERRA-2 retrospective reanalysis; operational availability not established',
                       'single sample is insufficient for training or generalization evaluation',
                       'foundation-model pretraining period must be audited separately',
                       'no pretrained weights loaded or inference performed by this script'],
    }
    (ROOT / 'paired_sample_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), 'utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
