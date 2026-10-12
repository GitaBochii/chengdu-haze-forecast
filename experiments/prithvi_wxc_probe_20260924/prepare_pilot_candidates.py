"""Independently recompute and verify the frozen 2020 candidate audit.

Read-only: never rewrites the protocol, manifest, labels, or split.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.labels import add_particle_primary
from haze_forecast.observed_business import build_business_states
from haze_forecast.interval_recovery_labels import interval_recovery_bounds

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def summary(frame):
    return {'days': len(frame), 'episodes': int(frame.episode_id.nunique()),
            'continuity_blocks': int(frame.continuity_block.nunique()),
            'paired_metadata_days': int(frame.paired_metadata.sum()),
            'exact_labels': int(frame.exact_hours.notna().sum()),
            'interval_labels': int(frame.reason.eq('interval').sum())}


def verify():
    manifest_path = HERE / 'pilot_candidate_audit.json'
    frozen_bytes = manifest_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    source = ROOT / 'data/processed/regional_air_models_full_history/city_hourly.parquet'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != frozen['source_sha256']:
        raise ValueError('city source changed from frozen audit')
    city = pd.read_parquet(source)
    states = build_business_states(add_particle_primary(city))
    candidates = states.loc[states.timestamp.dt.year.eq(2020)
                            & states.timestamp.dt.hour.eq(14) & states.eligible].copy()
    candidates['date'] = candidates.timestamp.dt.strftime('%Y%m%d')
    inventory = json.loads((HERE / 'hf_merra_inventory.json').read_text('utf-8'))
    if inventory['revision'] != 'e2f62b3e19890f1a5e225bb123acb4b33df85ef8':
        raise ValueError('unexpected inventory revision')
    objects = {item['path']: item for item in inventory['entries']}
    prefixes = ('MERRA2_sfc_', 'MERRA_pres_')
    candidates['paired_metadata'] = candidates.date.map(
        lambda date: all(f'merra-2/{prefix}{date}.nc' in objects for prefix in prefixes))
    candidates = candidates.merge(interval_recovery_bounds(city, candidates.timestamp),
                                  on='timestamp', validate='one_to_one')
    # Calendar boundaries were frozen before model fitting; do not optimize them.
    candidates['partition'] = np.where(candidates.timestamp.dt.month.lt(11), 'train',
                                      np.where(candidates.timestamp.dt.month.lt(12), 'dev', 'test'))
    cuts = [pd.Timestamp(value, tz='Asia/Shanghai') for value in
            ('2020-11-01', '2020-12-01', '2021-01-01')]
    crossing = set()
    for cut in cuts:
        crossing |= (set(states.loc[states.timestamp.lt(cut), 'continuity_block'])
                     & set(states.loc[states.timestamp.ge(cut), 'continuity_block']))
    ends = dict(zip(('train', 'dev', 'test'), cuts))
    starts = {'train': pd.Timestamp('2019-01-01', tz='Asia/Shanghai'),
              'dev': cuts[0], 'test': cuts[1]}
    candidates['cross_partition_block'] = candidates.continuity_block.isin(crossing)
    candidates['mature_within_partition'] = [time < ends[part] for time, part in
                                           zip(candidates.label_available_at, candidates.partition)]
    candidates['history_after_partition_start'] = [time - pd.Timedelta(hours=23) >= starts[part]
                                                  for time, part in zip(candidates.timestamp,
                                                                        candidates.partition)]
    expected = pd.DataFrame(frozen['rows'])
    for name in ('timestamp', 'label_available_at'):
        expected[name] = pd.to_datetime(expected[name], utc=True).dt.tz_convert('Asia/Shanghai')
    actual = candidates[expected.columns].sort_values('timestamp').reset_index(drop=True)
    expected = expected.sort_values('timestamp').reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    if (len(candidates) != 25 or candidates.cross_partition_block.any()
            or not candidates.mature_within_partition.all()
            or not candidates.history_after_partition_start.all()):
        raise ValueError('frozen candidate count or causal split invariant changed')
    for group_column in ('continuity_block', 'episode_id'):
        if candidates.groupby(group_column).partition.nunique().gt(1).any():
            raise ValueError(f'{group_column} crosses partitions')
    computed = {
        'total': summary(candidates),
        'monthly': {str(month): summary(candidates[candidates.timestamp.dt.month.eq(month)])
                    for month in range(1, 13)},
        'quarterly': {str(quarter): summary(candidates[candidates.timestamp.dt.quarter.eq(quarter)])
                      for quarter in range(1, 5)},
        'partitions': {part: summary(group) for part, group in candidates.groupby('partition', sort=False)},
        'cross_boundary_blocks_excluded': sorted(int(value) for value in crossing),
        'selected_rows_removed_by_boundaries': int(candidates.cross_partition_block.sum()),
        'selected_surface_vertical_bytes': sum(objects[f'merra-2/{prefix}{date}.nc']['size']
                                              for date in candidates.date for prefix in prefixes),
        'ordinary_aqi_gt100_daily_origins': int((states.timestamp.dt.year.eq(2020)
                                                & states.timestamp.dt.hour.eq(14)
                                                & states.aqi.gt(100)).sum()),
    }
    for name, value in computed.items():
        if frozen[name] != value:
            raise ValueError(f'audit summary differs: {name}')
    if manifest_path.read_bytes() != frozen_bytes:
        raise ValueError('frozen manifest changed during verification')
    return {'status': 'verified_against_frozen_manifest', 'rows': len(actual),
            'partitions': computed['partitions'], 'manifest_unchanged': True,
            'comparison': 'all frozen row fields, exact timestamps, labels, split, maturity and summary counts'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
