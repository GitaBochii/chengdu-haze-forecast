"""Inventory every eligible historical hour; never imply missing weather is available."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from haze_forecast.labels import add_particle_primary
from haze_forecast.observed_business import build_business_states
from haze_forecast.interval_recovery_labels import interval_recovery_bounds

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def weather_times(origin):
    latest = origin.tz_convert('UTC').floor('6h')
    return latest - pd.Timedelta(hours=6), latest


def assign_partition(year):
    if year <= 2023:
        return 'train'
    return {2024: 'dev', 2025: 'retrospective_test'}.get(year, 'recent_audit')


def inventory():
    source = ROOT / 'data/processed/regional_air_models_full_history/city_hourly.parquet'
    city = pd.read_parquet(source)
    states = build_business_states(add_particle_primary(city))
    eligible = states.loc[states.eligible].copy()
    rows = eligible[['timestamp','episode_id','continuity_block']].merge(
        interval_recovery_bounds(city, eligible.timestamp), on='timestamp', validate='one_to_one')
    rows['partition'] = rows.timestamp.dt.year.map(assign_partition)
    inputs = rows.timestamp.map(weather_times)
    rows['weather_earlier_utc'] = [pair[0] for pair in inputs]
    rows['weather_latest_utc'] = [pair[1] for pair in inputs]
    rows['weather_age_hours'] = (rows.timestamp-rows.weather_latest_utc).dt.total_seconds()/3600
    metadata = json.loads((HERE/'hf_merra_inventory.json').read_text('utf-8'))
    paths = {item['path'] for item in metadata['entries']}
    def has_metadata(pair):
        return all(f'merra-2/{prefix}{time:%Y%m%d}.nc' in paths
                   for time in pair for prefix in ('MERRA2_sfc_','MERRA_pres_'))
    rows['hf_day_pair_metadata_only'] = inputs.map(has_metadata)
    pilot = json.loads((HERE/'pilot_candidate_audit.json').read_text('utf-8'))
    ready = {pd.Timestamp(row['timestamp']) for row in pilot['rows']}
    rows['existing_pilot_origin'] = rows.timestamp.isin(ready)
    # Blocks spanning proposed partitions must not become train/test duplicates.
    full_partitions = states.timestamp.dt.year.map(assign_partition)
    crossing = states.assign(partition=full_partitions).groupby('continuity_block').partition.nunique()
    rows['cross_partition_block'] = rows.continuity_block.isin(crossing[crossing.gt(1)].index)
    ends = {'train':pd.Timestamp('2024-01-01',tz='Asia/Shanghai'),
            'dev':pd.Timestamp('2025-01-01',tz='Asia/Shanghai'),
            'retrospective_test':pd.Timestamp('2026-01-01',tz='Asia/Shanghai')}
    rows['label_mature_within_partition'] = [
        stamp < ends[part] if part in ends else stamp <= city.timestamp.max()
        for stamp,part in zip(rows.label_available_at,rows.partition)]
    starts = {'dev':pd.Timestamp('2024-01-01',tz='Asia/Shanghai'),
              'retrospective_test':pd.Timestamp('2025-01-01',tz='Asia/Shanghai'),
              'recent_audit':pd.Timestamp('2026-01-01',tz='Asia/Shanghai')}
    rows['history_within_partition'] = [stamp-pd.Timedelta(hours=23) >= starts.get(part,city.timestamp.min())
                                         for stamp,part in zip(rows.timestamp,rows.partition)]
    def summarize(group):
        return dict(eligible_hours=len(group),days=int(group.timestamp.dt.date.nunique()),
                    episodes=int(group.episode_id.nunique()),blocks=int(group.continuity_block.nunique()),
                    weather_day_metadata_hours=int(group.hf_day_pair_metadata_only.sum()),
                    existing_pilot_origins=int(group.existing_pilot_origin.sum()),
                    cross_partition_hours=int(group.cross_partition_block.sum()),
                    labels_not_mature=int((~group.label_mature_within_partition).sum()),
                    history_crosses_start=int((~group.history_within_partition).sum()))
    output = HERE/'full_history'
    output.mkdir(exist_ok=True)
    rows.to_parquet(output/'candidate_hours.parquet',index=False)
    requests = pd.DatetimeIndex([time for pair in inputs for time in pair]).unique().sort_values()
    pd.DataFrame({'weather_time_utc':requests}).to_parquet(output/'required_weather_times.parquet',index=False)
    report = dict(status='inventory_only_not_trained',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_start=city.timestamp.min().isoformat(),source_end=city.timestamp.max().isoformat(),
        total=summarize(rows),yearly={str(year):summarize(group) for year,group in rows.groupby(rows.timestamp.dt.year)},
        required_weather_times=len(requests),required_weather_days=len(set(requests.date)),
        limitations=['Daily file metadata does not verify intraday times, variables or integrity.',
                    'Existing pilot origins are listed, not an assertion that every new window is ready.',
                    'Proposed calendar partitions require history, label maturity and event-purge checks before training.',
                    '2025 was previously examined and is not a new blind test.',
                    'Historical reanalysis is not evidence of real-time data availability.'])
    (output/'coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(inventory(),ensure_ascii=False,indent=2))
