"""Separate conservative interval-support manifest, not operational availability."""
import hashlib
import json
from pathlib import Path

import pandas as pd

BASE=Path(__file__).resolve().parent/'full_history'


def align(rows):
    if rows.timestamp.dt.tz is None or not rows.timestamp.is_unique or rows.timestamp.isna().any():
        raise ValueError('Unique timezone-aware origins required')
    result=rows.copy()
    origin=rows.timestamp.dt.tz_convert('UTC')
    latest=(origin-pd.Timedelta(hours=1)).dt.floor('6h')
    result['support_latest_utc']=latest
    result['support_earlier_utc']=latest-pd.Timedelta(hours=6)
    result['support_end_utc']=latest+pd.Timedelta(hours=1)
    result['support_alignment_changed']=latest.ne(rows.weather_latest_utc)
    result['support_weather_age_hours']=(origin-latest).dt.total_seconds()/3600
    if not result.support_end_utc.le(origin).all():
        raise ValueError('Weather support extends beyond origin')
    times=pd.concat([result.support_earlier_utc,result.support_latest_utc]).drop_duplicates().sort_values()
    required=pd.DataFrame({'weather_time_utc':times.reset_index(drop=True)})
    required['source_left_utc']=required.weather_time_utc-pd.Timedelta(minutes=30)
    required['source_right_utc']=required.weather_time_utc+pd.Timedelta(minutes=30)
    required['source_right_support_end_utc']=required.weather_time_utc+pd.Timedelta(hours=1)
    return result,required


def main():
    source=BASE/'candidate_hours.parquet'
    aligned,required=align(pd.read_parquet(source))
    out=BASE/'support_aligned_plan'
    out.mkdir(parents=True,exist_ok=True)
    aligned.to_parquet(out/'candidate_hours.parquet',index=False)
    required.to_parquet(out/'required_weather_times.parquet',index=False)
    nominal_dates=set(required.weather_time_utc.dt.strftime('%Y-%m-%d'))
    source_times=pd.concat([required.source_left_utc,required.source_right_utc]).drop_duplicates().sort_values()
    source_dates=set(source_times.dt.strftime('%Y-%m-%d'))
    old_catalog=json.loads((BASE/'catalog/coverage.json').read_text('utf-8'))
    catalog_dates=set(old_catalog['all_products_metadata_dates'])
    report=dict(status='support_plan_only_not_downloaded_or_trained',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        candidate_count=len(aligned),changed_origins=int(aligned.support_alignment_changed.sum()),
        changed_by_partition=aligned.groupby('partition').support_alignment_changed.sum().astype(int).to_dict(),
        unique_nominal_instants=len(required),nominal_dates=len(nominal_dates),
        averaged_source_instants=len(source_times),averaged_source_dates=len(source_dates),
        source_dates_not_in_prior_catalog=sorted(source_dates-catalog_dates),
        all_support_ends_at_or_before_origin=bool(aligned.support_end_utc.le(aligned.timestamp).all()),
        operational_availability_verified=False,full_preprocessing_verified=False,
        basis='Conservative one-hour support end for centered hourly source averages; no midnight clamping.',
        limitations=['Old weather and HF metadata columns retained only as original inventory provenance.',
                     'MERRA-2 publication latency is not represented by this physical support alignment.',
                     'New manifest does not prove file availability, unit compatibility, static or climatology coverage.',
                     'No existing pilot, model, split or website output changed.'])
    (out/'coverage.json').write_text(json.dumps(report,indent=2,allow_nan=False),'utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
