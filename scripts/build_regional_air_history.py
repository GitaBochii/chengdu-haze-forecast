"""Build and verify the isolated 2014-2026 unfilled historical pollution panel."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from haze_forecast.regional_air_data import load_station_metadata
from haze_forecast.regional_air_history import (
    POLLUTANTS,
    aggregate_history_city_hourly,
    complete_history_hours,
    history_coverage_report,
    normalize_history_interim,
    parse_history_daily,
)


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def guard_outputs(output: Path, audit: Path) -> None:
    targets = [output / name for name in
               ['station_hourly.parquet', 'city_hourly.parquet', 'station_metadata.csv']]
    targets.append(audit / 'report.json')
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f'refusing to overwrite existing historical artifacts: {existing}')


def read_daily_source(payload: bytes, name: str, metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    needed = {'date', 'hour', 'type', *metadata.station_id}
    raw = pd.read_csv(io.BytesIO(payload), usecols=lambda column: column in needed)
    frame = parse_history_daily(raw, metadata)
    if frame.empty:
        raise ValueError(f'no selected pollution observations: {name}')
    dates = frame.timestamp.dt.strftime('%Y%m%d').unique().tolist()
    if len(dates) != 1 or dates[0] not in Path(name).name:
        raise ValueError(f'source date does not match daily filename: {name}')
    report = {
        'path': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
        'date': dates[0], 'raw_rows': len(raw), 'panel_rows': len(frame),
        'ozone_type_present': bool(raw.type.eq('O3_8h').any()),
        'ozone_type_hour_count': int(raw.loc[raw.type.eq('O3_8h'), 'hour'].nunique()),
        'observed_hours': {key: int(frame[key].notna().sum()) for key in POLLUTANTS},
    }
    return frame, report


def compare_city_baseline(city: pd.DataFrame, path: Path) -> dict:
    baseline = pd.read_parquet(path).set_index('timestamp').sort_index()
    keys = ['aqi', 'pm2_5', 'pm10']
    if baseline.index.has_duplicates:
        raise ValueError(f'duplicate baseline timestamps: {path}')
    current = city.set_index('timestamp').sort_index()
    if not baseline.index.isin(current.index).all():
        raise AssertionError(f'baseline timestamps outside panel: {path}')
    pd.testing.assert_frame_equal(current.loc[baseline.index, keys], baseline[keys],
                                  check_dtype=False, check_freq=False, check_exact=True)
    # Hours absent from old sparse tables must still be unobserved, not invented.
    extra = current.loc[current.index.to_series().between(baseline.index.min(), baseline.index.max())]
    extra = extra.loc[~extra.index.isin(baseline.index), keys]
    if extra.notna().any().any():
        raise AssertionError(f'new values at absent baseline timestamps: {path}')
    return {**file_record(path), 'equal': True, 'columns': keys, 'compared_hours': len(baseline),
            'extra_missing_grid_hours': len(extra)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/regional_air_models_full_history'))
    parser.add_argument('--audit', type=Path, default=Path('artifacts/regional_air_models/full_history_data_audit'))
    args = parser.parse_args()
    guard_outputs(args.output, args.audit)
    root = args.root
    metadata_path = root / 'data/raw/data/chengdu/stations.json'
    metadata = load_station_metadata(metadata_path)
    interim_path = root / 'data/interim/station_hourly.parquet'
    interim = normalize_history_interim(pd.read_parquet(interim_path), metadata)
    if not interim.timestamp.dt.year.between(2015, 2021).all():
        raise ValueError('historical interim must contain only 2015-2021')
    frames = [interim]
    source_files = [file_record(interim_path)]
    daily_reports = []
    seen_dates = set()

    def append_day(payload: bytes, name: str) -> None:
        frame, report = read_daily_source(payload, name, metadata)
        if report['date'] in seen_dates:
            raise ValueError(f'duplicate source date: {report["date"]}')
        seen_dates.add(report['date'])
        frames.append(frame)
        daily_reports.append(report)
        if len(daily_reports) % 100 == 0:
            print(f'parsed {len(daily_reports)} daily files', flush=True)

    for filename in ['20140513-20141231_CNEMC.zip', '20220101-20221231_CNEMC.zip']:
        path = root / 'data/raw' / filename
        source_files.append(file_record(path))
        with ZipFile(path) as archive:
            names = sorted(name for name in archive.namelist()
                           if Path(name).name.startswith('china_sites_') and name.endswith('.csv'))
            if not names:
                raise ValueError(f'no daily sources in {path}')
            for name in names:
                with archive.open(name) as handle:
                    append_day(handle.read(), f'{path.resolve()}::{name}')
    for directory in ['cnemc_2023_daily', 'cnemc_2024_2025_daily', 'cnemc_2026_daily']:
        paths = sorted((root / 'data/raw' / directory).glob('china_sites_????????.csv'))
        if not paths:
            raise ValueError(f'no daily sources in {directory}')
        for path in paths:
            append_day(path.read_bytes(), str(path.resolve()))
    observed = pd.concat(frames, ignore_index=True)
    station = complete_history_hours(observed, metadata, '2014-05-13', '2026-09-03')
    city = aggregate_history_city_hourly(station)
    comparisons = [compare_city_baseline(city, root / path) for path in [
        'data/interim/cnemc_chengdu_2014_hourly.parquet',
        'data/interim/city_hourly.parquet',
        'data/interim/cnemc_chengdu_2022_hourly.parquet',
        'data/processed/regional_air_models/city_hourly.parquet',
    ]]
    args.output.mkdir(parents=True, exist_ok=True)
    args.audit.mkdir(parents=True, exist_ok=True)
    station.to_parquet(args.output / 'station_hourly.parquet', index=False)
    city.to_parquet(args.output / 'city_hourly.parquet', index=False)
    metadata.to_csv(args.output / 'station_metadata.csv', index=False)
    report = {
        'start': station.timestamp.min().isoformat(), 'end': station.timestamp.max().isoformat(),
        'timezone': 'Asia/Shanghai', 'imputation': 'none',
        'ozone_mapping': {'interim_o3': 'O3-8h', 'CNEMC': 'O3_8h', 'output': 'o3_8h'},
        'excluded_ozone_types': ['O3', 'O3_24h', 'O3_8h_24h'],
        'city_aggregation': 'median across all available station observations per pollutant',
        'station_rows': len(station), 'city_rows': len(city), 'station_count': len(metadata),
        'metadata': file_record(metadata_path), 'source_files': source_files,
        'workbook_provenance': [file_record(path) for path in sorted((root / 'data/raw/data/chengdu').glob('*.xlsx'))],
        'daily_sources': daily_reports, 'daily_file_count': len(daily_reports),
        'days_missing_exact_ozone_type': [row['path'] for row in daily_reports if not row['ozone_type_present']],
        'station_coverage': history_coverage_report(station, 'station_id'),
        'city_coverage': history_coverage_report(city, 'city'),
        'source_observed_coverage': history_coverage_report(observed, 'station_id'),
        'baseline_comparisons': comparisons,
        'limits': [
            'Static station metadata does not establish historical relocation.',
            'Legacy workbook preprocessing before ingestion is not independently established.',
            '2015-2021 interim was audited against local workbooks in history_coverage_20260905.md.',
            'No historical weather is carried from these sources; weather is a separate artifact.',
            '2014 and 2026 are partial calendar years; missing observations remain NaN.',
            'City AQI is station-median AQI, not official city AQI.',
        ],
        'outputs': [file_record(args.output / name) for name in
                    ['station_hourly.parquet', 'city_hourly.parquet', 'station_metadata.csv']],
    }
    (args.audit / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['start', 'end', 'station_rows', 'city_rows', 'daily_file_count']}), flush=True)


if __name__ == '__main__':
    main()
