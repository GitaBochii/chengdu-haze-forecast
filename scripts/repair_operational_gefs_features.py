"""Serial, bounded GEFS retry extraction into new immutable per-cycle shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import requests

from haze_forecast.operational_gefs import (
    download_operational_gefs_surface_messages,
    select_operational_gefs_run,
)
from haze_forecast.operational_gefs_dataset import (
    OPERATIONAL_GEFS_FEATURES,
    enrich_event_rows_with_operational_gefs,
)
from haze_forecast.operational_gefs_quality import consolidate_verified_shards


def _keys(frame: pd.DataFrame) -> set[tuple[pd.Timestamp, int]]:
    return set(zip(pd.to_datetime(frame.timestamp, utc=True), frame.event_id.astype(int), strict=True))


def repair_year(
    expected: pd.DataFrame, frames: list[pd.DataFrame], *, year: int,
    output_dir: Path, manifest_dir: Path, extractor, sleeper=time.sleep,
    max_attempts: int = 3, backoff_seconds: float = 5,
) -> dict:
    if max_attempts not in (1, 2, 3):
        raise ValueError('max attempts must be between 1 and 3')
    if not np.isfinite(backoff_seconds) or backoff_seconds < 0:
        raise ValueError('backoff must be finite and non-negative')
    expected = expected.copy()
    expected['timestamp'] = pd.to_datetime(expected.timestamp, utc=True, errors='raise')
    if expected.empty or expected.timestamp.isna().any() or expected.duplicated(['timestamp', 'event_id']).any():
        raise ValueError('expected keys must be nonempty, valid and unique')
    combined = consolidate_verified_shards(frames)
    expected_keys = _keys(expected)
    verified_keys = _keys(combined.loc[combined.gefs_row_verified.astype(bool)])
    missing = expected_keys - verified_keys
    requested = expected.loc[[key in missing for key in zip(expected.timestamp, expected.event_id, strict=True)]].copy()
    requested['repair_cycle_utc'] = [select_operational_gefs_run(stamp.to_pydatetime())[0]
                                     for stamp in requested.timestamp]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    session = uuid4().hex
    reports, successful, failed = [], [], []
    for run, group in requested.groupby('repair_cycle_utc', sort=True):
        rows = group.drop(columns='repair_cycle_utc').reset_index(drop=True)
        wanted = _keys(rows)
        for attempt in range(1, max_attempts + 1):
            stem = f'operational_gefs_event_features_{year}_repair_{run:%Y%m%d%H}_{session}_{attempt}'
            shard_path, manifest_path = output_dir / f'{stem}.parquet', manifest_dir / f'{stem}.json'
            if shard_path.exists() or manifest_path.exists():
                raise FileExistsError(f'refusing to overwrite repair artifact: {stem}')
            try:
                enriched = extractor(rows.copy())
            except (OSError, RuntimeError, ValueError, requests.RequestException) as error:
                enriched = rows.copy()
                enriched[list(OPERATIONAL_GEFS_FEATURES)] = np.nan
                enriched['gefs_source_manifest_json'] = None
                enriched['gefs_extraction_error'] = str(error)
            if _keys(enriched) != wanted or enriched.duplicated(['timestamp', 'event_id']).any():
                raise ValueError('repair extractor must return exactly the requested unique keys')
            checked = consolidate_verified_shards([enriched])
            # Check conflicts before accepting this attempt as a replacement.
            combined = consolidate_verified_shards([combined, checked])
            checked.to_parquet(shard_path, index=False)
            passed = bool(checked.gefs_row_verified.all())
            entry = {
                'year': year, 'cycle_utc': pd.Timestamp(run).isoformat(), 'attempt': attempt,
                'requested_keys': len(rows), 'verified_keys': int(checked.gefs_row_verified.sum()),
                'success': passed, 'shard': str(shard_path.resolve()),
                'sha256': hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                'quality_errors': checked.loc[~checked.gefs_row_verified, 'gefs_quality_error'].tolist(),
                'publication_availability': 'not established by extraction quality',
            }
            with manifest_path.open('x', encoding='utf-8') as handle:
                json.dump(entry, handle, indent=2)
            reports.append(entry)
            print(f'{year} {run} attempt {attempt}: {entry["verified_keys"]}/{len(rows)} verified', flush=True)
            if passed:
                successful.append(pd.Timestamp(run).isoformat())
                break
            if attempt < max_attempts:
                sleeper(backoff_seconds * 2 ** (attempt - 1))
        else:
            failed.append(pd.Timestamp(run).isoformat())
    final_keys = _keys(combined.loc[combined.gefs_row_verified.astype(bool)])
    report = {
        'year': year, 'expected_keys': len(expected_keys),
        'initial_verified_keys': len(expected_keys & verified_keys),
        'verified_keys': len(expected_keys & final_keys),
        'missing_or_failed_keys': len(expected_keys - final_keys),
        'successful_runs': successful, 'failed_runs': failed, 'attempts': reports,
        'status': 'complete' if expected_keys.issubset(final_keys) else 'incomplete',
        'max_attempts': max_attempts, 'serial_cycles': True,
    }
    report_path = manifest_dir / f'repair_{year}_{session}.json'
    with report_path.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    report['report_path'] = str(report_path.resolve())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', required=True, type=int)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--input-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--manifest-dir', type=Path, default=Path('artifacts/operational_gefs_ensemble/repair'))
    parser.add_argument('--max-attempts', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--backoff-seconds', type=float, default=5)
    parser.add_argument('--longitude', type=float, default=104.0665)
    parser.add_argument('--latitude', type=float, default=30.5723)
    parser.add_argument('--wgrib2', type=Path, default=Path('tools/wgrib2/v3.1.3/wgrib2.exe'))
    args = parser.parse_args()
    source = args.input or args.input_dir / f'cnemc_chengdu_{args.year}_training_features.parquet'
    expected = pd.read_parquet(source)
    timestamp = pd.to_datetime(expected.timestamp, utc=True, errors='raise')
    expected = expected.loc[timestamp.dt.tz_convert('Asia/Shanghai').dt.year.eq(args.year)
                            & expected.event_id.gt(0) & expected.remaining_hours.notna()
                            & expected.particle_primary.fillna(False)].copy()
    paths = sorted(set(args.input_dir.glob(f'operational_gefs_event_features_{args.year}_*.parquet'))
                   | set(args.output_dir.glob(f'operational_gefs_event_features_{args.year}_*.parquet')))
    extractor = partial(enrich_event_rows_with_operational_gefs,
                        longitude=args.longitude, latitude=args.latitude,
                        wgrib2_executable=args.wgrib2,
                        downloader=partial(download_operational_gefs_surface_messages, max_workers=1))
    report = repair_year(expected, [pd.read_parquet(path) for path in paths], year=args.year,
                         output_dir=args.output_dir, manifest_dir=args.manifest_dir,
                         extractor=extractor, max_attempts=args.max_attempts,
                         backoff_seconds=args.backoff_seconds)
    print(json.dumps({key: value for key, value in report.items() if key != 'attempts'}, indent=2))
    if report['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
