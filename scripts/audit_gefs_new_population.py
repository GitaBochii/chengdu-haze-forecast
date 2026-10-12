"""Check reusable audited forecast coverage on the corrected business population."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.operational_gefs_benchmark import publication_audited_features
from haze_forecast.operational_gefs_dataset import OPERATIONAL_GEFS_FEATURES
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'gefs_new_population_audit_v1'
    if out.exists():
        raise FileExistsError(out)
    old_report_path = ROOT / 'artifacts/operational_gefs_ensemble/report.json'
    old_report = json.loads(old_report_path.read_bytes())
    origin_path = base / 'interval_business_labels_v1/origins.parquet'
    train_path = base / 'interval_business_aft_v1/training_rows.parquet'
    paths = [origin_path, train_path, old_report_path, Path(__file__),
             ROOT / 'src/haze_forecast/operational_gefs_benchmark.py',
             ROOT / 'src/haze_forecast/app.py']
    files = {y: ROOT / f'data/processed/operational_gefs_audited/operational_gefs_event_features_{y}.parquet' for y in (2022, 2023, 2024, 2025)}
    paths += list(files.values())
    before = {str(p): sha(p) for p in paths}
    origins = pd.read_parquet(origin_path)
    reports, frames = {}, []
    for year, path in files.items():
        assert sha(path) == old_report['audited_gefs_sha256'][path.name]
        raw = pd.read_parquet(path)
        safe = publication_audited_features(raw)
        safe.timestamp = pd.to_datetime(safe.timestamp, utc=True)
        assert safe.timestamp.is_unique
        metadata = raw[['timestamp', 'gefs_forecast_valid_utc_f006', 'gefs_forecast_valid_utc_f024']].copy()
        metadata.timestamp = pd.to_datetime(metadata.timestamp, utc=True)
        safe = safe.merge(metadata, on='timestamp', validate='one_to_one')
        frame = origins.loc[origins.timestamp.dt.year.eq(year), ['timestamp', 'continuity_block']].merge(
            safe.drop(columns='event_id'), on='timestamp', how='left', validate='one_to_one', indicator=True)
        frame['any_audited_feature'] = frame[list(OPERATIONAL_GEFS_FEATURES)].notna().any(axis=1)
        stats = {'origins': len(frame), 'matched': int(frame._merge.eq('both').sum()),
                 'any_audited_feature': int(frame.any_audited_feature.sum())}
        for lead in (6, 24):
            fields = [n for n in OPERATIONAL_GEFS_FEATURES if n.endswith(f'f{lead:03d}')]
            valid = pd.to_datetime(frame[f'gefs_forecast_valid_utc_f{lead:03d}'], utc=True)
            future = valid.gt(frame.timestamp)
            frame[f'future_available_f{lead:03d}'] = frame[fields].notna().any(axis=1) & future
            stats[f'future_available_f{lead:03d}'] = int(frame[f'future_available_f{lead:03d}'].sum())
            # Feature labels refer to forecast initialization, not the user origin.
            for field in fields:
                frame.loc[~future, field] = np.nan
        reports[str(year)] = stats
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    train = pd.read_parquet(train_path)
    training = train[['timestamp']].merge(combined[['timestamp', 'future_available_f006', 'future_available_f024']],
                                        on='timestamp', how='left', validate='one_to_one')
    report = {'input_sha256': before, 'years': reports, 'pre2023_training_rows': len(train),
              'training_with_future_f024': int(training.future_available_f024.eq(True).sum()),
              'training_with_future_f006': int(training.future_available_f006.eq(True).sum()),
              'limitations': ['archive timestamp proxy retained; not historical first-publication certification',
                              'old extraction coverage does not imply corrected-origin coverage',
                              'no rows removed, no missing forecasts imputed, no training performed']}
    out.mkdir(parents=True)
    combined.to_parquet(out / 'future_only_features.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report['inputs_unchanged'] = True
    report['output_sha256'] = sha(out / 'future_only_features.parquet')
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
