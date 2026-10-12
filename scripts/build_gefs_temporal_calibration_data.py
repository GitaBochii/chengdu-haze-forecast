"""Pre-2022 fit with audited 2021 forecast support; frozen 2022 predictions."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.audit_aft_signed_errors import summarize
from scripts.benchmark_gefs_72h_aft import verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, sha


def temporal_split(rows, full):
    start, end = [pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai') for year in (2022, 2023)]
    def crossing(cut):
        return set(full.loc[full.timestamp < cut, 'continuity_block']) & set(
            full.loc[full.timestamp >= cut, 'continuity_block'])
    first, last = crossing(start), crossing(end)
    train = rows.loc[rows.timestamp.lt(start) & rows.label_available_at.lt(start)
                     & ~rows.continuity_block.isin(first)].copy()
    val = rows.loc[rows.timestamp.ge(start) & rows.timestamp.lt(end)].copy()
    val['score_allowed'] = val.label_available_at.lt(end) & ~val.continuity_block.isin(first | last)
    if train.empty or not val.score_allowed.any():
        raise ValueError('Empty temporal split')
    if set(train.continuity_block) & set(val.continuity_block):
        raise ValueError('Training and inference blocks overlap')
    return train, val


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_2021_extension_v1'
    out = base / 'gefs_temporal_calibration_data_v1'
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((source / 'report.json').read_bytes())
    protocol = json.loads((source / 'protocol.json').read_bytes())
    inputs = dict(protocol['input_sha256'])
    inputs.update({str(source / name): value for name, value in report['output_sha256'].items()})
    paths = [Path(__file__), source / 'report.json',
             base / 'observed_business_dataset_v1/features.parquet',
             base / 'aft_rounds_temporal_v1/2022_predictions.parquet',
             ROOT / 'scripts/audit_aft_signed_errors.py',
             ROOT / 'src/haze_forecast/observed_aft.py',
             ROOT / 'src/haze_forecast/training.py', ROOT / 'src/haze_forecast/app.py']
    inputs.update({str(p): sha(p) for p in paths})
    verify_hashes(inputs)
    names = protocol['features']
    rows = pd.read_parquet(source / 'training_rows.parquet')
    full = pd.read_parquet(base / 'observed_business_dataset_v1/features.parquet')
    train, val = temporal_split(rows, full)
    previous = pd.read_parquet(base / 'aft_rounds_temporal_v1/2022_predictions.parquet')
    pd.testing.assert_series_equal(val.timestamp.reset_index(drop=True), previous.timestamp.reset_index(drop=True))
    pd.testing.assert_series_equal(val.score_allowed.reset_index(drop=True), previous.score_allowed.reset_index(drop=True))
    if len(names) != 104 or train.gefs_mean_temperature_c_f024.notna().sum() != 498:
        raise ValueError('Unexpected forecast support or feature schema')
    out.mkdir()
    settings = {
        'input_sha256': inputs, 'features': names, 'seed': 42, 'rounds': 500,
        'train_cutoff': '2022-01-01 Asia/Shanghai', 'score_end': '2023-01-01 Asia/Shanghai',
        'selection': 'None; create historical out-of-time predictions before choosing calibration.',
        'limitations': ['2022 historical diagnostic, not untouched test',
                        '2021 training forecast support differs from later refits',
                        'No calibration or deployment selected by this artifact'],
    }
    (out / 'protocol.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    train.to_parquet(out / 'training_rows.parquet', index=False)
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
        train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    val['forecast_aft'] = model.predict(val[names])
    np.testing.assert_array_equal(val.forecast_aft, joblib.load(out / 'model.joblib').predict(val[names]))
    val.to_parquet(out / 'predictions.parquet', index=False)
    scores = summarize(val.loc[val.score_allowed], 'forecast_aft')
    verify_hashes(inputs)
    result = {
        'training_rows': len(train), 'training_blocks': int(train.continuity_block.nunique()),
        'inference_rows': len(val), 'scoring_rows': int(val.score_allowed.sum()),
        'scores': scores, 'deployment_authorized': False,
        'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
    }
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
