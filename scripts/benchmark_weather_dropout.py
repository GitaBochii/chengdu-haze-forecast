"""Fixed training-only weather masking with a matched duplication control."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from haze_forecast.weather_source_ablation import WEATHER_FIELDS
from scripts.benchmark_gefs_72h_aft import compare_cohort, verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, sha


def augmented_training(train, names, *, masked):
    fields = [name for name in names if name in WEATHER_FIELDS]
    if not fields or not train.index.is_unique:
        raise ValueError('Invalid training schema or index')
    copy = train.copy()
    if masked:
        copy[fields] = np.nan
    untouched = [name for name in train if name not in fields]
    pd.testing.assert_frame_equal(copy[untouched], train[untouched])
    result = pd.concat([train, copy], ignore_index=True)
    weights = np.r_[np.full(len(train), 0.8), np.full(len(train), 0.2)]
    np.testing.assert_allclose(weights[:len(train)] + weights[len(train):], 1)
    return result, weights


def main():
    started = time.monotonic()
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_2021_extension_v1'
    old = base / 'reusable_gefs_aft_v1'
    out = base / 'weather_dropout_development_v1'
    if out.exists():
        raise FileExistsError(out)
    inputs = {}
    for folder in (source, old):
        report = json.loads((folder / 'report.json').read_bytes())
        for name, value in report['output_sha256'].items():
            if Path(name).name != name:
                raise ValueError('Invalid artifact name')
            inputs[str(folder / name)] = value
        inputs[str(folder / 'report.json')] = sha(folder / 'report.json')
    verify_hashes(inputs)
    previous = json.loads((old / 'protocol.json').read_bytes())
    names = previous['features']
    for name in ('src/haze_forecast/observed_aft.py', 'src/haze_forecast/training.py'):
        path = ROOT / name
        expected = previous['input_sha256'].get(str(path))
        if expected and sha(path) != expected:
            raise ValueError(f'Changed frozen model source: {path}')
    protected = [Path(__file__).resolve(), ROOT / 'src/haze_forecast/observed_aft.py',
                 ROOT / 'src/haze_forecast/training.py',
                 ROOT / 'src/haze_forecast/weather_source_ablation.py',
                 ROOT / 'scripts/benchmark_gefs_72h_aft.py',
                 ROOT / 'scripts/benchmark_observed_business_aft.py',
                 ROOT / 'src/haze_forecast/app.py', ROOT / 'src/haze_forecast/workbench_ui.py']
    protected += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    inputs.update({str(path): sha(path) for path in protected})
    train = pd.read_parquet(source / 'f024_training_rows.parquet')
    val = pd.read_parquet(source / 'inference_rows.parquet')
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    if not (train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()):
        raise ValueError('Training crosses cutoff')
    frames = {key: pd.read_parquet(old / filename) for key, filename in
              [('original', 'predictions.parquet'), ('expanded', 'expanded_predictions.parquet')]}
    baseline = joblib.load(old / 'model.joblib').predict(val[names])
    for frame in frames.values():
        pd.testing.assert_series_equal(frame.timestamp, val.timestamp)
        np.testing.assert_array_equal(baseline, frame.forecast_aft)
        if set(train.continuity_block) & set(frame.continuity_block):
            raise ValueError('Block overlap')
    out.mkdir()
    protocol = {'input_sha256': inputs, 'features': names, 'training_rows': len(train),
                'inference_rows': len(val), 'mask_fields': [n for n in names if n in WEATHER_FIELDS],
                'weights': [0.8, 0.2], 'rounds': 500, 'seed': 42,
                'arms': ['duplicate_control', 'weather_masked'],
                'selection': 'Fixed hypothesis; no 2024/2025 evaluation or weight tuning',
                'gate': 'Both 2023 cohorts: row and block MAE improve versus frozen and duplicate controls; tail no worse',
                'training_support': 'Original f024, without additional 2021 forecasts',
                'limitations': ['Reused development year; no untouched holdout',
                                'No new real-time availability evidence for existing inputs',
                                'Masking derived fields is a training augmentation, not a physical missing-hour simulation'],
                'deployment_authorized': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    val.to_parquet(out / 'inference.parquet', index=False)
    scores = {}
    for arm in protocol['arms']:
        tr, weights = augmented_training(train, names, masked=arm == 'weather_masked')
        tr.to_parquet(out / f'{arm}_training.parquet', index=False)
        np.save(out / f'{arm}_weights.npy', weights)
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
            tr[names], tr.lower_bound_hours, tr.upper_bound_hours, sample_weight=weights)
        path = out / f'{arm}.joblib'
        joblib.dump(model, path)
        prediction = model.predict(val[names])
        np.testing.assert_array_equal(prediction, joblib.load(path).predict(val[names]))
        scores[arm] = {}
        for key, frame in frames.items():
            scores[arm][key] = compare_cohort(frame, prediction)
            saved = frame.copy()
            saved['dropout_candidate'] = prediction
            saved.to_parquet(out / f'{arm}_{key}.parquet', index=False)
    checks = {}
    for key in frames:
        masked = scores['weather_masked'][key]
        dup = scores['duplicate_control'][key]['candidate']
        candidate = masked['candidate']
        checks[key] = bool(masked['passed'] and candidate['row_mae'] < dup['row_mae']
                           and candidate['block_mae'] < dup['block_mae']
                           and candidate['tail_24h_block_mae'] <= dup['tail_24h_block_mae'])
    verify_hashes(inputs)
    report = {'scores': scores, 'checks': checks, 'passed': all(checks.values()),
              'deployment_authorized': False, 'elapsed_seconds': time.monotonic() - started,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
