"""Matched offline weather diagnostic on frozen current interval/GEFS inputs."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from haze_forecast.weather_source_ablation import WEATHER_FIELDS, uniform_weather_rows
from scripts.benchmark_gefs_72h_aft import compare_cohort, verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, sha


def weather_arms(original, proxy, names):
    fields = [name for name in names if name in WEATHER_FIELDS]
    rebuilt = uniform_weather_rows(original, proxy)
    uniform = original.copy()
    uniform[fields] = rebuilt[fields]
    masked = uniform.copy()
    masked[fields] = masked[fields].where(original[fields].notna())
    untouched = [name for name in original if name not in fields]
    for frame in (uniform, masked):
        pd.testing.assert_frame_equal(frame[untouched], original[untouched])
    pd.testing.assert_frame_equal(masked[fields].isna(), original[fields].isna())
    return {'uniform': uniform, 'masked': masked}


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_2021_extension_v1'
    old = base / 'reusable_gefs_aft_v1'
    out = base / 'interval_weather_triplet_v1'
    weather_path = ROOT / 'data/processed/regional_air_models_full_history/weather_proxy.parquet'
    delivery = ROOT / 'mision/20260908_task02_final_delivery/audit.json'
    if out.exists():
        raise FileExistsError(out)
    expected = json.loads(delivery.read_bytes())['key_input_sha256']['weather_proxy.parquet']
    if sha(weather_path) != expected:
        raise ValueError('Weather differs from Task02 delivered fingerprint')
    paths = [Path(__file__), weather_path, delivery,
             ROOT / 'src/haze_forecast/weather_source_ablation.py', ROOT / 'src/haze_forecast/features.py']
    inputs = {str(p): sha(p) for p in paths}
    for folder in (source, old):
        report = json.loads((folder / 'report.json').read_bytes())
        inputs.update({str(folder / n): h for n, h in report['output_sha256'].items()})
        inputs[str(folder / 'report.json')] = sha(folder / 'report.json')
        inputs.update(json.loads((folder / 'protocol.json').read_bytes())['input_sha256'])
    verify_hashes(inputs)
    names = json.loads((old / 'protocol.json').read_bytes())['features']
    train = pd.read_parquet(source / 'f024_training_rows.parquet')
    val = pd.read_parquet(source / 'inference_rows.parquet')
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    if not (train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()):
        raise ValueError('Training labels not mature before development')
    frames = {key: pd.read_parquet(old / filename) for key, filename in
              [('original', 'predictions.parquet'), ('expanded', 'expanded_predictions.parquet')]}
    reference = joblib.load(old / 'model.joblib').predict(val[names])
    for frame in frames.values():
        pd.testing.assert_series_equal(frame.timestamp, val.timestamp)
        np.testing.assert_array_equal(reference, frame.forecast_aft)
        if set(train.continuity_block) & set(frame.continuity_block):
            raise ValueError('Training and validation blocks overlap')
    proxy = pd.read_parquet(weather_path)
    training = weather_arms(train, proxy, names)
    inference = weather_arms(val, proxy, names)
    fields = [name for name in names if name in WEATHER_FIELDS]
    out.mkdir()
    protocol = {'input_sha256': inputs, 'features': names, 'replaced_fields': fields,
                'training_rows': len(train), 'inference_rows': len(val),
                'arms': ['frozen_original', 'uniform', 'masked'],
                'mask_rule': 'Restore original missingness per derived model field, not raw-hour masking',
                'unchanged': 'Labels, cohorts, historical PBL, GEFS, model settings and nonweather features',
                'selection': 'offline diagnostic only; no deployment selection',
                'limitations': ['Retrospective ERA5, not real-time input',
                                'Value/source changes include spatial representativeness',
                                '2023 reused development data; no all-year conclusion'],
                'deployment_authorized': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    results, coverage = {}, {}
    for arm in ('uniform', 'masked'):
        tr, va = training[arm], inference[arm]
        tr.to_parquet(out / f'{arm}_training.parquet', index=False)
        va.to_parquet(out / f'{arm}_inference.parquet', index=False)
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
            tr[names], tr.lower_bound_hours, tr.upper_bound_hours)
        joblib.dump(model, out / f'{arm}.joblib')
        prediction = model.predict(va[names])
        results[arm] = {}
        for key, original in frames.items():
            results[arm][key] = compare_cohort(original, prediction)
            saved = original.copy()
            saved['weather_candidate'] = prediction
            saved.to_parquet(out / f'{arm}_{key}.parquet', index=False)
        coverage[arm] = {'train_nonnull': float(tr[fields].notna().to_numpy().mean()),
                         'inference_nonnull': float(va[fields].notna().to_numpy().mean())}
    verify_hashes(inputs)
    report = {'results': results, 'coverage': coverage, 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
