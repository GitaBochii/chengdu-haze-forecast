"""Frozen annual weather ablation; offline evidence, never deployment selection."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_gefs_72h_aft import verify_hashes
from scripts.benchmark_interval_weather_triplet import weather_arms
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha
from scripts.benchmark_original_gefs_without_pbl_crossyear import validate_split


def aligned_predictions(frame, times, prediction):
    if times.duplicated().any() or frame.timestamp.duplicated().any():
        raise ValueError('Duplicate timestamp in inference or cohort')
    return pd.Series(prediction, index=times).loc[frame.timestamp].to_numpy()


def main():
    started = time.monotonic()
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'interval_weather_crossyear_v1'
    if out.exists():
        raise FileExistsError(out)
    old = base / 'reusable_gefs_crossyear_v1'
    source = base / 'original_gefs_without_pbl_crossyear_v1'
    dev = base / 'interval_weather_triplet_v1'
    inputs = {}
    for folder in (old, source, dev):
        report = json.loads((folder / 'report.json').read_bytes())
        for name, fingerprint in report['output_sha256'].items():
            if Path(name).name != name:
                raise ValueError('Invalid artifact filename')
            inputs[str(folder / name)] = fingerprint
        inputs[str(folder / 'report.json')] = sha(folder / 'report.json')
    verify_hashes(inputs)
    protocol = json.loads((old / 'protocol.json').read_bytes())
    dev_protocol = json.loads((dev / 'protocol.json').read_bytes())
    names = protocol['features']
    if names != dev_protocol['features'] or len(names) != 104:
        raise ValueError('Development feature schema changed')
    weather = ROOT / 'data/processed/regional_air_models_full_history/weather_proxy.parquet'
    expected = dev_protocol['input_sha256'][str(weather)]
    if sha(weather) != expected:
        raise ValueError('Weather no longer matches frozen development input')
    inputs[str(weather)] = expected
    dependencies = ['src/haze_forecast/observed_aft.py', 'src/haze_forecast/training.py',
                    'src/haze_forecast/weather_source_ablation.py', 'src/haze_forecast/features.py',
                    'scripts/benchmark_interval_weather_triplet.py',
                    'scripts/benchmark_gefs_72h_aft.py',
                    'scripts/benchmark_original_gefs_without_pbl_crossyear.py',
                    'scripts/benchmark_observed_business_aft.py']
    for name in dependencies:
        path = ROOT / name
        # Check frozen scientific dependencies when their source manifest records them.
        expected = dev_protocol['input_sha256'].get(str(path), protocol['input_sha256'].get(str(path)))
        if expected is not None and sha(path) != expected:
            raise ValueError(f'Scientific dependency changed: {path}')
        inputs[str(path)] = sha(path)
    inputs[str(Path(__file__).resolve())] = sha(Path(__file__))
    protected = [ROOT / 'src/haze_forecast/app.py', ROOT / 'src/haze_forecast/workbench_ui.py']
    protected += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    inputs.update({str(path): sha(path) for path in protected})
    proxy = pd.read_parquet(weather)
    prepared = {}
    for year in (2024, 2025):
        train = pd.read_parquet(source / f'{year}_training_rows.parquet')
        x = pd.read_parquet(old / f'{year}_features.parquet')
        cohorts = {key: pd.read_parquet(old / f'{year}_{key}.parquet')
                   for key in ('predictions', 'expanded', 'common')}
        val = cohorts['predictions']
        validate_split(train, val, year)
        if len(x) != len(val):
            raise ValueError('Inference rows differ')
        x = x.copy()
        x['timestamp'] = val.timestamp.to_numpy()
        baseline = joblib.load(old / f'{year}.joblib').predict(x[names])
        for frame in cohorts.values():
            np.testing.assert_array_equal(aligned_predictions(frame, val.timestamp, baseline),
                                          frame.forecast_aft.to_numpy())
        prepared[year] = (train, x, cohorts)
    out.mkdir()
    frozen = {'input_sha256': inputs, 'features': names, 'replaced_fields': dev_protocol['replaced_fields'],
              'years': [2024, 2025], 'rounds': 500, 'seed': 42,
              'arms': ['original', 'uniform', 'masked'],
              'training': 'Same annual frozen rows in all arms; no additional 2021 forecasts',
              'selection': 'No tuning; compare every frozen original, expanded and common cohort',
              'gate': 'Each year/common: all 3 MAEs below original and webpage; offline evidence only',
              'limitations': ['ERA5 retrospective; no publication-time evidence',
                              'Reused evaluation years, not untouched holdout',
                              '2023 development had additional 2021 training forecasts; annual arms do not',
                              'Frozen artifact hashes verified; historical UI source hash intentionally not replayed'],
              'deployment_authorized': False}
    (out / 'protocol.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    results = {}
    for year, (train, x, cohorts) in prepared.items():
        training = {'original': train, **weather_arms(train, proxy, names)}
        inference = {'original': x, **weather_arms(x, proxy, names)}
        scores, coverage = {}, {}
        for arm in ('original', 'uniform', 'masked'):
            tr, va = training[arm], inference[arm]
            model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
                tr[names], tr.lower_bound_hours, tr.upper_bound_hours)
            prediction = model.predict(va[names])
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite model output')
            if arm == 'original':
                np.testing.assert_array_equal(prediction, cohorts['predictions'].forecast_aft.to_numpy())
            checkpoint = out / f'{year}_{arm}.joblib'
            joblib.dump(model, checkpoint)
            np.testing.assert_array_equal(prediction, joblib.load(checkpoint).predict(va[names]))
            tr.to_parquet(out / f'{year}_{arm}_training.parquet', index=False)
            va.to_parquet(out / f'{year}_{arm}_inference.parquet', index=False)
            scores[arm] = {}
            for key, frame in cohorts.items():
                saved = frame.copy()
                saved['weather_candidate'] = aligned_predictions(frame, x.timestamp, prediction)
                scores[arm][key] = metrics(saved, saved.weather_candidate.to_numpy())
                saved.to_parquet(out / f'{year}_{arm}_{key}.parquet', index=False)
            fields = frozen['replaced_fields']
            coverage[arm] = {'train': float(tr[fields].notna().to_numpy().mean()),
                             'inference': float(va[fields].notna().to_numpy().mean())}
        page = metrics(cohorts['common'], cohorts['common'].strict_prediction_hours.to_numpy())
        keys = ('row_mae', 'block_mae', 'tail_24h_block_mae')
        checks = {arm: all(scores[arm]['common'][key] < ref[key]
                           for ref in (scores['original']['common'], page) for key in keys)
                  for arm in ('uniform', 'masked')}
        results[str(year)] = {'scores': scores, 'page_common': page, 'coverage': coverage,
                              'common_gate': checks, 'training_rows': len(train), 'inference_rows': len(x)}
        print(json.dumps({str(year): results[str(year)]}), flush=True)
    verify_hashes(inputs)
    report = {'years': results, 'elapsed_seconds': time.monotonic() - started,
              'deployment_authorized': False, 'original_retraining_exact_match': True,
              'checkpoint_replays_passed': True,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Completed in {report["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
