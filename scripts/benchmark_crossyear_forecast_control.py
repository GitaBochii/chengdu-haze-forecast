"""Fixed metadata-only GEFS control on existing annual research cohorts."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_original_gefs_without_pbl_crossyear import validate_split
from scripts.evaluate_crossyear_scorecard import ROOT, canonical_cohort, sha


def metadata_only(frame, values):
    result = frame.copy(deep=True)
    result[values] = result[values].where(result[values].isna(), 0.)
    pd.testing.assert_frame_equal(result[values].isna(), frame[values].isna())
    pd.testing.assert_frame_equal(result.drop(columns=values), frame.drop(columns=values))
    return result


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'interval_weather_crossyear_v1'
    out = base / 'forecast_metadata_crossyear_20260916'
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads((source / 'protocol.json').read_bytes())
    source_report = json.loads((source / 'report.json').read_bytes())
    inputs = {}
    for name, expected in source_report['output_sha256'].items():
        if Path(name).name != name or sha(source / name) != expected:
            raise ValueError(f'Frozen source mismatch: {name}')
        inputs[str(source / name)] = expected
    names = protocol['features']
    values = names[92:101]
    if len(names) != 104 or len(values) != 9 or not all(n.startswith('gefs_') for n in values):
        raise ValueError('Unexpected forecast schema')
    label_path = base / 'interval_business_labels_v1/origins.parquet'
    if sha(label_path) != json.loads((label_path.parent / 'report.json').read_bytes())['output_sha256']:
        raise ValueError('Canonical labels changed')
    labels = pd.read_parquet(label_path)
    protected = list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    protected += [label_path, source / 'report.json', Path(__file__).resolve(),
                  ROOT / 'src/haze_forecast/observed_aft.py',
                  ROOT / 'src/haze_forecast/cohort_scorecard.py',
                  ROOT / 'scripts/evaluate_crossyear_scorecard.py',
                  ROOT / 'scripts/benchmark_original_gefs_without_pbl_crossyear.py']
    inputs.update({str(p): sha(p) for p in protected})
    prepared = {}
    for year in (2024, 2025):
        train = pd.read_parquet(source / f'{year}_original_training.parquet')
        x = pd.read_parquet(source / f'{year}_original_inference.parquet')
        saved = {key: pd.read_parquet(source / f'{year}_original_{key}.parquet')
                 for key in ('common', 'expanded')}
        validate_split(train, saved['expanded'], year)
        if not x.timestamp.is_unique:
            raise ValueError('Duplicate inference origin')
        full = joblib.load(source / f'{year}_original.joblib').predict(x[names])
        indexed = pd.Series(full, index=x.timestamp)
        cohorts = {}
        for key, frame in saved.items():
            cohort, stale = canonical_cohort(labels, frame, year)
            np.testing.assert_array_equal(indexed.loc[cohort.timestamp], frame.weather_candidate)
            cohorts[key] = (cohort, stale)
        prepared[year] = (train, x, indexed, cohorts)
    out.mkdir()
    manifest = {'input_sha256': inputs, 'features': names, 'zeroed_fields': values,
                'years': [2024, 2025], 'seed': 42, 'rounds': 500,
                'comparison': 'Same frozen annual training rows and bounds; numeric GEFS values zeroed, NaNs and ages unchanged',
                'selection': 'No tuning or model selection; both years and both fixed cohorts reported',
                'limitations': ['Reused historical years, not fresh tests',
                                'Constant placeholders are not meteorology and are not deployable',
                                'Removing splits also changes optimization; not a pure causal estimate',
                                'No new real-time availability evidence',
                                'Retrospective training labels retained exactly; no relabeling',
                                'Common and expanded cohorts overlap and must not be pooled'],
                'deployment_authorized': False}
    (out / 'protocol.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    results = {}
    for year, (train, x, indexed, cohorts) in prepared.items():
        control_train, control_x = metadata_only(train, values), metadata_only(x, values)
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
            control_train[names], train.lower_bound_hours, train.upper_bound_hours)
        prediction = model.predict(control_x[names])
        checkpoint = out / f'{year}_metadata.joblib'
        joblib.dump(model, checkpoint)
        np.testing.assert_array_equal(prediction, joblib.load(checkpoint).predict(control_x[names]))
        by_time = pd.Series(prediction, index=x.timestamp)
        present = train[values].notna().all(axis=1)
        support = {'training_rows': len(train), 'training_blocks': int(train.continuity_block.nunique()),
                   'complete_forecast_rows': int(present.sum()),
                   'complete_forecast_blocks': int(train.loc[present, 'continuity_block'].nunique())}
        results[str(year)] = {'support': support, 'cohorts': {}}
        for key, (cohort, stale) in cohorts.items():
            frame = cohort.copy()
            frame['full_forecast'] = indexed.loc[frame.timestamp].to_numpy()
            frame['metadata_only'] = by_time.loc[frame.timestamp].to_numpy()
            scores = {arm: score_predictions(frame, frame[arm].to_numpy())
                      for arm in ('full_forecast', 'metadata_only')}
            results[str(year)]['cohorts'][key] = {'scores': scores, 'stale_maturity_rows': stale}
            frame.to_parquet(out / f'{year}_{key}.parquet', index=False)
        print(json.dumps({str(year): results[str(year)]}), flush=True)
    for path, expected in inputs.items():
        if sha(Path(path)) != expected:
            raise ValueError(f'Input changed during experiment: {path}')
    report = {'years': results, 'checkpoint_replays_passed': True, 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
