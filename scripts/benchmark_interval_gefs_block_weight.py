"""Single fixed block-equal AFT comparison on current interval/GEFS inputs."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_gefs_72h_aft import compare_cohort, verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, sha


def block_weights(keys):
    if keys.empty or keys.isna().any():
        raise ValueError('Nonempty nonmissing block keys required')
    weights = 1. / keys.map(keys.value_counts()).to_numpy(dtype=float)
    return weights / weights.mean()


def main():
    started = time.monotonic()
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_2021_extension_v1'
    old = base / 'reusable_gefs_aft_v1'
    out = base / 'interval_gefs_block_weight_v1'
    if out.exists():
        raise FileExistsError(out)
    inputs = {str(Path(__file__)): sha(Path(__file__))}
    for folder in (source, old):
        report = json.loads((folder / 'report.json').read_bytes())
        inputs.update({str(folder / n): h for n, h in report['output_sha256'].items()})
        inputs[str(folder / 'report.json')] = sha(folder / 'report.json')
        inputs.update(json.loads((folder / 'protocol.json').read_bytes())['input_sha256'])
    for filename in ('scripts/benchmark_gefs_72h_aft.py',
                     'scripts/benchmark_observed_business_aft.py'):
        inputs[str(ROOT / filename)] = sha(ROOT / filename)
    verify_hashes(inputs)
    settings = json.loads((old / 'protocol.json').read_bytes())
    names = settings['features']
    train = pd.read_parquet(source / 'f024_training_rows.parquet')
    inference = pd.read_parquet(source / 'inference_rows.parquet')
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    assert len(names) == 104 and len(train) == 6770 and len(inference) == 519
    assert train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()
    assert not set(train.continuity_block) & set(
        pd.read_parquet(old / 'predictions.parquet').continuity_block)
    weights = block_weights(train.continuity_block)
    totals = pd.Series(weights, index=train.index).groupby(train.continuity_block).sum()
    np.testing.assert_allclose(totals, totals.iloc[0])
    control = joblib.load(old / 'model.joblib')
    control_prediction = control.predict(inference[names])
    frames = {}
    for name, filename in [('original', 'predictions.parquet'),
                           ('expanded', 'expanded_predictions.parquet')]:
        frame = pd.read_parquet(old / filename)
        pd.testing.assert_series_equal(frame.timestamp, inference.timestamp)
        np.testing.assert_array_equal(control_prediction, frame.forecast_aft)
        frames[name] = frame
    out.mkdir()
    protocol = {
        'input_sha256': inputs, 'features': names, 'seed': 42, 'rounds': 500,
        'weight': 'inverse training continuity-block row count, normalized to mean one',
        'arms': ['frozen_row_control', 'block_equal'],
        'training_rows': len(train), 'training_blocks': int(totals.size),
        'training_forecast_rows': int(train.gefs_mean_temperature_c_f024.notna().sum()),
        'selection': 'No sweep. Both original and expanded row/block MAE improve; tail not worse.',
        'crossyear': 'Only after development passes; no deployment authorization.',
        'limitations': ['Reused development data', 'Block weighting may reduce long-process influence',
                        'Continuity blocks are grouping units, not certified independent events'],
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    provenance = train[['timestamp', 'continuity_block', 'lower_bound_hours',
                        'upper_bound_hours', 'label_available_at']].copy()
    provenance['weight'] = weights
    provenance.to_parquet(out / 'training_weights.parquet', index=False)
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
        train[names], train.lower_bound_hours, train.upper_bound_hours, sample_weight=weights)
    joblib.dump(model, out / 'model.joblib')
    prediction = model.predict(inference[names])
    np.testing.assert_array_equal(prediction, joblib.load(out / 'model.joblib').predict(inference[names]))
    cohorts = {}
    for name, frame in frames.items():
        cohorts[name] = compare_cohort(frame, prediction)
        frame['block_equal_aft'] = prediction
        frame.to_parquet(out / f'{name}_predictions.parquet', index=False)
    verify_hashes(inputs)
    report = {'cohorts': cohorts, 'passed': all(v['passed'] for v in cohorts.values()),
              'deployment_authorized': False, 'elapsed_seconds': time.monotonic() - started,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
