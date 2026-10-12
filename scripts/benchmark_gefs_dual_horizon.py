"""Append f072 information without replacing any frozen f024 inputs."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_gefs_72h_aft import AGES, ROOT, compare_cohort, prepare, sha, verify_hashes
from scripts.build_gefs_72h_features import FIELDS, require


def combine(short, long, short_names, long_names):
    require(short.timestamp.tolist() == long.timestamp.tolist(), 'origin mismatch')
    extra = [name + '_f072' if name in AGES else name for name in long_names]
    require(not set(extra) & set(short.columns) and len(set(extra)) == len(extra), 'column collision')
    result = short.copy()
    result[extra] = long[long_names].to_numpy()
    pd.testing.assert_frame_equal(result[short.columns], short)
    return result, short_names + extra


def main():
    output = ROOT / 'artifacts/task01_recovery_head/gefs_dual_horizon_v1'
    if output.exists():
        raise FileExistsError(output)
    inputs, _, old_names, long_train, long_val, short_train, original, expanded = prepare()
    train, names = combine(short_train, long_train, old_names, FIELDS + AGES)
    inference, val_names = combine(original, long_val, old_names, FIELDS + AGES)
    require(names == val_names and len(names) == 116, 'unexpected feature schema')
    inputs[str(Path(__file__))] = sha(Path(__file__))
    verify_hashes(inputs)
    protocol = {'input_sha256': inputs, 'features': names, 'seed': 42, 'rounds': 500,
                'training_rows': len(train), 'validation_rows': len(inference),
                'selection': 'Keep all 104 f024 inputs and append nine f072 values plus three distinct ages.',
                'gate': 'Original and expanded: row/block MAE improve versus f024, tail block MAE no worse.',
                'limitations': ['Reused 2023 development set; no deployment authorization.',
                                'Archive availability proxy, not first-publication proof.',
                                'More input dimensions; not a matched-complexity causal comparison.']}
    output.mkdir()
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    train.to_parquet(output / 'training_rows.parquet', index=False)
    inference.to_parquet(output / 'inference_rows.parquet', index=False)
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
        train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, output / 'model.joblib')
    predictions = model.predict(inference[names])
    np.testing.assert_array_equal(joblib.load(output / 'model.joblib').predict(inference[names]), predictions)
    results = {}
    for name, frame in [('original', original), ('expanded', expanded)]:
        results[name] = compare_cohort(frame, predictions)
        frame['dual_horizon_aft'] = predictions
        frame.to_parquet(output / f'{name}_predictions.parquet', index=False)
    verify_hashes(inputs)
    report = {'cohorts': results, 'passed': all(r['passed'] for r in results.values()),
              'inputs_unchanged': True, 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
