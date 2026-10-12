"""Unused-column control for fixed dual-horizon AFT; diagnostic only."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_gefs_72h_aft import ROOT, compare_cohort, sha, verify_hashes


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_dual_horizon_v1'
    out = base / 'gefs_empty_columns_v1'
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((source / 'report.json').read_bytes())
    protocol = json.loads((source / 'protocol.json').read_bytes())
    inputs = {str(source / name): value for name, value in report['output_sha256'].items()}
    inputs.update(protocol['input_sha256'])
    inputs[str(source / 'report.json')] = sha(source / 'report.json')
    inputs[str(Path(__file__))] = sha(Path(__file__))
    verify_hashes(inputs)
    names = protocol['features']
    assert len(names) == 116
    train = pd.read_parquet(source / 'training_rows.parquet')
    inference = pd.read_parquet(source / 'inference_rows.parquet')
    for frame in [train, inference]:
        preserved = frame[names[:104]].copy()
        frame[names[104:]] = np.nan
        pd.testing.assert_frame_equal(frame[names[:104]], preserved)
        assert frame[names[104:]].isna().all().all()
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({
        'input_sha256': inputs, 'features': names, 'seed': 42, 'rounds': 500,
        'diagnostic_only': True, 'intervention': 'All twelve appended columns missing in training and inference.'
    }, indent=2), encoding='utf-8')
    train.to_parquet(out / 'training_rows.parquet', index=False)
    inference.to_parquet(out / 'inference_rows.parquet', index=False)
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
        train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    predictions = model.predict(inference[names])
    np.testing.assert_array_equal(joblib.load(out / 'model.joblib').predict(inference[names]), predictions)
    results = {}
    for cohort in ['original', 'expanded']:
        data = pd.read_parquet(source / f'{cohort}_predictions.parquet')
        results[cohort] = compare_cohort(data, predictions)
        data['empty_columns_aft'] = predictions
        data.to_parquet(out / f'{cohort}_predictions.parquet', index=False)
    verify_hashes(inputs)
    result = {'cohorts': results, 'deployment_authorized': False,
              'max_change_from_f024_hours': float(np.max(np.abs(predictions-inference.forecast_aft.to_numpy()))),
              'limitations': ['Diagnostic only; random column selection and tree fitting can change.',
                              'Does not by itself isolate the causal contribution of forecast values.'],
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
