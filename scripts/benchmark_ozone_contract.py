"""Frozen 2020 ozone-window correction, with unchanged labels and model settings."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    source = ROOT / 'artifacts/task01_recovery_head/transition_earlier_recovery_v1'
    hourly_path = ROOT / 'data/processed/regional_air_models_full_history/city_hourly.parquet'
    out = ROOT / 'artifacts/task01_recovery_head/ozone_contract_2020_v1'
    if out.exists():
        raise FileExistsError(out)
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    columns = ['o3', 'o3_lag_1', 'o3_lag_3', 'o3_lag_24', 'o3_roll_3', 'o3_roll_24']
    assert set(columns).issubset(names)
    paths = [hourly_path, Path(__file__), source / '2020_training.parquet',
             source / '2020_predictions.parquet', source / '2020_aft.joblib',
             source / 'protocol.json', ROOT / 'src/haze_forecast/observed_aft.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    out.mkdir()
    protocol = {'year': 2020, 'changed_columns': columns, 'seed': 42, 'rounds': 500,
                'input_sha256': hashes, 'deployment_authorized': False,
                'gate': 'row and process MAE decrease; tail24 process MAE does not increase',
                'limitations': ['Historical development year, not untouched test',
                                'Ozone source and averaging semantics change together',
                                'No future ozone, no label changes, no missing-row deletion']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    hourly = pd.read_parquet(hourly_path)
    hourly = hourly.loc[(hourly.city == 'chengdu') &
                        (hourly.timestamp < pd.Timestamp('2021-01-01', tz='Asia/Shanghai'))]
    assert not hourly.timestamp.duplicated().any()
    ozone = hourly.set_index('timestamp').o3_8h.sort_index().asfreq('h')
    replacement = pd.DataFrame({'o3': ozone})
    for lag in (1, 3, 24):
        replacement[f'o3_lag_{lag}'] = ozone.shift(lag)
    replacement['o3_roll_3'] = ozone.shift(1).rolling(3, min_periods=1).mean()
    replacement['o3_roll_24'] = ozone.shift(1).rolling(24, min_periods=6).mean()
    train = pd.read_parquet(source / '2020_training.parquet')
    valid = pd.read_parquet(source / '2020_predictions.parquet')
    cutoff = pd.Timestamp('2020-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()
    assert valid.timestamp.ge(cutoff).all()
    assert valid.label_available_at.lt(pd.Timestamp('2021-01-01', tz='Asia/Shanghai')).all()
    assert not set(train.continuity_block) & set(valid.continuity_block)
    baseline = joblib.load(source / '2020_aft.joblib').predict(valid[names])
    np.testing.assert_array_equal(baseline, valid.paired_aft)
    matrices, audit = [], {}
    for label, frame in [('training', train), ('validation', valid)]:
        original = frame[names].copy()
        fixed = original.copy()
        values = replacement.reindex(pd.DatetimeIndex(frame.timestamp))
        assert len(values) == len(frame)
        fixed[columns] = values[columns].to_numpy()
        pd.testing.assert_frame_equal(original.drop(columns=columns), fixed.drop(columns=columns))
        audit[label] = {'rows': len(frame), 'changed_cells': int((~np.isclose(
            original[columns], fixed[columns], equal_nan=True)).sum()),
            'original_missing': int(original[columns].isna().sum().sum()),
            'corrected_missing': int(fixed[columns].isna().sum().sum())}
        fixed.to_parquet(out / f'{label}_features.parquet', index=False)
        matrices.append(fixed)
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
        matrices[0], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    prediction = model.predict(matrices[1])
    np.testing.assert_array_equal(prediction, joblib.load(out / 'model.joblib').predict(matrices[1]))
    assert np.isfinite(prediction).all()
    result = valid.copy()
    result['ozone_corrected_prediction'] = prediction
    result.to_parquet(out / 'predictions.parquet', index=False)
    control, candidate = score_predictions(valid, baseline), score_predictions(valid, prediction)
    checks = {k: candidate[k] < control[k] for k in ('exact_row_mae', 'exact_block_mae')}
    checks['tail24_block_mae'] = candidate['tail24_block_mae'] <= control['tail24_block_mae']
    assert hashes == {str(p): sha(p) for p in paths}
    report = {'control': control, 'candidate': candidate, 'audit': audit,
              'checks': checks, 'passed': all(checks.values()), 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
