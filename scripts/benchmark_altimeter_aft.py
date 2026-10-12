"""Delayed pressure numeric signal versus same-missingness AFT control."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.altimeter_features import COLUMNS, altimeter_features
from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'altimeter_aft_2023_20260918'
    if out.exists():
        raise FileExistsError(out)
    old = base / 'grouped_forecast_aft_2023_20260916'
    frozen = json.loads((old / 'protocol.json').read_bytes())
    for p, h in frozen['input_sha256'].items():
        assert digest(Path(p)) == h
    paths = [base / 'gefs_2021_extension_v1/f024_training_rows.parquet',
             base / 'gefs_2021_extension_v1/inference_rows.parquet', old / 'expanded.parquet',
             base / 'altimeter_source_20260918/hourly.parquet', old / 'protocol.json',
             base / 'reusable_gefs_aft_v1/model.joblib', Path(__file__).resolve(),
             ROOT / 'src/haze_forecast/altimeter_features.py', ROOT / 'altimeter_aft_plan_20260918.md']
    hashes = dict(frozen['input_sha256'])
    hashes.update({str(p): digest(p) for p in paths})
    train, valid, cohort, hourly = [pd.read_parquet(p) for p in paths[:4]]
    names, params = frozen['features'], frozen['params']
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()
    assert not set(train.continuity_block) & set(cohort.continuity_block)
    for frame in (train, valid):
        assert frame.timestamp.is_unique
        frame[COLUMNS] = altimeter_features(hourly, frame.timestamp).to_numpy()
    before = hourly.loc[hourly.timestamp < cutoff]
    np.testing.assert_allclose(altimeter_features(before, train.timestamp), train[COLUMNS], equal_nan=True)
    baseline = joblib.load(paths[5]).predict(valid[names])
    np.testing.assert_array_equal(pd.Series(baseline, index=valid.timestamp).loc[cohort.timestamp], cohort.forecast_aft)
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({'input_sha256': hashes, 'features': names,
        'added_features': COLUMNS, 'params': params, 'rounds': 500}, indent=2), encoding='utf-8')
    predictions = valid[['timestamp', *COLUMNS]].copy()
    scores = {}
    for label in ('native', 'missingness', 'pressure'):
        cols = names if label == 'native' else names+COLUMNS
        x, xv = train[cols].astype(float).copy(), valid[cols].astype(float).copy()
        if label == 'missingness':
            x[COLUMNS] = x[COLUMNS].where(x[COLUMNS].isna(), 0.)
            xv[COLUMNS] = xv[COLUMNS].where(xv[COLUMNS].isna(), 0.)
        matrix, infer = xgb.DMatrix(x), xgb.DMatrix(xv)
        matrix.set_float_info('label_lower_bound', train.lower_bound_hours)
        matrix.set_float_info('label_upper_bound', train.upper_bound_hours)
        model = xgb.train(params, matrix, num_boost_round=500)
        pred = model.predict(infer)
        if label == 'native':
            np.testing.assert_array_equal(pred, baseline)
        model.save_model(out / f'{label}.ubj')
        predictions[label] = pred
        cohort[label] = pd.Series(pred, index=valid.timestamp).loc[cohort.timestamp].to_numpy()
        scores[label] = score_predictions(cohort, cohort[label].to_numpy())
    paired = {n: paired_error_summary(cohort, cohort[n], cohort.pressure) for n in ('native', 'missingness')}
    gates = {n: all(scores['pressure'][k] < scores[n][k] for k in ('exact_row_mae', 'exact_block_mae'))
             and scores['pressure']['tail24_block_mae'] <= scores[n]['tail24_block_mae']
             and paired[n]['all_row_delta_upper'] <= 0 and paired[n]['all_block_delta_upper'] <= 0
             for n in paired}
    train.to_parquet(out / 'training.parquet', index=False)
    valid.to_parquet(out / 'inference.parquet', index=False)
    predictions.to_parquet(out / 'predictions.parquet', index=False)
    cohort.to_parquet(out / 'scored.parquet', index=False)
    for p, h in hashes.items():
        assert digest(Path(p)) == h
    result = {'scores': scores, 'paired': paired, 'gates': gates, 'pass': all(gates.values()),
              'output_sha256': {p.name: digest(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
