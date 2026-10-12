"""One distribution substitution on original interval/GEFS support."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.benchmark_gefs_72h_aft import compare_cohort, verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, sha


def fit_model(frame, names, distribution, rounds=500):
    if distribution not in ('normal', 'logistic'):
        raise ValueError('Only the two predeclared distributions are permitted')
    matrix = xgb.DMatrix(frame[names].astype(float))
    matrix.set_float_info('label_lower_bound', frame.lower_bound_hours.to_numpy())
    matrix.set_float_info('label_upper_bound', frame.upper_bound_hours.to_numpy())
    params = {'objective': 'survival:aft', 'eval_metric': 'aft-nloglik',
              'aft_loss_distribution': distribution, 'aft_loss_distribution_scale': 1.2,
              'eta': .05, 'max_depth': 4, 'min_child_weight': 8, 'subsample': .9,
              'colsample_bytree': .8, 'lambda': 3., 'seed': 42,
              'tree_method': 'hist', 'verbosity': 0, 'nthread': 1}
    return xgb.train(params, matrix, num_boost_round=rounds)


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_2021_extension_v1'
    old = base / 'reusable_gefs_aft_v1'
    out = base / 'interval_gefs_logistic_v1'
    if out.exists():
        raise FileExistsError(out)
    inputs = {str(Path(__file__)): sha(Path(__file__))}
    for folder in (source, old):
        report = json.loads((folder / 'report.json').read_bytes())
        inputs.update({str(folder / n): h for n, h in report['output_sha256'].items()})
        inputs[str(folder / 'report.json')] = sha(folder / 'report.json')
        inputs.update(json.loads((folder / 'protocol.json').read_bytes())['input_sha256'])
    verify_hashes(inputs)
    names = json.loads((old / 'protocol.json').read_bytes())['features']
    train = pd.read_parquet(source / 'f024_training_rows.parquet')
    val = pd.read_parquet(source / 'inference_rows.parquet')
    frames = {key: pd.read_parquet(old / filename) for key, filename in
              [('original', 'predictions.parquet'), ('expanded', 'expanded_predictions.parquet')]}
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    assert len(names) == 104 and len(train) == 6770 and len(val) == 519
    assert train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()
    for frame in frames.values():
        pd.testing.assert_series_equal(frame.timestamp, val.timestamp)
        assert not set(train.continuity_block) & set(frame.continuity_block)
    out.mkdir()
    protocol = {'input_sha256': inputs, 'features': names, 'rounds': 500,
                'change': 'normal to logistic only; scale 1.2 unchanged',
                'prediction': 'exp margin, fitted distribution median for both symmetric noise laws',
                'gate': 'Both original/expanded row and block MAE improve; tail not worse than original f024.',
                'selection': 'one alternative, no scale or seed search', 'deployment_authorized': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    matrix = xgb.DMatrix(val[names].astype(float))
    normal = fit_model(train, names, 'normal')
    np.testing.assert_array_equal(normal.predict(matrix), frames['original'].forecast_aft)
    normal.save_model(out / 'normal.ubj')
    logistic = fit_model(train, names, 'logistic')
    logistic.save_model(out / 'logistic.ubj')
    prediction = logistic.predict(matrix)
    cohorts = {}
    for key, frame in frames.items():
        cohorts[key] = compare_cohort(frame, prediction)
        frame['logistic_aft'] = prediction
        frame.to_parquet(out / f'{key}_predictions.parquet', index=False)
    verify_hashes(inputs)
    report = {'cohorts': cohorts, 'passed': all(v['passed'] for v in cohorts.values()),
              'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
