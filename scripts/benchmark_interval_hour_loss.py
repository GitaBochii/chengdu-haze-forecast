"""One fixed 2022 development test of robust hour-scale interval distance."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.interval_hour_loss import interval_loss

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_temporal_calibration_data_v1'
    out = base / 'interval_hour_loss_2022_v1'
    if out.exists():
        raise FileExistsError(out)
    original = json.loads((source / 'report.json').read_bytes())
    inputs = {}
    for name, expected in original['output_sha256'].items():
        path = source / name
        if Path(name).name != name or sha(path) != expected:
            raise ValueError('Frozen input mismatch')
        inputs[str(path)] = expected
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    train = pd.read_parquet(source / 'training_rows.parquet')
    val = pd.read_parquet(source / 'predictions.parquet')
    cutoff = pd.Timestamp('2022-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    if not (train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()
            and val.timestamp.ge(cutoff).all() and val.timestamp.lt(end).all()
            and not set(train.continuity_block) & set(val.continuity_block)):
        raise ValueError('Invalid temporal split')
    mask = val.score_allowed.to_numpy()
    if val.score_allowed.dtype != bool or not val.loc[mask, 'label_available_at'].lt(end).all():
        raise ValueError('Invalid score mask')
    baseline = joblib.load(source / 'model.joblib').predict(val[names])
    np.testing.assert_array_equal(baseline, val.forecast_aft)
    for path in [Path(__file__).resolve(), ROOT / 'src/haze_forecast/interval_hour_loss.py',
                 ROOT / 'src/haze_forecast/cohort_scorecard.py', ROOT / 'src/haze_forecast/training.py',
                 ROOT / 'src/haze_forecast/app.py', source / 'report.json',
                 *list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*'))]:
        if path.is_file():
            inputs[str(path)] = sha(path)
    params = {'objective': 'reg:squarederror', 'disable_default_eval_metric': 1,
              'base_score': float(train.exact_hours.dropna().median()),
              'eta': 0.05, 'max_depth': 4, 'min_child_weight': 8,
              'subsample': 0.9, 'colsample_bytree': 0.8, 'lambda': 3.,
              'max_delta_step': 12., 'seed': 42, 'nthread': 1, 'tree_method': 'hist'}
    protocol = {'features': names, 'input_sha256': inputs, 'parameters': params,
                'delta_hours': 24., 'rounds': 500, 'development_year': 2022,
                'gate': 'Exact row/block strictly improve, tail24 and all-row/block error bounds no worse',
                'selection': 'One preset scale and one training run; no 2023-2025 scoring',
                'limitations': ['Reused development year, not untouched holdout',
                                'Interval distance is not a censoring likelihood; zero inside interval is not certainty',
                                'Different curvature changes effective tree regularization; not a pure controlled loss swap',
                                'Identity-link hour prediction clipped at zero, no probability interval output',
                                '2021 forecast support uses historical archive-availability proxy'],
                'deployment_authorized': False}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    matrix = xgb.DMatrix(train[names], label=np.zeros(len(train)))
    lo, hi = train.lower_bound_hours.to_numpy(), train.upper_bound_hours.to_numpy()

    def objective(prediction, _matrix):
        _, gradient, hessian = interval_loss(prediction, lo, hi)
        return gradient, hessian

    initial = float(interval_loss(np.full(len(train), params['base_score']), lo, hi)[0].mean())
    model = xgb.train(params, matrix, num_boost_round=500, obj=objective)
    model.save_model(out / 'model.ubj')
    replay = xgb.Booster()
    replay.load_model(out / 'model.ubj')
    xval = xgb.DMatrix(val[names])
    prediction = model.predict(xval).clip(min=0.)
    np.testing.assert_array_equal(prediction, replay.predict(xval).clip(min=0.))
    final = float(interval_loss(model.predict(matrix), lo, hi)[0].mean())
    saved = val.copy()
    saved['hour_loss_prediction'] = prediction
    saved.to_parquet(out / 'predictions.parquet', index=False)
    scores = {'aft': score_predictions(val.loc[mask], baseline[mask]),
              'hour_loss': score_predictions(val.loc[mask], prediction[mask])}
    reference, candidate = scores['aft'], scores['hour_loss']
    strict = ('exact_row_mae', 'exact_block_mae')
    nonworse = ('tail24_block_mae', 'all_row_error_lower', 'all_row_error_upper',
                'all_block_error_lower', 'all_block_error_upper')
    checks = {k: candidate[k] < reference[k] for k in strict}
    checks.update({k: candidate[k] <= reference[k] for k in nonworse})
    if inputs != {p: sha(Path(p)) for p in inputs}:
        raise ValueError('Input or deployment changed')
    report = {'scores': scores, 'checks': checks, 'passed': all(checks.values()),
              'train_loss_initial': initial, 'train_loss_final': final,
              'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
