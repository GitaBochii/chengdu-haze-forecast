"""Fixed full-duration age-conditional development screen, with solver control."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.conditional_age_aft import conditional_age_terms, remaining_median
from haze_forecast.observed_pollution_run import build_observed_pollution_run
from haze_forecast.paired_interval_score import paired_error_summary
from scripts.diagnose_regional_curves import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'conditional_age_2023_20260918_v2'
    if out.exists():
        raise FileExistsError(out)
    previous = base / 'grouped_forecast_aft_2023_20260916'
    frozen = json.loads((previous / 'protocol.json').read_bytes())
    hashes = dict(frozen['input_sha256'])
    paths = [Path(__file__).resolve(), ROOT / 'src/haze_forecast/conditional_age_aft.py',
             ROOT / 'src/haze_forecast/lognormal_aft_audit.py', ROOT / 'src/haze_forecast/observed_pollution_run.py',
             ROOT / 'conditional_age_plan_20260918.md', previous / 'expanded.parquet',
             base / 'observed_business_dataset_v1/features.parquet']
    hashes.update({str(p): digest(p) for p in paths})
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    names, params = frozen['features'], frozen['params']
    train = pd.read_parquet(base / 'gefs_2021_extension_v1/f024_training_rows.parquet')
    inference = pd.read_parquet(base / 'gefs_2021_extension_v1/inference_rows.parquet')
    cohort = pd.read_parquet(previous / 'expanded.parquet')
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()
    assert not set(train.continuity_block) & set(cohort.continuity_block)
    hourly = pd.read_parquet(paths[-1], columns=['timestamp', 'aqi', 'pm2_5_iaqi', 'pm10_iaqi'])
    hourly = hourly.loc[hourly.timestamp.lt(pd.Timestamp('2024-01-01', tz='Asia/Shanghai'))]
    run = build_observed_pollution_run(hourly).set_index('timestamp')
    prefix = build_observed_pollution_run(hourly.loc[hourly.timestamp.lt(cutoff)]).set_index('timestamp')
    pd.testing.assert_frame_equal(prefix, run.loc[run.index < cutoff])
    ages = [run.loc[f.timestamp, 'observed_particle_run_hours'].to_numpy()-1 for f in (train, inference)]
    assert all(np.isfinite(a).all() and (a >= 0).all() for a in ages)
    matrix = xgb.DMatrix(train[names].astype(float), missing=np.nan)
    test = xgb.DMatrix(inference[names].astype(float), missing=np.nan)
    lo, hi = train.lower_bound_hours.to_numpy(), train.upper_bound_hours.to_numpy()
    control = joblib.load(base / 'reusable_gefs_aft_v1/model.joblib').predict(inference[names])
    np.testing.assert_array_equal(pd.Series(control, index=inference.timestamp).loc[cohort.timestamp], cohort.forecast_aft)
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({'input_sha256': hashes, 'features': names,
        'params': params, 'training_rows': len(train), 'inference_rows': len(inference),
        'rounds': 500, 'scale': 1.2, 'curvature_step': 1e-4, 'hessian_floor': 1e-6,
        'target': 'Full remaining hours, no cap', 'deployed': False}, indent=2), encoding='utf-8')
    all_predictions = inference[['timestamp']].copy()
    all_predictions['native_aft'] = control
    summaries = {}
    for name, age, validation_age in [('zero_age', np.zeros(len(train)), np.zeros(len(inference))),
                                       ('conditional_age', ages[0], ages[1])]:
        exact = lo == hi
        intercept = float(np.median(np.log(age[exact] + lo[exact])))
        custom_params = {k: v for k, v in params.items() if k not in (
            'objective', 'eval_metric', 'aft_loss_distribution', 'aft_loss_distribution_scale')}
        custom_params.update(objective='reg:squarederror', base_score=intercept, disable_default_eval_metric=1)
        counts = {'hessian_floored': 0, 'gradient_cells': 0}

        def objective(margin, _matrix, age=age, counts=counts):
            _, gradient = conditional_age_terms(lo, hi, margin, age)
            eps = 1e-4
            plus = conditional_age_terms(lo, hi, margin.astype(float)+eps, age)[1]
            minus = conditional_age_terms(lo, hi, margin.astype(float)-eps, age)[1]
            hessian = (plus-minus)/(2*eps)
            if not np.isfinite(hessian).all():
                raise FloatingPointError('Nonfinite curvature')
            counts['hessian_floored'] += int((hessian < 1e-6).sum())
            counts['gradient_cells'] += len(hessian)
            return gradient, np.maximum(hessian, 1e-6)

        initial_loss = float(conditional_age_terms(lo, hi, np.full(len(train), intercept), age)[0].mean())
        model = xgb.train(custom_params, matrix, num_boost_round=500, obj=objective)
        path = out / f'{name}.ubj'
        model.save_model(path)
        restored = xgb.Booster()
        restored.load_model(path)
        margin = model.predict(test, output_margin=True)
        np.testing.assert_array_equal(margin, restored.predict(test, output_margin=True))
        prediction = remaining_median(margin, validation_age)
        all_predictions[name] = prediction
        cohort[name] = pd.Series(prediction, index=inference.timestamp).loc[cohort.timestamp].to_numpy()
        summaries[name] = {'counts': counts, 'initial_nll': initial_loss,
            'final_nll': float(conditional_age_terms(lo, hi, model.predict(matrix, output_margin=True), age)[0].mean()),
            'params': custom_params, 'scores': score_predictions(cohort, cohort[name].to_numpy())}
    native_score = score_predictions(cohort, cohort.forecast_aft.to_numpy())
    candidate = summaries['conditional_age']['scores']
    gates = {name: all(candidate[key] < scores[key] for key in ('exact_row_mae', 'exact_block_mae'))
             and candidate['tail24_block_mae'] <= scores['tail24_block_mae']
             for name, scores in [('native', native_score), ('zero_age', summaries['zero_age']['scores'])]}
    pairs = {name: paired_error_summary(cohort, cohort[column], cohort.conditional_age)
             for name, column in [('native', 'forecast_aft'), ('zero_age', 'zero_age')]}
    for name, paired in pairs.items():
        gates[name] &= paired['all_row_delta_lower'] <= 0 and paired['all_block_delta_lower'] <= 0
    all_predictions.to_parquet(out / 'all_predictions.parquet', index=False)
    cohort.to_parquet(out / 'scored_predictions.parquet', index=False)
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    report = {'models': summaries, 'native_scores': native_score, 'paired': pairs, 'gates': gates,
              'passed_development': all(gates.values()), 'deployed': False,
              'output_sha256': {p.name: digest(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
