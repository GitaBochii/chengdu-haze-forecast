"""Fixed pre-2022 AFT experiment adding eight causal weather-gap descriptors."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from haze_forecast.weather_observation_age import weather_observation_age
from haze_forecast.weather_source_ablation import WEATHER_BASE
from scripts.benchmark_interval_hour_loss import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_temporal_calibration_data_v1'
    out = base / 'weather_observation_age_2022_v1'
    if out.exists():
        raise FileExistsError(out)
    old = json.loads((source / 'report.json').read_bytes())
    inputs = {str(source / name): value for name, value in old['output_sha256'].items()}
    if inputs != {p: sha(Path(p)) for p in inputs}:
        raise ValueError('Frozen source changed')
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    train = pd.read_parquet(source / 'training_rows.parquet')
    val = pd.read_parquet(source / 'predictions.parquet')
    grid_path = base / 'observed_business_dataset_v1/features.parquet'
    full = pd.read_parquet(grid_path)
    # Do not inspect later development/test weather when constructing this experiment.
    full = full.loc[full.timestamp.lt(pd.Timestamp('2023-01-01', tz='Asia/Shanghai'))].copy()
    descriptors = weather_observation_age(full).set_index('timestamp')
    extra = descriptors.columns.tolist()
    for frame in (train, val):
        np.testing.assert_allclose(frame[list(WEATHER_BASE)],
                                   full.set_index('timestamp').loc[frame.timestamp, list(WEATHER_BASE)],
                                   equal_nan=True)
    enriched = []
    for frame in (train, val):
        copy = frame.copy()
        copy[extra] = descriptors.loc[frame.timestamp].to_numpy()
        pd.testing.assert_frame_equal(copy[frame.columns], frame)
        enriched.append(copy)
    tr, va = enriched
    cutoff = pd.Timestamp('2022-01-01', tz='Asia/Shanghai')
    if not (tr.timestamp.lt(cutoff).all() and tr.label_available_at.lt(cutoff).all()
            and not set(tr.continuity_block) & set(va.continuity_block)):
        raise ValueError('Temporal leakage')
    mask = va.score_allowed.to_numpy()
    if va.score_allowed.dtype != bool or not va.loc[mask, 'label_available_at'].lt(
            pd.Timestamp('2023-01-01', tz='Asia/Shanghai')).all():
        raise ValueError('Invalid scoring clock')
    baseline = joblib.load(source / 'model.joblib').predict(val[names])
    np.testing.assert_array_equal(baseline, val.forecast_aft)
    protected = [grid_path, Path(__file__).resolve(), source / 'report.json',
                 ROOT / 'src/haze_forecast/weather_observation_age.py',
                 ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py',
                 ROOT / 'src/haze_forecast/app.py',
                 *list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*'))]
    inputs.update({str(p): sha(p) for p in protected if p.is_file()})
    protocol = {'input_sha256': inputs, 'features': names + extra, 'extra_features': extra,
                'seed': 42, 'rounds': 500, 'year': 2022,
                'intervention': 'Four observed-gap ages and four missing flags; no imputation',
                'gate': 'Row/block MAE strictly better and tail24 no worse than paired AFT',
                'limitations': ['2022 reused development, not independent holdout',
                                'Observation age is not publication or local receipt age',
                                'Eight added columns also change feature subsampling',
                                'No 2023-2025 evaluation or deployment'],
                'deployment_authorized': False}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    tr.to_parquet(out / 'training.parquet', index=False)
    va.to_parquet(out / 'inference.parquet', index=False)
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
        tr[names + extra], tr.lower_bound_hours, tr.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    prediction = model.predict(va[names + extra])
    np.testing.assert_array_equal(prediction, joblib.load(out / 'model.joblib').predict(va[names + extra]))
    scores = {'aft': score_predictions(val.loc[mask], baseline[mask]),
              'observation_age': score_predictions(va.loc[mask], prediction[mask])}
    result = va.copy()
    result['candidate'] = prediction
    result.to_parquet(out / 'predictions.parquet', index=False)
    reference, candidate = scores['aft'], scores['observation_age']
    checks = {key: candidate[key] < reference[key] for key in ('exact_row_mae', 'exact_block_mae')}
    checks['tail24_block_mae'] = candidate['tail24_block_mae'] <= reference['tail24_block_mae']
    if inputs != {p: sha(Path(p)) for p in inputs}:
        raise ValueError('Protected input changed')
    report = {'scores': scores, 'checks': checks, 'passed': all(checks.values()),
              'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
