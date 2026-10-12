"""Fixed full-duration spatial-feature experiment with metadata-only control."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary
from haze_forecast.station_dispersion import SPATIAL_COLUMNS, station_dispersion
from scripts.diagnose_regional_curves import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'station_dispersion_2023_20260918'
    if out.exists():
        raise FileExistsError(out)
    previous = base / 'grouped_forecast_aft_2023_20260916'
    frozen = json.loads((previous / 'protocol.json').read_bytes())
    hashes = dict(frozen['input_sha256'])
    paths = [Path(__file__).resolve(), ROOT / 'src/haze_forecast/station_dispersion.py',
             ROOT / 'spatial_dispersion_plan_20260918.md', previous / 'expanded.parquet',
             ROOT / 'data/processed/regional_air_models_full_history/station_hourly.parquet']
    hashes.update({str(p): digest(p) for p in paths})
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    names, params = frozen['features'], frozen['params']
    train = pd.read_parquet(base / 'gefs_2021_extension_v1/f024_training_rows.parquet')
    val = pd.read_parquet(base / 'gefs_2021_extension_v1/inference_rows.parquet')
    cohort = pd.read_parquet(previous / 'expanded.parquet')
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()
    assert not set(train.continuity_block) & set(cohort.continuity_block)
    stations = pd.read_parquet(paths[-1])
    stations = stations.loc[stations.timestamp.lt(pd.Timestamp('2024-01-01', tz='Asia/Shanghai'))]
    spatial = station_dispersion(stations).set_index('timestamp')
    prefix = station_dispersion(stations.loc[stations.timestamp.lt(cutoff)]).set_index('timestamp')
    pd.testing.assert_frame_equal(prefix, spatial.loc[spatial.index < cutoff])
    median = stations.groupby('timestamp').aqi.median()
    for frame in (train, val):
        assert frame.timestamp.is_unique and frame.timestamp.isin(spatial.index).all()
        np.testing.assert_allclose(median.loc[frame.timestamp], frame.aqi, rtol=0, atol=0)
    original_train, original_val = train.copy(), val.copy()
    for frame in (train, val):
        frame[SPATIAL_COLUMNS] = spatial.loc[frame.timestamp, SPATIAL_COLUMNS].to_numpy()
    pd.testing.assert_frame_equal(train[original_train.columns], original_train)
    pd.testing.assert_frame_equal(val[original_val.columns], original_val)
    baseline = joblib.load(base / 'reusable_gefs_aft_v1/model.joblib').predict(val[names])
    np.testing.assert_array_equal(pd.Series(baseline, index=val.timestamp).loc[cohort.timestamp], cohort.forecast_aft)
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({'input_sha256': hashes, 'baseline_features': names,
        'features': names+SPATIAL_COLUMNS, 'params': params, 'rounds': 500,
        'training_rows': len(train), 'inference_rows': len(val), 'score_rows': len(cohort),
        'target': 'full remaining hours', 'deployed': False}, indent=2), encoding='utf-8')
    predictions = val[['timestamp', *SPATIAL_COLUMNS]].copy()
    predictions['native'] = baseline
    reports = {}
    for name in ('native', 'metadata', 'spatial'):
        columns = names if name == 'native' else names+SPATIAL_COLUMNS
        x, xv = train[columns].astype(float).copy(), val[columns].astype(float).copy()
        if name == 'metadata':
            for feature in SPATIAL_COLUMNS[:-1]:
                x[feature] = x[feature].where(x[feature].isna(), 0.)
                xv[feature] = xv[feature].where(xv[feature].isna(), 0.)
        matrix, inference = xgb.DMatrix(x, missing=np.nan), xgb.DMatrix(xv, missing=np.nan)
        matrix.set_float_info('label_lower_bound', train.lower_bound_hours)
        matrix.set_float_info('label_upper_bound', train.upper_bound_hours)
        model = xgb.train(params, matrix, num_boost_round=500)
        prediction = model.predict(inference)
        if name == 'native':
            np.testing.assert_array_equal(prediction, baseline)
        model.save_model(out / f'{name}.ubj')
        restored = xgb.Booster()
        restored.load_model(out / f'{name}.ubj')
        np.testing.assert_array_equal(prediction, restored.predict(inference))
        predictions[name] = prediction
        cohort[name] = pd.Series(prediction, index=val.timestamp).loc[cohort.timestamp].to_numpy()
        reports[name] = score_predictions(cohort, cohort[name].to_numpy())
    paired = {name: paired_error_summary(cohort, cohort[name], cohort.spatial) for name in ('native', 'metadata')}
    gates = {name: all(reports['spatial'][key] < reports[name][key] for key in ('exact_row_mae', 'exact_block_mae'))
             and reports['spatial']['tail24_block_mae'] <= reports[name]['tail24_block_mae']
             and paired[name]['all_row_delta_lower'] <= 0 and paired[name]['all_block_delta_lower'] <= 0
             for name in ('native', 'metadata')}
    train.to_parquet(out / 'training.parquet', index=False)
    predictions.to_parquet(out / 'all_predictions.parquet', index=False)
    cohort.to_parquet(out / 'scored_predictions.parquet', index=False)
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    report = {'scores': reports, 'paired': paired, 'gates': gates, 'passed_development': all(gates.values()),
              'native_reproduction_exact': True, 'deployed': False,
              'output_sha256': {p.name: digest(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
