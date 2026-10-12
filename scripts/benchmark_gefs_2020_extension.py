"""One fixed AFT fit changing only audited 2020 forecast support."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary

ROOT = Path(__file__).resolve().parents[1]


def extend_training(train, features, names, expected_rows=273):
    if not train.timestamp.is_unique or not features.timestamp.is_unique or len(features) != expected_rows:
        raise ValueError('Unexpected origin population')
    times = features.timestamp.dt.tz_convert('Asia/Shanghai')
    if not (times.dt.year.eq(2020) & times.ge(pd.Timestamp('2020-09-23', tz='Asia/Shanghai'))).all():
        raise ValueError('Only audited post-upgrade 2020 rows allowed')
    mask = train.timestamp.isin(features.timestamp)
    if mask.sum() != expected_rows or not train.loc[mask, names].isna().all().all():
        raise ValueError('Origins missing or original forecast is not empty')
    values = features.set_index('timestamp').loc[train.loc[mask, 'timestamp'], names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Expected complete audited features')
    result = train.copy(deep=True)
    result.loc[mask, names] = values
    pd.testing.assert_frame_equal(result.loc[~mask], train.loc[~mask])
    pd.testing.assert_frame_equal(result.drop(columns=names), train.drop(columns=names))
    return result


def main():
    b = ROOT / 'artifacts/task01_recovery_head'
    audit = b / 'gefs_2020_support_audit_20260918'
    old = b / 'gefs_2021_extension_v1'
    out = b / 'gefs_2020_extension_2023_20260918'
    if out.exists():
        raise FileExistsError(out)
    ar = json.loads((audit / 'report.json').read_bytes())
    assert (ar['verified_runs'], ar['verified_grib_messages'], ar['covered_origins']) == (56,504,273)
    for name,digest in ar['output_sha256'].items():
        assert hashlib.sha256((audit/name).read_bytes()).hexdigest() == digest
    baselines = {'native': b / 'reusable_gefs_aft_v1/model.joblib', 'extended_2021': old / 'model.joblib'}
    paths = [audit / 'report.json', audit / 'features.parquet', old / 'training_rows.parquet',
             old / 'inference_rows.parquet', b / 'grouped_forecast_aft_2023_20260916/expanded.parquet',
             b / 'grouped_forecast_aft_2023_20260916/protocol.json', *baselines.values(),
             Path(__file__).resolve(), ROOT / 'gefs_2020_extension_plan_20260918.md',
             ROOT / 'src/haze_forecast/cohort_scorecard.py', ROOT / 'src/haze_forecast/paired_interval_score.py',
             ROOT / 'src/haze_forecast/training.py', ROOT / 'src/haze_forecast/observed_aft.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    config = json.loads(paths[5].read_bytes())
    names = config['features']
    forecast = [n for n in names if n.startswith(('gefs_', 'forecast_'))]
    assert len(names) == 104 and len(forecast) == 12
    train = extend_training(pd.read_parquet(paths[2]), pd.read_parquet(paths[1]), forecast)
    infer = pd.read_parquet(paths[3])
    cohort = pd.read_parquet(paths[4])
    assert len(train) == 6770 and len(infer) == 519 and len(cohort) == 396
    cut = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(cut).all() and train.label_available_at.lt(cut).all()
    assert not set(train.continuity_block) & set(cohort.continuity_block)
    assert train[forecast].notna().all(axis=1).sum() == 1135
    out.mkdir()
    (out/'protocol.json').write_text(json.dumps({'input_sha256': hashes, 'features': names,
        'forecast_fields': forecast, 'params': config['params'], 'rounds': 500,
        'baselines': {k:str(p) for k,p in baselines.items()}}, indent=2), encoding='utf-8')
    train.to_parquet(out/'training.parquet', index=False)
    dm = xgb.DMatrix(train[names].astype(float))
    dm.set_float_info('label_lower_bound', train.lower_bound_hours)
    dm.set_float_info('label_upper_bound', train.upper_bound_hours)
    model = xgb.train(config['params'], dm, num_boost_round=500)
    model.save_model(out/'candidate.ubj')
    pred = infer[['timestamp']].copy()
    pred['candidate'] = model.predict(xgb.DMatrix(infer[names].astype(float)))
    for label,path in baselines.items():
        pred[label] = joblib.load(path).predict(infer[names])
    scores = {}
    for label in ('candidate', *baselines):
        cohort[label] = pred.set_index('timestamp').loc[cohort.timestamp, label].to_numpy()
        scores[label] = score_predictions(cohort, cohort[label])
    paired = {label: paired_error_summary(cohort, cohort[label], cohort.candidate) for label in baselines}
    a = scores['candidate']
    passed = {label: a['exact_row_mae'] < scores[label]['exact_row_mae']
              and a['exact_block_mae'] < scores[label]['exact_block_mae']
              and a['tail24_block_mae'] <= scores[label]['tail24_block_mae']
              and paired[label]['all_row_delta_upper'] <= 0 and paired[label]['all_block_delta_upper'] <= 0
              for label in baselines}
    pred.to_parquet(out/'predictions.parquet', index=False)
    cohort.to_parquet(out/'scored.parquet', index=False)
    for p,digest in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
    report = {'scores': scores, 'paired': paired, 'passed_by_control': passed, 'passed': all(passed.values()),
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'scores':scores, 'passed_by_control':passed}, indent=2))


if __name__ == '__main__':
    main()
