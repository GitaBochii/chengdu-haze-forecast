"""Frozen observation backbone plus time-separated forecast AFT correction."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary

ROOT = Path(__file__).resolve().parents[1]


def head_inputs(frame, names, margin, use_weather):
    margin = np.asarray(margin, dtype=float)
    if margin.shape != (len(frame),) or not np.isfinite(margin).all():
        raise ValueError('Expected aligned finite raw margins')
    result = frame[names].astype(float).copy()
    if np.isinf(result.to_numpy()).any():
        raise ValueError('Infinite forecast input')
    supported = result.notna().all(axis=1).to_numpy()
    if not use_weather:
        result = result.where(result.isna(), 0.)
    result['base_log_hours'] = margin
    return result, supported


def bounded_matrix(frame, names):
    matrix = xgb.DMatrix(frame[names].astype(float))
    matrix.set_float_info('label_lower_bound', frame.lower_bound_hours)
    matrix.set_float_info('label_upper_bound', frame.upper_bound_hours)
    return matrix


def main():
    b = ROOT / 'artifacts/task01_recovery_head'
    out = b / 'staged_forecast_aft_2022_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [b / 'gefs_2021_extension_v1/training_rows.parquet',
             b / 'aft_temporal_search_20260918/protocol.json',
             b / 'aft_temporal_search_20260918/2022_8.ubj',
             Path(__file__).resolve(), ROOT / 'staged_forecast_aft_plan_20260918.md',
             ROOT / 'src/haze_forecast/cohort_scorecard.py',
             ROOT / 'src/haze_forecast/paired_interval_score.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    d = pd.read_parquet(paths[0])
    frozen = json.loads(paths[1].read_bytes())
    names = frozen['features']
    weather = [n for n in names if n.startswith(('gefs_', 'forecast_'))]
    observed = [n for n in names if n not in weather]
    c1, c2, c3 = [pd.Timestamp(f'{y}-01-01', tz='Asia/Shanghai') for y in (2021, 2022, 2023)]
    v = d.loc[d.timestamp.ge(c2) & d.timestamp.lt(c3) & d.label_available_at.lt(c3)].copy()
    h = d.loc[d.timestamp.ge(c1) & d.timestamp.lt(c2) & d.label_available_at.lt(c2)
              & ~d.continuity_block.isin(v.continuity_block) & d[weather].notna().all(axis=1)].copy()
    t = d.loc[d.timestamp.lt(c1) & d.label_available_at.lt(c1)
              & ~d.continuity_block.isin(set(h.continuity_block) | set(v.continuity_block))].copy()
    assert not set(t.continuity_block) & set(h.continuity_block)
    assert not set(t.continuity_block) & set(v.continuity_block)
    assert not set(h.continuity_block) & set(v.continuity_block)
    assert len(observed) == 92 and len(weather) == 12 and len(v) == 411 and len(h) > 0
    params = frozen['base_params']
    head_params = {**params, 'max_depth': 2, 'min_child_weight': 8, 'lambda': 10.}
    out.mkdir()
    protocol = {'input_sha256': hashes, 'observed': observed, 'weather': weather,
                'base_params': params, 'head_params': head_params, 'base_rounds': 500,
                'head_rounds': 100, 'training_rows': len(t), 'head_rows': len(h), 'validation_rows': len(v)}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    t.to_parquet(out / 'base_training.parquet', index=False)
    base = xgb.train(params, bounded_matrix(t, observed), num_boost_round=500)
    base.save_model(out / 'base.ubj')
    h['base_margin'] = base.predict(xgb.DMatrix(h[observed].astype(float)), output_margin=True)
    v['base_margin'] = base.predict(xgb.DMatrix(v[observed].astype(float)), output_margin=True)
    v['base'] = base.predict(xgb.DMatrix(v[observed].astype(float)))
    for label, use_weather in [('control', False), ('weather', True)]:
        hx, hs = head_inputs(h, weather, h.base_margin, use_weather)
        vx, vs = head_inputs(v, weather, v.base_margin, use_weather)
        assert hs.all()
        train = xgb.DMatrix(hx, base_margin=h.base_margin)
        train.set_float_info('label_lower_bound', h.lower_bound_hours)
        train.set_float_info('label_upper_bound', h.upper_bound_hours)
        head = xgb.train(head_params, train, num_boost_round=100)
        head.save_model(out / f'{label}.ubj')
        pred = head.predict(xgb.DMatrix(vx, base_margin=v.base_margin))
        v[label] = np.where(vs, pred, v['base'])
    native = xgb.Booster()
    native.load_model(paths[2])
    v['native'] = native.predict(xgb.DMatrix(v[names].astype(float)))
    scores = {n: score_predictions(v, v[n]) for n in ('base', 'control', 'weather', 'native')}
    paired = {n: paired_error_summary(v, v[n], v.weather) for n in ('base', 'control', 'native')}
    h.to_parquet(out / 'head_training.parquet', index=False)
    v.to_parquet(out / 'predictions.parquet', index=False)
    for p, digest in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
    report = {'scores': scores, 'paired': paired,
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == '__main__':
    main()
