"""Replay staged models and independently check support, isolation and metrics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]


def main():
    b = ROOT / 'artifacts/task01_recovery_head/staged_forecast_aft_2022_20260918'
    c = json.loads((b / 'protocol.json').read_bytes())
    r = json.loads((b / 'report.json').read_bytes())
    for p, h in c['input_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    for p, h in r['output_sha256'].items():
        assert hashlib.sha256((b / p).read_bytes()).hexdigest() == h
    t, h, v = [pd.read_parquet(b / p) for p in ('base_training.parquet', 'head_training.parquet', 'predictions.parquet')]
    for frame, year in ((t, 2021), (h, 2022), (v, 2023)):
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        assert frame.timestamp.lt(cut).all() and frame.label_available_at.lt(cut).all()
    assert h.timestamp.dt.year.eq(2021).all() and v.timestamp.dt.year.eq(2022).all()
    assert not set(t.continuity_block) & set(h.continuity_block)
    assert not set(t.continuity_block) & set(v.continuity_block)
    assert not set(h.continuity_block) & set(v.continuity_block)
    source = pd.read_parquet(ROOT / 'artifacts/task01_recovery_head/gefs_2021_extension_v1/training_rows.parquet').set_index('timestamp')
    for frame in (t, h, v):
        shared = list(source.columns)
        pd.testing.assert_frame_equal(frame.set_index('timestamp')[shared], source.loc[frame.timestamp, shared])
    base = xgb.Booster()
    base.load_model(b / 'base.ubj')
    for frame in (h, v):
        dm = xgb.DMatrix(frame[c['observed']].astype(float))
        np.testing.assert_array_equal(base.predict(dm, output_margin=True), frame.base_margin)
    np.testing.assert_array_equal(base.predict(dm), v.base)
    supported = v[c['weather']].notna().all(axis=1).to_numpy()
    assert h[c['weather']].notna().all().all()
    for name in ('control', 'weather'):
        x = v[c['weather']].astype(float).copy()
        if name == 'control':
            for col in c['weather']:
                x.loc[x[col].notna(), col] = 0.
        x['base_log_hours'] = v.base_margin
        model = xgb.Booster()
        model.load_model(b / f'{name}.ubj')
        pred = model.predict(xgb.DMatrix(x, base_margin=v.base_margin))
        pred[~supported] = v.base.to_numpy()[~supported]
        np.testing.assert_array_equal(pred, v[name])
    native_path = next(Path(p) for p in c['input_sha256'] if p.endswith('2022_8.ubj'))
    model = xgb.Booster()
    model.load_model(native_path)
    features = json.loads((native_path.parent / 'protocol.json').read_bytes())['features']
    np.testing.assert_array_equal(model.predict(xgb.DMatrix(v[features].astype(float))), v.native)
    assert (v.lower_bound_hours == v.upper_bound_hours).all()
    assert (v.exact_hours == v.lower_bound_hours).all()
    errors = {}
    for name in ('base', 'control', 'weather', 'native'):
        e = np.array([abs(float(p) - float(y)) for p, y in zip(v[name], v.exact_hours)])
        groups = v.continuity_block.to_numpy()
        tail = v.exact_hours.to_numpy() >= 24
        metrics = [e.mean(), np.mean([e[groups == g].mean() for g in set(groups)]),
                   np.mean([e[(groups == g) & tail].mean() for g in set(groups[tail])])]
        keys = ('exact_row_mae', 'exact_block_mae', 'tail24_block_mae')
        np.testing.assert_allclose(metrics, [r['scores'][name][k] for k in keys], atol=1e-12)
        errors[name] = e
    for name in ('base', 'control', 'native'):
        delta = errors['weather'] - errors[name]
        block = np.mean([delta[groups == g].mean() for g in set(groups)])
        for suffix in ('lower', 'upper'):
            np.testing.assert_allclose([delta.mean(), block],
                                      [r['paired'][name][f'all_row_delta_{suffix}'],
                                       r['paired'][name][f'all_block_delta_{suffix}']], atol=1e-12)
        a, z = r['scores']['weather'], r['scores'][name]
        assert a['exact_row_mae'] < z['exact_row_mae'] and a['exact_block_mae'] < z['exact_block_mae']
        assert a['tail24_block_mae'] <= z['tail24_block_mae'] and delta.mean() <= 0 and block <= 0
    print('1644 predictions, 909 margins, 12 metrics, paired deltas, source identity, time/block isolation and hashes passed')


if __name__ == '__main__':
    main()
