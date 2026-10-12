"""Fixed weather-value ablation retaining forecast age and frozen backbone margin."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.benchmark_staged_forecast_aft import head_inputs

ROOT = Path(__file__).resolve().parents[1]


def main():
    b = ROOT / 'artifacts/task01_recovery_head'
    source = b / 'staged_forecast_aft_2022_20260918'
    out = b / 'staged_weather_specificity_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / n for n in ('protocol.json', 'head_training.parquet', 'predictions.parquet')]
    paths += [Path(__file__).resolve(), ROOT / 'scripts/benchmark_staged_forecast_aft.py',
              ROOT / 'staged_weather_specificity_plan_20260918.md']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    c = json.loads(paths[0].read_bytes())
    h, v = pd.read_parquet(paths[1]), pd.read_parquet(paths[2])
    fields = [n for n in c['weather'] if n.startswith('gefs_')]
    ages = [n for n in c['weather'] if n.startswith('forecast_')]
    assert len(fields) == 9 and len(ages) == 3
    matrices = []
    for frame in (h, v):
        original, supported = head_inputs(frame, c['weather'], frame.base_margin, True)
        x = original.copy()
        x[fields] = x[fields].where(x[fields].isna(), 0.)
        pd.testing.assert_frame_equal(x.isna(), original.isna())
        pd.testing.assert_frame_equal(x.drop(columns=fields), original.drop(columns=fields))
        matrices.append((x, supported))
    hx, hs = matrices[0]
    vx, vs = matrices[1]
    assert hs.all() and int((~vs).sum()) == 47
    assert not set(h.continuity_block) & set(v.continuity_block)
    assert h.timestamp.dt.year.eq(2021).all() and v.timestamp.dt.year.eq(2022).all()
    assert h.label_available_at.lt(pd.Timestamp('2022-01-01', tz='Asia/Shanghai')).all()
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({'input_sha256': hashes, 'params': c['head_params'],
        'rounds': c['head_rounds'], 'masked_fields': fields, 'retained_ages': ages}, indent=2), encoding='utf-8')
    train = xgb.DMatrix(hx, base_margin=h.base_margin)
    train.set_float_info('label_lower_bound', h.lower_bound_hours)
    train.set_float_info('label_upper_bound', h.upper_bound_hours)
    model = xgb.train(c['head_params'], train, num_boost_round=c['head_rounds'])
    model.save_model(out / 'metadata_head.ubj')
    prediction = model.predict(xgb.DMatrix(vx, base_margin=v.base_margin))
    v['metadata'] = np.where(vs, prediction, v.base)
    # Rebuild from the archived source without the shared feature helper.
    replay = v[c['weather']].astype(float).copy()
    for n in fields:
        replay.loc[replay[n].notna(), n] = 0.
    replay['base_log_hours'] = v.base_margin.to_numpy()
    pd.testing.assert_frame_equal(replay, vx)
    restored = xgb.Booster()
    restored.load_model(out / 'metadata_head.ubj')
    p = restored.predict(xgb.DMatrix(replay, base_margin=v.base_margin))
    np.testing.assert_array_equal(np.where(vs, p, v.base), v.metadata)
    np.testing.assert_array_equal(v.loc[~vs, 'metadata'], v.loc[~vs, 'base'])
    scores, errors = {}, {}
    groups = v.continuity_block.to_numpy()
    tail = v.exact_hours.to_numpy() >= 24
    assert (v.exact_hours == v.lower_bound_hours).all() and (v.exact_hours == v.upper_bound_hours).all()
    for name in ('weather', 'metadata'):
        e = np.array([abs(float(p)-float(y)) for p,y in zip(v[name], v.exact_hours)])
        scores[name] = [float(e.mean()), float(np.mean([e[groups == g].mean() for g in set(groups)])),
                       float(np.mean([e[(groups == g) & tail].mean() for g in set(groups[tail])]))]
        vector = (v[name] - v.exact_hours).abs()
        np.testing.assert_allclose(scores[name], [vector.mean(), vector.groupby(v.continuity_block).mean().mean(),
            vector[tail].groupby(v.loc[tail, 'continuity_block']).mean().mean()], atol=1e-12)
        errors[name] = e
    delta = errors['weather'] - errors['metadata']
    block_delta = [float(delta[groups == g].mean()) for g in sorted(set(groups))]
    for p,digest in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
    v.to_parquet(out / 'predictions.parquet', index=False)
    report = {'scores_row_block_tail': scores, 'weather_minus_metadata_block_deltas': block_delta,
              'weather_improved_blocks': sum(d < 0 for d in block_delta),
              'weather_worse_blocks': sum(d > 0 for d in block_delta),
              'replayed_predictions': len(v), 'fallback_rows': int((~vs).sum()),
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
