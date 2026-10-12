"""One same-data joint-fit control; never refit or tune the staged candidate."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]


def main():
    b = ROOT / 'artifacts/task01_recovery_head'
    stage = b / 'staged_forecast_aft_2022_20260918'
    out = b / 'staged_forecast_matched_control_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [stage / n for n in ('protocol.json', 'base_training.parquet', 'head_training.parquet', 'predictions.parquet')]
    paths += [Path(__file__).resolve(), ROOT / 'staged_forecast_matched_control_plan_20260918.md']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    c = json.loads(paths[0].read_bytes())
    t = pd.concat([pd.read_parquet(paths[1]), pd.read_parquet(paths[2])], ignore_index=True).sort_values('timestamp')
    v = pd.read_parquet(paths[3])
    features = c['observed'] + c['weather']
    assert t.timestamp.is_unique and not set(t.continuity_block) & set(v.continuity_block)
    cut = pd.Timestamp('2022-01-01', tz='Asia/Shanghai')
    assert t.timestamp.lt(cut).all() and t.label_available_at.lt(cut).all()
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({'input_sha256': hashes, 'features': features,
        'params': c['base_params'], 'rounds': 500, 'rows': len(t)}, indent=2), encoding='utf-8')
    dm = xgb.DMatrix(t[features].astype(float))
    dm.set_float_info('label_lower_bound', t.lower_bound_hours)
    dm.set_float_info('label_upper_bound', t.upper_bound_hours)
    infer = xgb.DMatrix(v[features].astype(float))
    model = xgb.train(c['base_params'], dm, num_boost_round=500)
    v['joint'] = model.predict(infer)
    model.save_model(out / 'joint.ubj')
    restored = xgb.Booster()
    restored.load_model(out / 'joint.ubj')
    np.testing.assert_array_equal(restored.predict(infer), v.joint)
    scores = {}
    for name in ('joint', 'weather'):
        error = np.array([abs(float(p) - float(y)) for p, y in zip(v[name], v.exact_hours)])
        groups = v.continuity_block.to_numpy()
        tail = v.exact_hours.to_numpy() >= 24
        scores[name] = [float(error.mean()), float(np.mean([error[groups == g].mean() for g in set(groups)])),
                        float(np.mean([error[(groups == g) & tail].mean() for g in set(groups[tail])]))]
    a, z = scores['weather'], scores['joint']
    passed = a[0] < z[0] and a[1] < z[1] and a[2] <= z[2]
    v.to_parquet(out / 'predictions.parquet', index=False)
    for p, digest in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
    report = {'scores_row_block_tail': scores, 'staged_passed_matched_control': passed,
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
