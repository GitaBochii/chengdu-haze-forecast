"""Five predeclared process-bootstrap members and paired ordinary-seed controls."""

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


def main():
    b = ROOT / 'artifacts/task01_recovery_head'
    out = b / 'process_bagging_2023_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [b / 'gefs_2021_extension_v1/f024_training_rows.parquet',
             b / 'gefs_2021_extension_v1/inference_rows.parquet',
             b / 'grouped_forecast_aft_2023_20260916/expanded.parquet',
             b / 'grouped_forecast_aft_2023_20260916/protocol.json',
             b / 'reusable_gefs_aft_v1/model.joblib', Path(__file__).resolve(),
             ROOT / 'process_bagging_plan_20260918.md']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    t, v, scored = [pd.read_parquet(p) for p in paths[:3]]
    protocol = json.loads(paths[3].read_bytes())
    for p, h in protocol['input_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    names = protocol['features']
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    assert t.timestamp.lt(cutoff).all() and t.label_available_at.lt(cutoff).all()
    assert not set(t.continuity_block) & set(scored.continuity_block)
    ids = np.sort(t.continuity_block.unique())
    assert len(ids) == 190 and len(t) == 6770
    baseline = joblib.load(paths[4]).predict(v[names])
    inference = xgb.DMatrix(v[names].astype(float))
    predictions = v[['timestamp']].copy()
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({'input_sha256': hashes, 'features': names,
        'params': protocol['params'], 'seeds': list(range(42, 47)), 'rounds': 500,
        'blocks': ids.tolist(), 'aggregation': 'median of five predicted hours'}, indent=2), encoding='utf-8')
    weights = t[['timestamp', 'continuity_block']].copy()
    multiplicities = pd.DataFrame({'continuity_block': ids})
    scores = {}
    for seed in range(42, 47):
        sampled = np.random.default_rng(seed).choice(ids, size=len(ids), replace=True)
        counts = pd.Series(sampled).value_counts().reindex(ids, fill_value=0)
        raw_weights = t.continuity_block.map(counts).to_numpy(float)
        w = raw_weights * (len(t)/raw_weights.sum())
        np.testing.assert_allclose(w.sum(), len(t))
        weights[f'w_{seed}'] = w
        multiplicities[f'count_{seed}'] = counts.to_numpy()
        for arm in ('control', 'bootstrap'):
            key = f'{arm}_{seed}'
            matrix = xgb.DMatrix(t[names].astype(float), weight=None if arm == 'control' else w)
            matrix.set_float_info('label_lower_bound', t.lower_bound_hours)
            matrix.set_float_info('label_upper_bound', t.upper_bound_hours)
            model = xgb.train({**protocol['params'], 'seed': seed}, matrix, num_boost_round=500)
            p = model.predict(inference)
            if key == 'control_42':
                np.testing.assert_array_equal(p, baseline)
            model.save_model(out / f'{key}.ubj')
            predictions[key] = p
            scored[key] = pd.Series(p, index=v.timestamp).loc[scored.timestamp].to_numpy()
            scores[key] = score_predictions(scored, scored[key].to_numpy())
            print(key, 'completed', flush=True)
    for arm in ('control', 'bootstrap'):
        key = arm+'_median'
        predictions[key] = predictions[[f'{arm}_{seed}' for seed in range(42, 47)]].median(axis=1)
        scored[key] = pd.Series(predictions[key].to_numpy(), index=v.timestamp).loc[scored.timestamp].to_numpy()
        scores[key] = score_predictions(scored, scored[key].to_numpy())
    paired = {n: paired_error_summary(scored, scored[n], scored.bootstrap_median)
              for n in ('control_42', 'control_median')}
    candidate = scores['bootstrap_median']
    gates = {n: candidate['exact_row_mae'] < scores[n]['exact_row_mae']
             and candidate['exact_block_mae'] < scores[n]['exact_block_mae']
             and candidate['tail24_block_mae'] <= scores[n]['tail24_block_mae']
             and paired[n]['all_row_delta_upper'] <= 0 and paired[n]['all_block_delta_upper'] <= 0 for n in paired}
    weights.to_parquet(out / 'weights.parquet', index=False)
    multiplicities.to_parquet(out / 'multiplicities.parquet', index=False)
    predictions.to_parquet(out / 'predictions.parquet', index=False)
    scored.to_parquet(out / 'scored.parquet', index=False)
    for p, h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    report = {'scores': scores, 'paired': paired, 'gates': gates, 'pass': all(gates.values()),
        'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'scores': {n:scores[n] for n in ['control_42','control_median','bootstrap_median']},
                      'paired': paired, 'gates': gates}, indent=2), flush=True)


if __name__ == '__main__':
    main()
