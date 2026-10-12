"""Predeclared top-three training-block deletions; no forecast error selection."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]


def main():
    b = ROOT / 'artifacts/task01_recovery_head'
    out = b / 'training_process_influence_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [b / 'gefs_2021_extension_v1/f024_training_rows.parquet',
             b / 'gefs_2021_extension_v1/inference_rows.parquet',
             b / 'grouped_forecast_aft_2023_20260916/protocol.json',
             b / 'reusable_gefs_aft_v1/model.joblib', Path(__file__).resolve(),
             ROOT / 'training_process_influence_plan_20260918.md',
             b / 'interval_business_labels_v1/origins.parquet']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    train, valid = [pd.read_parquet(p) for p in paths[:2]]
    identities = pd.read_parquet(paths[-1], columns=['timestamp', 'continuity_block'])
    valid = valid.merge(identities, on='timestamp', how='left', validate='one_to_one')
    assert valid.continuity_block.notna().all()
    frozen = json.loads(paths[2].read_bytes())
    for p, h in frozen['input_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    names, params = frozen['features'], frozen['params']
    counts = train.groupby('continuity_block').size().rename('rows').reset_index()
    selected = counts.sort_values(['rows', 'continuity_block'], ascending=[False, True]).head(3)
    cut = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(cut).all() and train.label_available_at.lt(cut).all()
    assert not set(train.continuity_block) & set(valid.continuity_block)
    baseline = joblib.load(paths[3]).predict(valid[names])
    inference = xgb.DMatrix(valid[names].astype(float))
    predictions = valid[['timestamp', 'continuity_block']].copy()
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({'input_sha256': hashes,
        'selected': selected.to_dict('records'), 'features': names, 'params': params,
        'rounds': 500, 'diagnostic_only': True}, indent=2), encoding='utf-8')
    results = {}
    for block in [None, *selected.continuity_block.tolist()]:
        label = 'native' if block is None else f'without_{block}'
        rows = train if block is None else train.loc[train.continuity_block != block]
        matrix = xgb.DMatrix(rows[names].astype(float))
        matrix.set_float_info('label_lower_bound', rows.lower_bound_hours)
        matrix.set_float_info('label_upper_bound', rows.upper_bound_hours)
        model = xgb.train(params, matrix, num_boost_round=500)
        p = model.predict(inference)
        if block is None:
            np.testing.assert_array_equal(p, baseline)
        model.save_model(out / f'{label}.ubj')
        predictions[label] = p
        delta = p.astype(float)-baseline
        absolute = np.abs(delta)
        results[label] = {'training_rows': len(rows), 'mean_absolute_change': float(absolute.mean()),
            'median_absolute_change': float(np.median(absolute)),
            'p95_absolute_change': float(np.quantile(absolute, .95)),
            'max_absolute_change': float(absolute.max()), 'mean_signed_change': float(delta.mean()),
            'long_status_flips': int(((p >= 24) != (baseline >= 24)).sum()),
            'increased_rows': int((delta > 0).sum()), 'decreased_rows': int((delta < 0).sum())}
        print(json.dumps({label: results[label]}), flush=True)
    predictions.to_parquet(out / 'predictions.parquet', index=False)
    detail = predictions.copy()
    for n in results:
        detail[n] = (detail[n]-detail.native).abs() if n != 'native' else detail[n]
    detail.groupby('continuity_block')[[n for n in results if n != 'native']].mean().to_parquet(out / 'block_changes.parquet')
    for p, h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    result = {'results': results, 'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
