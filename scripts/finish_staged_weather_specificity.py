"""Finish the saved diagnostic after a verifier dtype mismatch; do not retrain."""

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
    out = b / 'staged_weather_specificity_20260918'
    if (out / 'report.json').exists():
        raise FileExistsError(out / 'report.json')
    c = json.loads((out / 'protocol.json').read_bytes())
    for p,h in c['input_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    stage = b / 'staged_forecast_aft_2022_20260918'
    original = json.loads((stage / 'protocol.json').read_bytes())
    v = pd.read_parquet(stage / 'predictions.parquet')
    x = v[original['weather']].astype(float).copy()
    supported = x.notna().all(axis=1).to_numpy()
    for n in c['masked_fields']:
        x.loc[x[n].notna(), n] = 0.
    x['base_log_hours'] = v.base_margin.to_numpy(dtype=float)
    prior, support2 = head_inputs(v, original['weather'], v.base_margin, True)
    prior[c['masked_fields']] = prior[c['masked_fields']].where(prior[c['masked_fields']].isna(), 0.)
    pd.testing.assert_frame_equal(x, prior)
    np.testing.assert_array_equal(supported, support2)
    assert int((~supported).sum()) == 47
    model = xgb.Booster()
    model.load_model(out / 'metadata_head.ubj')
    prediction = model.predict(xgb.DMatrix(x, base_margin=v.base_margin))
    chunks = [model.predict(xgb.DMatrix(x.iloc[i:i+37], base_margin=v.base_margin.iloc[i:i+37]))
              for i in range(0, len(v), 37)]
    np.testing.assert_array_equal(prediction, np.concatenate(chunks))
    v['metadata'] = np.where(supported, prediction, v.base)
    np.testing.assert_array_equal(v.loc[~supported, 'metadata'], v.loc[~supported, 'base'])
    scores, errors = {}, {}
    assert v.exact_hours.notna().all() and (v.exact_hours == v.lower_bound_hours).all()
    assert (v.exact_hours == v.upper_bound_hours).all()
    for name in ('weather', 'metadata'):
        error = pd.Series([abs(float(p)-float(y)) for p,y in zip(v[name], v.exact_hours)], index=v.index)
        np.testing.assert_allclose(error, (v[name]-v.exact_hours).abs(), atol=1e-12)
        tail = v.exact_hours.ge(24)
        scores[name] = [float(error.mean()), float(error.groupby(v.continuity_block).mean().mean()),
                       float(error[tail].groupby(v.loc[tail, 'continuity_block']).mean().mean())]
        errors[name] = error
    delta = (errors['weather']-errors['metadata']).groupby(v.continuity_block).mean()
    v.to_parquet(out / 'predictions.parquet', index=False)
    report = {'scores_row_block_tail': scores, 'weather_improved_blocks': int(delta.lt(0).sum()),
              'weather_worse_blocks': int(delta.gt(0).sum()), 'block_deltas': delta.to_dict(),
              'replayed_predictions': len(v), 'fallback_rows': int((~supported).sum()),
              'recovery': 'Original verifier compared float32 to float64 margin columns; recreated float64 input. Saved fitted head unchanged; no retraining.',
              'recovery_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
