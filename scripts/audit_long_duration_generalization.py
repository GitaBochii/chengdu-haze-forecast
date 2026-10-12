"""Frozen-model training fit and difficult-process attribution, not a new model."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    old = base / 'reusable_gefs_aft_v1'
    trial = base / 'grouped_forecast_aft_2023_20260916'
    out = base / 'long_duration_generalization_20260916'
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads((trial / 'protocol.json').read_bytes())
    names = protocol['features']
    inputs = dict(protocol['input_sha256'])
    for folder in (trial, old):
        report = json.loads((folder / 'report.json').read_bytes())
        for name, digest in report['output_sha256'].items():
            if Path(name).name != name:
                raise ValueError('Invalid artifact filename')
            path = folder / name
            if str(path) in inputs and inputs[str(path)] != digest:
                raise ValueError('Conflicting manifests')
            inputs[str(path)] = digest
    inputs[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for path, digest in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Input changed: {path}')
    model = joblib.load(old / 'model.joblib')
    train = pd.read_parquet(base / 'gefs_2021_extension_v1/f024_training_rows.parquet')
    val = pd.read_parquet(base / 'gefs_2021_extension_v1/inference_rows.parquet')
    scored = pd.read_parquet(trial / 'expanded.parquet')
    train['prediction'] = model.predict(train[names])
    predicted = pd.Series(model.predict(val[names]), index=val.timestamp)
    scored['prediction'] = predicted.loc[scored.timestamp].to_numpy()
    np.testing.assert_array_equal(scored.prediction, scored.forecast_aft)
    summaries = {}
    for key, frame in [('training_in_sample', train), ('2023_expanded', scored)]:
        exact = frame.loc[frame.exact_hours.notna()].copy()
        exact['error'] = abs(exact.prediction - exact.exact_hours)
        long = exact.loc[exact.exact_hours.ge(24)]
        summaries[key] = {
            'rows': len(frame), 'exact_rows': len(exact), 'long_exact_rows': len(long),
            'long_exact_blocks': int(long.continuity_block.nunique()),
            'max_prediction': float(frame.prediction.max()),
            'exact_block_mae': float(exact.groupby('continuity_block').error.mean().mean()),
            'long_predicted_at_least_24_fraction': float(long.prediction.ge(24).mean()),
            'long_actual_median': float(long.exact_hours.median()),
            'long_prediction_median': float(long.prediction.median()),
        }
    matrix = xgb.DMatrix(val[names].astype(float), missing=np.nan)
    contribution = model.booster.predict(matrix, pred_contribs=True)
    margin = model.booster.predict(matrix, output_margin=True)
    np.testing.assert_allclose(contribution.sum(axis=1), margin, rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.exp(margin), predicted.to_numpy(), rtol=1e-6, atol=1e-6)
    attribution = pd.DataFrame(contribution[:, :-1], columns=names)
    attribution.insert(0, 'timestamp', val.timestamp.to_numpy())
    processes = {}
    for block in (1249, 1258, 1323):
        rows = scored.loc[scored.continuity_block.eq(block)]
        selected = val.timestamp.isin(rows.timestamp).to_numpy()
        means = pd.Series(contribution[selected, :-1].mean(axis=0), index=names)
        processes[str(block)] = {
            'rows': len(rows), 'prediction_median': float(rows.prediction.median()),
            'prediction_max': float(rows.prediction.max()),
            'forecast_mean_log_contribution': float(means.iloc[92:].sum()),
            'lowest_mean_log_contributions': means.nsmallest(7).astype(float).to_dict(),
        }
    for path, digest in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Input changed during audit: {path}')
    out.mkdir()
    attribution.to_parquet(out / 'inference_contributions.parquet', index=False)
    result = {'input_sha256': inputs, 'summaries': summaries, 'processes': processes,
              'max_contribution_sum_error': float(abs(contribution.sum(axis=1) - margin).max()),
              'deployment_authorized': False,
              'limitations': ['In-sample training fit is not generalization evidence',
                              'Training and evaluation populations differ; no causal overfitting proof',
                              'Future-duration grouping is retrospective only and is not a routing rule',
                              'Long exact validation rows come from only three processes',
                              'Tree contributions are log-time attributions, not hours or causal effects',
                              'Named difficult processes were selected after observing errors',
                              'No tuning, new training, or feature removal'],
              'output_sha256': {'inference_contributions.parquet': hashlib.sha256(
                  (out / 'inference_contributions.parquet').read_bytes()).hexdigest()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'summaries': summaries, 'processes': processes}, indent=2))


if __name__ == '__main__':
    main()
