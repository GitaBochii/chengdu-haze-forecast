"""Replay frozen 2025 origins individually with explicit archive-proxy provenance."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    old = base / 'reusable_gefs_crossyear_v1'
    snapshots = base / 'reusable_gefs_snapshots_v1'
    out = base / 'historical_shadow_2025_v1'
    if out.exists():
        raise FileExistsError(out)
    evidence = {}
    for folder in (old, snapshots):
        report = json.loads((folder / 'report.json').read_bytes())
        for name, expected in report['output_sha256'].items():
            if Path(name).name != name or sha(folder / name) != expected:
                raise ValueError('Frozen input changed')
            evidence[str(folder / name)] = expected
    names = json.loads((old / 'protocol.json').read_bytes())['features']
    x = pd.read_parquet(old / '2025_features.parquet')
    original = pd.read_parquet(old / '2025_predictions.parquet')
    labels_path = base / 'interval_business_labels_v1/origins.parquet'
    labels_report = json.loads((labels_path.parent / 'report.json').read_bytes())
    if sha(labels_path) != labels_report['output_sha256']:
        raise ValueError('Canonical labels changed')
    evidence[str(labels_path)] = sha(labels_path)
    evidence[str(Path(__file__).resolve())] = sha(Path(__file__))
    labels = pd.read_parquet(labels_path).set_index('timestamp').loc[original.timestamp].reset_index()
    forecasts = pd.read_parquet(snapshots / 'features.parquet').set_index('timestamp')
    if not original.timestamp.is_unique or not original.timestamp.is_monotonic_increasing:
        raise ValueError('Origins must be unique and chronological')
    if list(x.columns) != names or len(x) != len(original):
        raise ValueError('Model schema mismatch')
    aligned = forecasts.loc[original.timestamp].reset_index()
    weather_names = [n for n in names if n.startswith(('gefs_', 'forecast_'))]
    np.testing.assert_allclose(x[weather_names], aligned[weather_names], equal_nan=True)
    known = aligned.known_at.notna()
    if not (aligned.loc[known, 'known_at'].le(aligned.loc[known, 'timestamp']).all()
            and aligned.loc[known, 'valid_at'].gt(aligned.loc[known, 'timestamp']).all()
            and aligned.loc[known, 'run_at'].le(aligned.loc[known, 'known_at']).all()):
        raise ValueError('Forecast proxy chronology failed')
    if not labels.label_available_at.lt(pd.Timestamp('2026-01-01', tz='Asia/Shanghai')).all():
        raise ValueError('Scoring labels not mature within year')
    model = joblib.load(old / '2025.joblib')
    # Each call receives only its frozen input row, never labels or later rows.
    prediction = np.array([float(model.predict(x.iloc[[i]])[0]) for i in range(len(x))])
    np.testing.assert_array_equal(prediction, original.forecast_aft)
    result = labels[['timestamp', 'continuity_block', 'lower_bound_hours',
                     'upper_bound_hours', 'exact_hours', 'label_available_at']].copy()
    result['prediction'] = prediction
    result['forecast_status'] = np.where(known, 'archive_proxy_available', 'missing_forecast')
    result['input_missing_count'] = x.isna().sum(axis=1)
    for column in ('known_at', 'run_at', 'valid_at'):
        result[column] = aligned[column]
    metrics = score_predictions(result, prediction)
    protocol = {
        'mode': 'historical_shadow_only', 'year': 2025, 'features': names,
        'model_sha256': sha(old / '2025.joblib'), 'input_sha256': evidence,
        'label_policy': 'Canonical interval labels, used only after row predictions',
        'input_policy': 'Frozen annual 104-feature matrix; no newly received 2026 forecasts',
        'limitations': ['Reused historical evaluation year, not a prospective trial',
                        'Archive proxy timestamps, not actual historical local receipts',
                        'Precomputed feature causality relies on existing source audits',
                        'Missing model inputs remain NaN as in training; explicitly flagged',
                        'This cannot demonstrate real-time inference without live pollution inputs'],
        'deployment_authorized': False,
    }
    if evidence != {p: sha(Path(p)) for p in evidence}:
        raise ValueError('Input changed during replay')
    out.mkdir()
    result.to_parquet(out / 'predictions.parquet', index=False)
    x.to_parquet(out / 'inputs.parquet', index=False)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    summary = {'scores': metrics, 'forecast_status': result.forecast_status.value_counts().to_dict(),
               'individual_replay_exact_match': True,
               'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
