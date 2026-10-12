"""Re-score three frozen development models on one canonical label cohort."""

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
    old = base / 'reusable_gefs_aft_v1'
    aug = base / 'weather_dropout_development_v1'
    out = base / 'unified_scorecard_2023_v1'
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads((aug / 'protocol.json').read_bytes())
    report = json.loads((aug / 'report.json').read_bytes())
    recorded = {Path(k): v for k, v in protocol['input_sha256'].items()}
    recorded.update({aug / k: v for k, v in report['output_sha256'].items()})
    for path, expected in recorded.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    label_path = base / 'interval_business_labels_v1/origins.parquet'
    labels = pd.read_parquet(label_path)
    source = pd.read_parquet(old / 'expanded_predictions.parquet')
    if not labels.timestamp.is_unique or not source.timestamp.is_unique:
        raise ValueError('Duplicate origin')
    canonical = labels.set_index('timestamp').loc[source.timestamp].reset_index()
    for column in ['lower_bound_hours', 'upper_bound_hours', 'exact_hours']:
        np.testing.assert_allclose(source[column], canonical[column], equal_nan=True)
    np.testing.assert_array_equal(source.continuity_block, canonical.continuity_block)
    if source.score_allowed.dtype != bool:
        raise ValueError('Invalid scoring mask')
    mask = source.score_allowed.to_numpy()
    cohort = canonical.loc[mask].reset_index(drop=True)
    end = pd.Timestamp('2024-01-01', tz='Asia/Shanghai')
    start = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    if not (cohort.timestamp.ge(start).all() and cohort.timestamp.lt(end).all()
            and cohort.label_available_at.lt(end).all()):
        raise ValueError('Invalid development clock')
    val = pd.read_parquet(aug / 'inference.parquet')
    pd.testing.assert_series_equal(val.timestamp, source.timestamp)
    paths = {'f024': old / 'model.joblib',
             'duplicate_control': aug / 'duplicate_control.joblib',
             'weather_masked': aug / 'weather_masked.joblib'}
    scores = {}
    replay = cohort[['timestamp', 'continuity_block']].copy()
    for arm, path in paths.items():
        prediction = joblib.load(path).predict(val[protocol['features']])
        saved = (source.forecast_aft if arm == 'f024' else
                 pd.read_parquet(aug / f'{arm}_expanded.parquet').dropout_candidate)
        np.testing.assert_array_equal(prediction, saved)
        scores[arm] = score_predictions(cohort, prediction[mask])
        prior = (report['scores']['duplicate_control']['expanded']['f024_control']
                 if arm == 'f024' else report['scores'][arm]['expanded']['candidate'])
        for new, legacy in [('exact_row_mae', 'row_mae'), ('exact_block_mae', 'block_mae'),
                            ('tail24_block_mae', 'tail_24h_block_mae')]:
            np.testing.assert_allclose(scores[arm][new], prior[legacy], rtol=0, atol=1e-12)
        replay[arm] = prediction[mask]
    evidence = list(recorded) + [label_path, aug / 'report.json', Path(__file__).resolve(),
                                 ROOT / 'src/haze_forecast/cohort_scorecard.py']
    manifest = {
        'scope': '2023 reused development cohort; no model selection or deployment',
        'cohort_policy': 'Frozen expanded score_allowed, canonical interval labels',
        'training_policy': 'Pre-2023 mature labels; original f024 forecast support',
        'features': protocol['features'],
        'model_sha256': {k: sha(v) for k, v in paths.items()},
        'input_sha256': {str(p): sha(p) for p in evidence},
        'deployment_authorized': False,
        'limitations': ['Not an untouched holdout', 'No new input availability evidence',
                        'Error bounds assume label intervals are correct; not confidence intervals',
                        'Only three matched models; not a full project ranking'],
    }
    out.mkdir()
    cohort.to_parquet(out / 'cohort.parquet', index=False)
    replay.to_parquet(out / 'predictions.parquet', index=False)
    (out / 'protocol.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    result = {'scores': scores, 'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
