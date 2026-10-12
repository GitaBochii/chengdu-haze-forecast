"""Fixed process-diverse retrieval baseline on the 2020 development cohort."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from scripts.benchmark_observed_business_aft import ROOT, sha


def interval_median(lower, upper):
    """Choose the midpoint of the minimizer set of summed interval distances."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if lower.ndim != 1 or not len(lower) or lower.shape != upper.shape:
        raise ValueError('Aligned nonempty vectors required')
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (upper < lower).any():
        raise ValueError('Finite ordered intervals required')
    endpoints = np.sort(np.concatenate([lower, upper]))
    n = len(lower)
    return float((endpoints[n - 1] + endpoints[n]) / 2)


def main():
    source = ROOT / 'artifacts/task01_recovery_head/transition_earlier_recovery_v1'
    out = ROOT / 'artifacts/task01_recovery_head/process_analog_2020_v2'
    if out.exists():
        raise FileExistsError(out)
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    paths = [source / '2020_training.parquet', source / '2020_predictions.parquet',
             source / '2020_aft.joblib', source / 'protocol.json', Path(__file__)]
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    out.mkdir()
    protocol = {'year': 2020, 'neighbors': 15, 'distance': 'training IQR scaled mean square',
                'process_policy': 'nearest row per training process, then nearest 15 processes',
                'prediction': 'midpoint of interval absolute-distance minimizer set',
                'features': names, 'input_sha256': hashes, 'deployment_authorized': False,
                'gate': 'row and process MAE decrease; tail24 process MAE does not increase',
                'limitations': ['Development cohort previously inspected',
                                'Training-only median imputation is a retrieval convention',
                                'Midpoint of minimizer set is a prediction, not a label imputation',
                                'No blending, neighbor search or validation-outcome-based routing']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    train, valid = (pd.read_parquet(source / f'2020_{name}.parquet')
                    for name in ('training', 'predictions'))
    cut = pd.Timestamp('2020-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp('2021-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(cut).all() and train.label_available_at.lt(cut).all()
    assert valid.timestamp.ge(cut).all() and valid.label_available_at.lt(end).all()
    assert not set(train.continuity_block) & set(valid.continuity_block)
    numeric = train[names].astype(float)
    med = numeric.median()
    scale = numeric.quantile(.75) - numeric.quantile(.25)
    kept = scale.index[scale.gt(0) & med.notna()].tolist()
    reference = ((train[kept].fillna(med) - med[kept]) / scale[kept]).to_numpy(float)
    query = ((valid[kept].fillna(med) - med[kept]) / scale[kept]).to_numpy(float)
    assert np.isfinite(reference).all() and np.isfinite(query).all()
    predictions, neighbors = [], []
    for i, row in enumerate(query):
        distance = np.square(reference - row).mean(axis=1)
        ranked = train.assign(distance=distance).sort_values('distance', kind='stable')
        selected = ranked.drop_duplicates('continuity_block').head(15)
        assert len(selected) == 15
        predictions.append(interval_median(selected.lower_bound_hours, selected.upper_bound_hours))
        for rank, donor in enumerate(selected.itertuples(), 1):
            neighbors.append({'query_timestamp': valid.iloc[i].timestamp, 'rank': rank,
                              'donor_timestamp': donor.timestamp, 'block': donor.continuity_block,
                              'distance': donor.distance, 'lower': donor.lower_bound_hours,
                              'upper': donor.upper_bound_hours})
    baseline = joblib.load(source / '2020_aft.joblib').predict(valid[names])
    np.testing.assert_array_equal(baseline, valid.paired_aft)
    result = valid.copy()
    result['process_analog_prediction'] = predictions
    result.to_parquet(out / 'predictions.parquet', index=False)
    pd.DataFrame(neighbors).to_parquet(out / 'neighbors.parquet', index=False)
    control, candidate = score_predictions(valid, baseline), score_predictions(valid, predictions)
    checks = {k: candidate[k] < control[k] for k in ('exact_row_mae', 'exact_block_mae')}
    checks['tail24_block_mae'] = candidate['tail24_block_mae'] <= control['tail24_block_mae']
    assert hashes == {str(p): sha(p) for p in paths}
    report = {'control': control, 'candidate': candidate, 'checks': checks,
              'passed': all(checks.values()), 'retained_features': kept,
              'dropped_constant_or_missing': sorted(set(names) - set(kept)),
              'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('output_sha256', 'retained_features')}, indent=2))


if __name__ == '__main__':
    main()
