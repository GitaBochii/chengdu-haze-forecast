"""Fixed training-only AFT leaf proximity, keeping 15 distinct donor processes."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from scripts.benchmark_observed_business_aft import ROOT, sha
from scripts.benchmark_process_analog import interval_median


def main():
    source = ROOT / 'artifacts/task01_recovery_head/transition_earlier_recovery_v1'
    out = ROOT / 'artifacts/task01_recovery_head/leaf_process_analog_2020_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / '2020_training.parquet', source / '2020_predictions.parquet',
             source / '2020_aft.joblib', source / 'protocol.json', Path(__file__),
             ROOT / 'scripts/benchmark_process_analog.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    names = json.loads(paths[3].read_bytes())['features']
    out.mkdir()
    protocol = {'input_sha256': hashes, 'neighbors': 15,
                'distance': 'fraction of 500 training-only AFT trees with different leaves',
                'year': 2020, 'features': names, 'deployment_authorized': False,
                'gate': 'row and process MAE decrease; tail24 process MAE does not increase',
                'limitations': ['Reused development year', 'No k search, no weights, no blend',
                                'Representation fitted on donor labels; not out-of-fold donors']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    train, valid = pd.read_parquet(paths[0]), pd.read_parquet(paths[1])
    cut = pd.Timestamp('2020-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp('2021-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(cut).all() and train.label_available_at.lt(cut).all()
    assert valid.timestamp.ge(cut).all() and valid.label_available_at.lt(end).all()
    assert not set(train.continuity_block) & set(valid.continuity_block)
    model = joblib.load(paths[2])
    baseline = model.predict(valid[names])
    np.testing.assert_array_equal(baseline, valid.paired_aft)
    leaves = [model.booster.predict(xgb.DMatrix(f[names].astype(float)), pred_leaf=True)
              for f in (train, valid)]
    assert leaves[0].shape == (len(train), 500) and leaves[1].shape == (len(valid), 500)
    predictions, records = [], []
    for i, row in enumerate(leaves[1]):
        distances = (leaves[0] != row).mean(axis=1)
        donors = train.assign(distance=distances).sort_values('distance', kind='stable')
        selected = donors.drop_duplicates('continuity_block').head(15)
        assert len(selected) == 15
        predictions.append(interval_median(selected.lower_bound_hours, selected.upper_bound_hours))
        for rank, d in enumerate(selected.itertuples(), 1):
            records.append({'query_timestamp': valid.iloc[i].timestamp, 'rank': rank,
                            'donor_timestamp': d.timestamp, 'block': d.continuity_block,
                            'distance': d.distance, 'lower': d.lower_bound_hours,
                            'upper': d.upper_bound_hours})
    valid = valid.copy()
    valid['leaf_analog_prediction'] = predictions
    valid.to_parquet(out / 'predictions.parquet', index=False)
    pd.DataFrame(records).to_parquet(out / 'neighbors.parquet', index=False)
    control, candidate = score_predictions(valid, baseline), score_predictions(valid, predictions)
    checks = {k: candidate[k] < control[k] for k in ('exact_row_mae', 'exact_block_mae')}
    checks['tail24_block_mae'] = candidate['tail24_block_mae'] <= control['tail24_block_mae']
    assert hashes == {str(p): sha(p) for p in paths}
    report = {'control': control, 'candidate': candidate, 'checks': checks,
              'passed': all(checks.values()), 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
