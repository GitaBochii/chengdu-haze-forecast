"""Monthly AFT refits using mature labels only, versus frozen annual predictions."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'transition_earlier_recovery_v1'
    out = base / 'monthly_refresh_2018_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / '2020_training.parquet', source / 'protocol.json',
             base / 'early_temporal_predictions_v1/predictions.parquet', Path(__file__),
             ROOT / 'src/haze_forecast/observed_aft.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    data = pd.read_parquet(paths[0])
    valid = pd.read_parquet(paths[2])
    valid = valid.loc[valid.timestamp.dt.year.eq(2018)].copy()
    names = json.loads(paths[1].read_bytes())['features']
    out.mkdir()
    protocol = {'input_sha256': hashes, 'year': 2018, 'cadence': 'month start',
                'features': names, 'seed': 42, 'rounds': 500,
                'gate': 'row and process MAE decrease; tail24 process MAE does not increase',
                'limitations': ['Retrospective development, no deployment',
                                'Mature labels and no overlap with forecast process required',
                                'No cadence search or calibration',
                                'Same frozen source population as annual baseline']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    parts, months = [], {}
    for month, query in valid.groupby(valid.timestamp.dt.month):
        cut = pd.Timestamp(f'2018-{month:02d}-01', tz='Asia/Shanghai')
        train = data.loc[data.timestamp.lt(cut) & data.label_available_at.lt(cut)
                         & ~data.continuity_block.isin(query.continuity_block)].copy()
        assert not set(train.continuity_block) & set(query.continuity_block)
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
            train[names], train.lower_bound_hours, train.upper_bound_hours)
        joblib.dump(model, out / f'{month:02d}_model.joblib')
        pred = model.predict(query[names])
        np.testing.assert_array_equal(pred, joblib.load(out / f'{month:02d}_model.joblib').predict(query[names]))
        train[['timestamp', 'continuity_block', 'label_available_at']].to_parquet(out / f'{month:02d}_members.parquet', index=False)
        query = query.copy()
        query['monthly_prediction'] = pred
        query['monthly_cutoff'] = cut
        parts.append(query)
        months[str(month)] = {'train_rows': len(train), 'query_rows': len(query)}
    result = pd.concat(parts).sort_values('timestamp').reset_index(drop=True)
    pd.testing.assert_series_equal(result.timestamp, valid.timestamp.reset_index(drop=True))
    assert len(result) == len(valid) and np.isfinite(result.monthly_prediction).all()
    result.to_parquet(out / 'predictions.parquet', index=False)
    control = score_predictions(result, result.temporal_prediction)
    candidate = score_predictions(result, result.monthly_prediction)
    checks = {k: candidate[k] < control[k] for k in ('exact_row_mae', 'exact_block_mae')}
    checks['tail24_block_mae'] = candidate['tail24_block_mae'] <= control['tail24_block_mae']
    assert hashes == {str(p): sha(p) for p in paths}
    report = {'months': months, 'control': control, 'candidate': candidate,
              'checks': checks, 'passed': all(checks.values()), 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'output_sha256'}, indent=2))


if __name__ == '__main__':
    main()
