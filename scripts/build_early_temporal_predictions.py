"""Frozen 2017-2019 expanding-window predictions for training-only diagnosis."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    source = ROOT / 'artifacts/task01_recovery_head/transition_earlier_recovery_v1'
    out = ROOT / 'artifacts/task01_recovery_head/early_temporal_predictions_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / '2020_training.parquet', source / 'protocol.json', Path(__file__),
             ROOT / 'src/haze_forecast/observed_aft.py']
    before = {str(p): sha(p) for p in paths}
    data = pd.read_parquet(paths[0])
    names = json.loads(paths[1].read_bytes())['features']
    out.mkdir()
    protocol = {'years': [2017, 2018, 2019], 'seed': 42, 'rounds': 500, 'features': names,
                'input_sha256': before, 'policy': 'train origins and label maturity strictly before year; exclude evaluation blocks',
                'limitations': ['Source is the frozen pre-2020 cohort, not a new population',
                                'Source eligibility already conditioned on label maturity before 2020',
                                'Each output includes label maturity; downstream must filter by its own cutoff',
                                'No residual fitting, model selection or deployment in this run']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    reports, outputs = {}, []
    for year in (2017, 2018, 2019):
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
        valid = data.loc[data.timestamp.ge(cut) & data.timestamp.lt(end)].copy()
        train = data.loc[data.timestamp.lt(cut) & data.label_available_at.lt(cut)
                         & ~data.continuity_block.isin(valid.continuity_block)].copy()
        assert len(train) and len(valid)
        assert not set(train.continuity_block) & set(valid.continuity_block)
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
            train[names], train.lower_bound_hours, train.upper_bound_hours)
        joblib.dump(model, out / f'{year}_model.joblib')
        pred = model.predict(valid[names])
        np.testing.assert_array_equal(pred, joblib.load(out / f'{year}_model.joblib').predict(valid[names]))
        train[['timestamp', 'continuity_block', 'label_available_at']].to_parquet(out / f'{year}_training_members.parquet', index=False)
        valid['temporal_prediction'] = pred
        valid['model_cutoff'] = cut
        outputs.append(valid)
        reports[str(year)] = {'training_rows': len(train), 'training_blocks': int(train.continuity_block.nunique()),
                             'scoring': score_predictions(valid, pred),
                             'labels_mature_at_next_year': int(valid.label_available_at.lt(end).sum())}
        print(json.dumps({str(year): reports[str(year)]}, indent=2), flush=True)
    combined = pd.concat(outputs, ignore_index=True)
    assert combined.timestamp.is_unique and np.isfinite(combined.temporal_prediction).all()
    combined.to_parquet(out / 'predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'years': reports, 'total_predictions': len(combined), 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
