"""Fixed three-year training window versus all-history AFT, gated by year."""

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
    out = ROOT / 'artifacts/task01_recovery_head/recent_three_years_v1'
    if out.exists():
        raise FileExistsError(out)
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    paths = [Path(__file__), source / 'protocol.json', ROOT / 'src/haze_forecast/observed_aft.py']
    paths += [source / f'{year}_{suffix}' for year in (2020, 2021)
              for suffix in ('training.parquet', 'predictions.parquet', 'aft.joblib')]
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    out.mkdir()
    protocol = {'years': [2020, 2021], 'lookback_years': 3, 'seed': 42, 'rounds': 500,
                'features': names, 'input_sha256': hashes,
                'gate': 'exact row/process MAE decrease; tail24 process MAE does not increase',
                'window': 'Keep whole blocks whose first available training origin is within window',
                'limitations': ['Previously inspected development years',
                                'Reduced sample size and changed source mix both change',
                                'No window search; no deployment authorized']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    scores = {}
    for year in (2020, 2021):
        train = pd.read_parquet(source / f'{year}_training.parquet')
        valid = pd.read_parquet(source / f'{year}_predictions.parquet')
        start = pd.Timestamp(f'{year-3}-01-01', tz='Asia/Shanghai')
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
        assert train.timestamp.lt(cut).all() and train.label_available_at.lt(cut).all()
        assert valid.timestamp.ge(cut).all() and valid.label_available_at.lt(end).all()
        assert not set(train.continuity_block) & set(valid.continuity_block)
        block_start = train.groupby('continuity_block').timestamp.transform('min')
        recent = train.loc[block_start.ge(start)].copy()
        assert recent.timestamp.ge(start).all() and len(recent) > 0
        recent.to_parquet(out / f'{year}_training.parquet', index=False)
        base = joblib.load(source / f'{year}_aft.joblib').predict(valid[names])
        np.testing.assert_array_equal(base, valid.paired_aft)
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
            recent[names], recent.lower_bound_hours, recent.upper_bound_hours)
        joblib.dump(model, out / f'{year}_model.joblib')
        pred = model.predict(valid[names])
        np.testing.assert_array_equal(pred, joblib.load(out / f'{year}_model.joblib').predict(valid[names]))
        result = valid.copy()
        result['recent_prediction'] = pred
        result.to_parquet(out / f'{year}_predictions.parquet', index=False)
        control, candidate = score_predictions(valid, base), score_predictions(valid, pred)
        checks = {k: candidate[k] < control[k] for k in ('exact_row_mae', 'exact_block_mae')}
        checks['tail24_block_mae'] = candidate['tail24_block_mae'] <= control['tail24_block_mae']
        scores[str(year)] = {'training_rows': len(recent), 'training_blocks': int(recent.continuity_block.nunique()),
                             'all_history_training_rows': len(train), 'control': control, 'candidate': candidate,
                             'checks': checks, 'passed': all(checks.values())}
        (out / f'{year}_decision.json').write_text(json.dumps(scores[str(year)], indent=2), encoding='utf-8')
        print(json.dumps({str(year): scores[str(year)]}, indent=2), flush=True)
        if not all(checks.values()):
            break
    assert hashes == {str(p): sha(p) for p in paths}
    report = {'scores': scores, 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
