"""Fixed six-column SO2 removal; 2021 runs only after the 2020 gate."""

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
    out = base / 'so2_ablation_v1'
    if out.exists():
        raise FileExistsError(out)
    old = json.loads((source / 'report.json').read_bytes())
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    removed = ['so2', 'so2_lag_1', 'so2_lag_3', 'so2_lag_24', 'so2_roll_3', 'so2_roll_24']
    kept = [n for n in names if n not in removed]
    assert len(names) == 104 and len(kept) == 98 and set(removed).issubset(names)
    paths = [source / 'report.json', source / 'protocol.json', Path(__file__),
              ROOT / 'docs/so2_ablation_design_20260911.md',
              ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py',
              ROOT / 'src/haze_forecast/cohort_scorecard.py', ROOT / 'src/haze_forecast/app.py']
    paths += [p for p in (ROOT / 'artifacts/cnemc_2025_deployment_candidate').iterdir() if p.is_file()]
    before = {str(p): sha(p) for p in paths}
    out.mkdir()
    protocol = {'input_sha256': before, 'features': kept, 'removed': removed,
                'seed': 42, 'rounds': 500, 'years_if_passed': [2020, 2021],
                'gate': 'Exact row and block MAE strictly decrease; tail24 block MAE does not increase',
                'limitations': ['Reused historical development years; hypothesis partly from 2022',
                                'Removing columns also changes feature subsampling',
                                'No calibration, label changes or deployment']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    scores = {}
    for year in (2020, 2021):
        files = [f'{year}_training.parquet', f'{year}_predictions.parquet', f'{year}_aft.joblib']
        for name in files:
            assert sha(source / name) == old['output_sha256'][name]
            before[str(source / name)] = sha(source / name)
        tr = pd.read_parquet(source / files[0])
        va = pd.read_parquet(source / files[1])
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
        assert tr.timestamp.lt(cut).all() and tr.label_available_at.lt(cut).all()
        assert va.timestamp.ge(cut).all() and va.timestamp.lt(end).all() and va.label_available_at.lt(end).all()
        assert not set(tr.continuity_block) & set(va.continuity_block)
        baseline = joblib.load(source / files[2]).predict(va[names])
        np.testing.assert_array_equal(baseline, va.paired_aft)
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
            tr[kept], tr.lower_bound_hours, tr.upper_bound_hours)
        joblib.dump(model, out / f'{year}_model.joblib')
        pred = model.predict(va[kept])
        np.testing.assert_array_equal(pred, joblib.load(out / f'{year}_model.joblib').predict(va[kept]))
        result = va.copy()
        result['without_so2'] = pred
        result.to_parquet(out / f'{year}_predictions.parquet', index=False)
        control, candidate = score_predictions(va, baseline), score_predictions(va, pred)
        checks = {k: candidate[k] < control[k] for k in ('exact_row_mae', 'exact_block_mae')}
        checks['tail24_block_mae'] = candidate['tail24_block_mae'] <= control['tail24_block_mae']
        scores[str(year)] = {'control': control, 'candidate': candidate, 'checks': checks, 'passed': all(checks.values())}
        (out / f'{year}_decision.json').write_text(json.dumps(scores[str(year)], indent=2), encoding='utf-8')
        print(json.dumps({str(year): scores[str(year)]}, indent=2), flush=True)
        if not all(checks.values()):
            break
    assert before == {str(p): sha(Path(p)) for p in before}
    report = {'input_sha256': before, 'scores': scores, 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
