"""Complete recovery scores for frozen early transition models with paired AFT."""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from haze_forecast.run_length_recovery import recovery_start_distribution
from haze_forecast.run_length_training import recovery_median
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'transition_earlier_years_v1'
    out = base / 'transition_earlier_recovery_v1'
    if out.exists():
        raise FileExistsError(out)
    prior = json.loads((source / 'protocol.json').read_bytes())
    saved = json.loads((source / 'report.json').read_bytes())
    for p, digest in prior['input_sha256'].items():
        assert sha(Path(p)) == digest
    for name, digest in saved['output_sha256'].items():
        assert sha(source / name) == digest
    names = prior['features']
    path = base / 'gefs_temporal_calibration_data_v1/training_rows.parquet'
    grid_path = base / 'observed_business_dataset_v1/features.parquet'
    paths = [path, grid_path, Path(__file__), source / 'report.json', source / 'protocol.json',
              ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/run_length_recovery.py',
              ROOT / 'src/haze_forecast/run_length_training.py', ROOT / 'src/haze_forecast/cohort_scorecard.py',
              ROOT / 'src/haze_forecast/app.py']
    paths += [source / f'{year}_model.txt' for year in (2020, 2021)]
    paths += [p for p in (ROOT / 'artifacts/cnemc_2025_deployment_candidate').iterdir() if p.is_file()]
    before = {str(p): sha(p) for p in paths}
    rows = pd.read_parquet(path)
    grid = pd.read_parquet(grid_path)
    ranges = grid.groupby('continuity_block').timestamp.agg(['min', 'max'])
    out.mkdir()
    protocol = {'input_sha256': before, 'features': names, 'years': [2020, 2021],
                'aft_rounds': 500, 'seed': 42,
                'horizon': 'Each annual training maximum upper bound plus five confirmation hours',
                'unresolved': 'No dropped rows, no cap; no full MAE if any unresolved',
                'purpose': 'Close recovery-score gap; no tuning, blending, calibration or promotion',
                'deployment_authorized': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    scores = {}
    for year in protocol['years']:
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year + 1}-01-01', tz='Asia/Shanghai')
        cross = ranges.index[(ranges['min'] < cut) & (ranges['max'] >= cut)]
        last = ranges.index[(ranges['min'] < end) & (ranges['max'] >= end)]
        tr = rows.loc[(rows.timestamp < cut) & (rows.label_available_at < cut)
                      & ~rows.continuity_block.isin(cross)].copy()
        va = rows.loc[(rows.timestamp >= cut) & (rows.timestamp < end) & (rows.label_available_at < end)
                      & ~rows.continuity_block.isin(cross.union(last))].copy()
        assert len(tr) == saved['years'][str(year)]['training_origins']
        assert len(va) == saved['years'][str(year)]['validation_origins']
        assert not set(tr.continuity_block) & set(va.continuity_block)
        horizon = int(tr.upper_bound_hours.max()) + 5
        aft = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
            tr[names], tr.lower_bound_hours, tr.upper_bound_hours)
        joblib.dump(aft, out / f'{year}_aft.joblib')
        ap = aft.predict(va[names])
        np.testing.assert_array_equal(ap, joblib.load(out / f'{year}_aft.joblib').predict(va[names]))
        model = lgb.Booster(model_str=(source / f'{year}_model.txt').read_text(encoding='utf-8'))
        distributions = []
        for start in range(0, len(va), 32):
            x = va[names].iloc[start:start + 32].to_numpy(dtype=float)
            augmented = np.column_stack((np.repeat(x, horizon * 6, axis=0),
                                         np.tile(np.repeat(np.arange(1, horizon + 1), 6), len(x)),
                                         np.tile(np.arange(6), len(x) * horizon)))
            q = model.predict(augmented, num_threads=1).reshape(len(x), horizon, 6)
            distributions.append(recovery_start_distribution(q))
        distribution = np.concatenate(distributions)
        tp = recovery_median(distribution)
        np.save(out / f'{year}_distribution.npy', distribution)
        tr.to_parquet(out / f'{year}_training.parquet', index=False)
        va['paired_aft'] = ap
        va['transition'] = tp
        va.to_parquet(out / f'{year}_predictions.parquet', index=False)
        control = score_predictions(va, ap)
        candidate = score_predictions(va, tp) if np.isfinite(tp).all() else None
        scores[str(year)] = {'horizon': horizon, 'unresolved_rows': int(np.isnan(tp).sum()),
                            'control': control, 'candidate': candidate}
        print(json.dumps({str(year): scores[str(year)]}, indent=2), flush=True)
    assert before == {str(p): sha(Path(p)) for p in before}
    report = {'scores': scores, 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
