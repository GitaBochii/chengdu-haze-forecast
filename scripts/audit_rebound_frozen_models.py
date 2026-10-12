"""Outcome-stratified replay of existing AFT and hazard models, no selection."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    aft = base / 'gefs_temporal_calibration_data_v1'
    hazard = base / 'interval_hazard_temporal_v1'
    out = base / 'rebound_frozen_model_audit_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = []
    for source in (aft, hazard):
        manifest = json.loads((source / 'report.json').read_bytes())['output_sha256']
        for name, digest in manifest.items():
            assert sha(source / name) == digest
            paths.append(source / name)
        paths.append(source / 'report.json')
    grid_path = base / 'observed_business_dataset_v1/features.parquet'
    paths += [grid_path, Path(__file__), ROOT / 'src/haze_forecast/app.py',
              ROOT / 'src/haze_forecast/cohort_scorecard.py']
    paths += [p for p in (ROOT / 'artifacts/cnemc_2025_deployment_candidate').iterdir() if p.is_file()]
    before = {str(p): sha(p) for p in paths}
    names = json.loads((aft / 'protocol.json').read_bytes())['features']
    assert names == json.loads((hazard / 'protocol.json').read_bytes())['features']
    rows = pd.read_parquet(aft / 'predictions.parquet')
    old = pd.read_parquet(hazard / 'predictions.parquet')
    pd.testing.assert_frame_equal(rows, old[rows.columns])
    assert len(rows) == 411 and rows.score_allowed.all() and rows.exact_hours.notna().all()
    a = joblib.load(aft / 'model.joblib').predict(rows[names])
    h = joblib.load(hazard / 'model.joblib').predict(rows[names])
    np.testing.assert_array_equal(a, rows.forecast_aft)
    np.testing.assert_array_equal(h, old.interval_hazard)
    assert np.isfinite(h).all()
    grid = pd.read_parquet(grid_path).set_index('timestamp')
    delays, runs = [], []
    for row in rows.itertuples():
        recovery = row.timestamp + pd.Timedelta(hours=row.exact_hours)
        future = grid.loc[row.timestamp + pd.Timedelta(hours=1):recovery + pd.Timedelta(hours=5), 'aqi']
        assert len(future) == int(row.exact_hours) + 5
        if future.isna().any():
            delays.append(np.nan)
            runs.append(np.nan)
            continue
        clean = future.le(100).to_numpy()
        first = int(np.flatnonzero(clean)[0]) + 1
        streak, completed, stable_start = 0, 0, None
        for i, value in enumerate(clean):
            if value:
                streak += 1
                if streak == 6:
                    stable_start = i - 4
                    break
            else:
                completed += int(streak > 0)
                streak = 0
        assert stable_start == row.exact_hours
        delays.append(row.exact_hours - first)
        runs.append(completed)
    result = rows[['timestamp', 'continuity_block', 'lower_bound_hours', 'upper_bound_hours', 'exact_hours']].copy()
    result['aft'] = a
    result['hazard'] = h
    result['first_clean_to_stable_delay'] = delays
    result['short_clean_runs_before_recovery'] = runs
    result['rebound'] = result.first_clean_to_stable_delay.gt(0)
    known = result.first_clean_to_stable_delay.notna()
    groups = {'all': np.ones(len(result), dtype=bool), 'rebound': result.rebound,
              'no_rebound': known & ~result.rebound, 'unknown_missing_aqi': ~known,
              'previously_selected_block_1232': result.continuity_block.eq(1232)}
    scores = {}
    for label, mask in groups.items():
        cohort = result.loc[mask]
        if cohort.empty:
            continue
        scores[label] = {model: score_predictions(cohort, cohort[model]) for model in ('aft', 'hazard')}
    out.mkdir()
    result.to_parquet(out / 'origins.parquet', index=False)
    report = {'input_sha256': before, 'scores': scores,
              'limitations': ['Rebound grouping reads future outcomes; diagnostic only, never an inference gate',
                              'Any missing future AQI makes rebound status unknown; all-model scores retain these rows',
                              'The same process may contribute origins to both groups',
                              '2022 reused development; no new fit, tuning, or later-year scores',
                              'Existing 2023 hazard failure is not cancelled by this audit'],
              'output_sha256': {'origins.parquet': sha(out / 'origins.parquet')},
              'deployment_authorized': False}
    assert before == {str(p): sha(Path(p)) for p in before}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for group, models in scores.items():
        print(group, {m: {k: v for k, v in score.items() if k in
                          ('rows', 'blocks', 'exact_row_mae', 'exact_block_mae', 'tail24_block_mae')}
                      for m, score in models.items()})


if __name__ == '__main__':
    main()
