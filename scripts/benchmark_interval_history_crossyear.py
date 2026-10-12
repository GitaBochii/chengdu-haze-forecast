"""Frozen five-count history experiment on both later diagnostic years."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    dev = base / 'interval_history_counts_v1'
    out = base / 'interval_history_crossyear_v1'
    if out.exists():
        raise FileExistsError(out)
    assert json.loads((dev / 'report.json').read_bytes())['passed']
    names = json.loads((dev / 'protocol.json').read_bytes())['features']
    extra = names[87:]
    paths = [dev / 'report.json', dev / 'protocol.json', Path(__file__),
             base / 'trajectory_reset_audit_v1/origins.parquet',
             base / 'interval_business_labels_v1/origins.parquet',
             base / 'ghcnh_weather_repair_v1/repaired_features.parquet',
             ROOT / 'artifacts/strict_history_backtest/predictions.parquet',
             ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py',
             ROOT / 'src/haze_forecast/app.py']
    paths += [base / f'interval_crossyear_v1/{y}/{n}.parquet' for y in (2024, 2025) for n in ('training_rows', 'predictions')]
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    labels = pd.read_parquet(base / 'interval_business_labels_v1/origins.parquet').set_index('timestamp')
    history = pd.read_parquet(base / 'trajectory_reset_audit_v1/origins.parquet').set_index('timestamp')
    page = pd.read_parquet(ROOT / 'artifacts/strict_history_backtest/predictions.parquet')[['timestamp', 'strict_prediction_hours']]
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'seed': 42, 'rounds': 500,
                'selection': 'fixed development-approved five-count history; no later-year tuning',
                'limitations': ['reused diagnostic years, not pristine holdout',
                                'annual candidate versus event-updated webpage',
                                '2025 candidate weather repaired; webpage original weather']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    results = {}
    for year in (2024, 2025):
        train = pd.read_parquet(base / f'interval_crossyear_v1/{year}/training_rows.parquet')
        val = pd.read_parquet(base / f'interval_crossyear_v1/{year}/predictions.parquet')
        for frame in (train, val):
            frame[extra] = history.loc[frame.timestamp, extra].to_numpy()
        x = val[names].copy()
        if year == 2025:
            repair = pd.read_parquet(base / 'ghcnh_weather_repair_v1/repaired_features.parquet')
            assert len(repair) == len(x)
            x[repair.columns] = repair.to_numpy()
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(
            train[names], train.lower_bound_hours, train.upper_bound_hours)
        joblib.dump(model, out / f'{year}.joblib')
        pred = model.predict(x)
        assert np.isfinite(pred).all()
        np.testing.assert_array_equal(joblib.load(out / f'{year}.joblib').predict(x), pred)
        val['history_aft'] = pred
        expanded = val.copy()
        for col in ('exact_hours', 'lower_bound_hours', 'upper_bound_hours', 'informative'):
            expanded[col] = labels.loc[val.timestamp, col].to_numpy()
        results[str(year)] = {}
        for cohort, frame in [('original', val), ('expanded', expanded)]:
            results[str(year)][cohort] = {arm: metrics(frame, frame[arm].to_numpy())
                for arm in ('history_aft', 'interval_bounds', 'legacy_training')}
        common = expanded.merge(page, on='timestamp', validate='one_to_one')
        results[str(year)]['page_common_expanded'] = {arm: metrics(common, common[arm].to_numpy())
            for arm in ('history_aft', 'strict_prediction_hours')}
        ranges = {}
        for arm in ('history_aft', 'strict_prediction_hours'):
            lo, hi, p = common.lower_bound_hours, common.upper_bound_hours, common[arm]
            lower = pd.Series(np.maximum.reduce([lo-p, p-hi, np.zeros(len(p))]), index=common.index)
            upper = pd.Series(np.maximum(abs(p-lo), abs(p-hi)), index=common.index)
            ranges[arm] = {'row_mae': [float(lower.mean()), float(upper.mean())],
                          'block_mae': [float(e.groupby(common.continuity_block).mean().mean()) for e in (lower, upper)]}
        results[str(year)]['common_compatible_mae_ranges_not_confidence_intervals'] = ranges
        val.to_parquet(out / f'{year}_predictions.parquet', index=False)
        expanded.to_parquet(out / f'{year}_expanded.parquet', index=False)
        common.to_parquet(out / f'{year}_common.parquet', index=False)
        x.to_parquet(out / f'{year}_inference_features.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'years': results, 'inputs_unchanged': True,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
