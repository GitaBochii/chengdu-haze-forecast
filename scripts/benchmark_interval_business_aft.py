"""Audit sharp recovery bounds independently and run fixed 2023 development arms."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import DATA, ROOT, metrics, sha


def main():
    started = time.monotonic()
    labels = ROOT / 'artifacts/task01_recovery_head/interval_business_labels_v1'
    control = ROOT / 'artifacts/task01_recovery_head/observed_business_aft_v1'
    out = ROOT / 'artifacts/task01_recovery_head/interval_business_aft_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [DATA / n for n in ('features.parquet', 'origins.parquet', 'report.json')]
    paths += [labels / 'origins.parquet', labels / 'report.json', control / 'predictions.parquet',
              control / 'report.json', Path(__file__), ROOT / 'src/haze_forecast/observed_aft.py',
              ROOT / 'src/haze_forecast/training.py', ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    label_report = json.loads((labels / 'report.json').read_bytes())
    assert sha(labels / 'origins.parquet') == label_report['output_sha256']
    names = json.loads((DATA / 'report.json').read_bytes())['feature_names']
    full = pd.read_parquet(DATA / 'features.parquet')
    old = pd.read_parquet(DATA / 'origins.parquet')
    new = pd.read_parquet(labels / 'origins.parquet')
    pd.testing.assert_frame_equal(old[['timestamp', 'continuity_block'] + names],
                                  new[['timestamp', 'continuity_block'] + names])
    assert full.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    aqi = full.aqi.to_numpy()
    positions = pd.Index(full.timestamp).get_indexer(new.timestamp)
    assert (positions >= 0).all()
    # Direct forward windows, independent of production convolution/search implementation.
    for row, pos in zip(new.itertuples(), positions):
        lower = upper = None
        for start in range(pos + 1, len(aqi) + 1):
            window = aqi[start:start + 6]
            if lower is None and not np.any(window > 100):
                lower = start - pos
            if len(window) == 6 and np.all(np.isfinite(window) & (window <= 100)):
                upper = start - pos
                break
        assert lower == row.lower_bound_hours
        assert (np.inf if upper is None else upper) == row.upper_bound_hours
        available = full.timestamp.iloc[-1] if upper is None else row.timestamp + pd.Timedelta(hours=upper + 5)
        assert available == row.label_available_at
    exact = old.exact_hours.notna()
    np.testing.assert_array_equal(old.loc[exact, 'exact_hours'], new.loc[exact, 'exact_hours'])
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    cross = set(full.loc[full.timestamp.lt(cutoff), 'continuity_block']) & set(
        full.loc[full.timestamp.ge(cutoff), 'continuity_block'])
    train = new.loc[new.timestamp.lt(cutoff) & new.label_available_at.lt(cutoff)
                    & new.informative & ~new.continuity_block.isin(cross)].copy()
    validation = pd.read_parquet(control / 'predictions.parquet')
    pd.testing.assert_frame_equal(validation[names].reset_index(drop=True),
                                  old.loc[old.timestamp.dt.year.eq(2023), names].reset_index(drop=True))
    assert not set(train.continuity_block) & set(validation.continuity_block)
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'seed': 42, 'rounds': 500,
                'cutoff': str(cutoff), 'crossing_blocks': sorted(cross),
                'scoring': 'unchanged original exact 2023 score_allowed cohort',
                'arms': ['interval_exact_only', 'interval_bounds'],
                'selection': 'fixed development diagnostic, no parameter search',
                'limitations': ['reused development year, not untouched holdout',
                                'no complete-population MAE from exact subset',
                                'finite bounds do not identify probabilities inside interval']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    train.to_parquet(out / 'training_rows.parquet', index=False)
    scores = {}
    allowed = validation.score_allowed
    for arm in ('legacy_training', 'exact_only', 'observed_bounds'):
        scores[arm] = metrics(validation.loc[allowed], validation.loc[allowed, arm])
    for arm in protocol['arms']:
        selected = train if arm == 'interval_bounds' else train.loc[train.exact_hours.notna()]
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(
            selected[names], selected.lower_bound_hours, selected.upper_bound_hours)
        path = out / f'{arm}.joblib'
        joblib.dump(model, path)
        pred = model.predict(validation[names])
        assert np.isfinite(pred).all()
        np.testing.assert_array_equal(joblib.load(path).predict(validation[names]), pred)
        validation[arm] = pred
        scores[arm] = metrics(validation.loc[allowed], pred[allowed])
        scores[arm]['training_rows'] = len(selected)
        baseline = scores['legacy_training']
        scores[arm]['passes_gate'] = bool(
            scores[arm]['row_mae'] < baseline['row_mae'] and
            scores[arm]['block_mae'] < baseline['block_mae'] and
            scores[arm]['tail_24h_block_mae'] <= baseline['tail_24h_block_mae'])
    validation.to_parquet(out / 'predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'status': 'completed_development_diagnostic', 'scores': scores,
              'independently_verified_origins': len(new), 'old_exact_preserved': int(exact.sum()),
              'train_rows': len(train), 'train_intervals': int(train.reason.eq('interval').sum()),
              'inputs_unchanged': True, 'elapsed_seconds': time.monotonic() - started,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
