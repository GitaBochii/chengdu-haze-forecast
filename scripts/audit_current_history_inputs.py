"""Read-only model/history audit; no fitting or feature intervention."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.features import MEASUREMENT_COLUMNS, TRAJECTORY_FEATURE_NAMES
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_temporal_calibration_data_v1'
    out = base / 'current_history_input_audit_v1'
    if out.exists():
        raise FileExistsError(out)
    grid_path = base / 'observed_business_dataset_v1/features.parquet'
    paths = [source / name for name in
             ('protocol.json', 'report.json', 'model.joblib',
              'training_rows.parquet', 'predictions.parquet')]
    paths += [grid_path, Path(__file__), ROOT / 'src/haze_forecast/features.py',
              ROOT / 'src/haze_forecast/app.py']
    paths += [p for p in (ROOT / 'artifacts/cnemc_2025_deployment_candidate').iterdir()
              if p.is_file()]
    before = {str(p): sha(p) for p in paths}
    manifest = json.loads((source / 'report.json').read_bytes())['output_sha256']
    for name, digest in manifest.items():
        if sha(source / name) != digest:
            raise ValueError(f'Frozen output changed: {name}')
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    model = joblib.load(source / 'model.joblib')
    assert model.booster.feature_names == names
    grid = pd.read_parquet(grid_path)
    grid = grid.loc[grid.timestamp.lt(pd.Timestamp('2023-01-01', tz='Asia/Shanghai'))]
    assert grid.timestamp.is_unique and grid.timestamp.is_monotonic_increasing
    assert grid.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    values = {}
    for name in MEASUREMENT_COLUMNS:
        series = grid[name].astype(float)
        for lag in (1, 3, 24):
            values[f'{name}_lag_{lag}'] = series.shift(lag)
        for window, minimum in ((3, 1), (24, 6)):
            values[f'{name}_roll_{window}'] = series.shift(1).rolling(window, min_periods=minimum).mean()
    high = grid.aqi.gt(100)
    block = grid.continuity_block
    first = grid.timestamp.where(high).groupby(block).transform('min')
    # Future first highs in pre-onset rows must not activate past history.
    started = first.notna() & grid.timestamp.ge(first)
    values['hours_since_first_observed_high'] = (grid.timestamp - first).dt.total_seconds() / 3600 + 1
    conditions = {'observed_high_hours': high, 'past_missing_hours': grid.aqi.isna(),
                  'past_short_clean_hours': grid.aqi.le(100),
                  'past_high_nonparticle_hours': high & ~grid.particle_primary.fillna(False).astype(bool)}
    for name, condition in conditions.items():
        values[name] = (condition & started).astype(int).groupby(block).cumsum()
    rebuilt = pd.DataFrame(values)
    rebuilt.index = pd.DatetimeIndex(grid.timestamp)
    assert set(rebuilt.columns).issubset(names)
    counts = {}
    for filename in ('training_rows.parquet', 'predictions.parquet'):
        frame = pd.read_parquet(source / filename)
        np.testing.assert_allclose(frame[rebuilt.columns].to_numpy(dtype=float),
                                   rebuilt.loc[frame.timestamp].to_numpy(dtype=float),
                                   rtol=1e-12, atol=1e-12, equal_nan=True)
        counts[filename] = len(frame)
        if filename == 'predictions.parquet':
            np.testing.assert_array_equal(model.predict(frame[names]), frame.forecast_aft)
    splits = model.booster.get_score(importance_type='weight')
    history = list(rebuilt.columns)
    report = {'input_sha256': before, 'rows_checked': counts,
              'rebuilt_history_columns': history,
              'history_columns_used_in_splits': {k: splits[k] for k in history if k in splits},
              'history_columns_without_splits': [k for k in history if k not in splits],
              'excluded_trajectory_columns': [k for k in TRAJECTORY_FEATURE_NAMES if k not in names],
              'feature_count': len(names), 'prediction_replay_exact': True,
              'limitations': ['Split usage does not prove predictive value or causal importance',
                              'No new model, later-year evaluation, or deployment',
                              'Grouped first-high reconstruction checked eligible origins only; not a production feature API']}
    assert before == {str(p): sha(Path(p)) for p in before}
    out.mkdir()
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
