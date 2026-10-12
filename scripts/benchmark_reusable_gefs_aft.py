"""Fixed forecast-snapshot addition to the interval/history AFT development model."""

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
    forecast_dir = base / 'reusable_gefs_snapshots_v1'
    out = base / 'reusable_gefs_aft_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [dev / 'protocol.json', dev / 'predictions.parquet', dev / 'expanded_predictions.parquet',
             base / 'interval_business_aft_v1/training_rows.parquet',
             base / 'trajectory_reset_audit_v1/origins.parquet', forecast_dir / 'features.parquet',
             forecast_dir / 'report.json', Path(__file__), ROOT / 'src/haze_forecast/observed_aft.py',
             ROOT / 'src/haze_forecast/training.py', ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    old_names = json.loads((dev / 'protocol.json').read_bytes())['features']
    forecast_report = json.loads((forecast_dir / 'report.json').read_bytes())
    assert sha(forecast_dir / 'features.parquet') == forecast_report['output_sha256']['features.parquet']
    extra = forecast_report['fields'] + ['forecast_known_age_hours', 'forecast_run_age_hours', 'forecast_remaining_lead_hours']
    names = old_names + extra
    forecasts = pd.read_parquet(forecast_dir / 'features.parquet').set_index('timestamp')
    history = pd.read_parquet(base / 'trajectory_reset_audit_v1/origins.parquet').set_index('timestamp')
    train = pd.read_parquet(base / 'interval_business_aft_v1/training_rows.parquet')
    train[old_names[87:]] = history.loc[train.timestamp, old_names[87:]].to_numpy()
    val = pd.read_parquet(dev / 'predictions.parquet')
    for frame in (train, val):
        frame[extra] = forecasts.loc[frame.timestamp, extra].to_numpy()
        meta = forecasts.loc[frame.timestamp]
        known = meta.known_at.notna()
        assert meta.loc[known, 'known_at'].le(meta.index[known]).all()
        assert meta.loc[known, 'valid_at'].gt(meta.index[known]).all()
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'seed': 42, 'rounds': 500,
                'training_rows': len(train), 'training_forecast_rows': int(train[extra[0]].notna().sum()),
                'selection': 'single fixed 9-variable forecast plus three ages; no tuning',
                'gate': 'row/block MAEs improve and tail not worse on original and expanded development cohorts',
                'limitations': ['2014-2021 have no forecast snapshots; all rows retained',
                                'archive availability proxy, not operational certification',
                                'reused 2023 development data']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    pred = model.predict(val[names])
    assert np.isfinite(pred).all()
    np.testing.assert_array_equal(joblib.load(out / 'model.joblib').predict(val[names]), pred)
    val['forecast_aft'] = pred
    expanded = pd.read_parquet(dev / 'expanded_predictions.parquet')
    np.testing.assert_array_equal(expanded.timestamp, val.timestamp)
    expanded['forecast_aft'] = pred
    results = {}
    for name, frame in [('original', val), ('expanded', expanded)]:
        frame = frame.loc[frame.score_allowed]
        a = metrics(frame, frame.forecast_aft.to_numpy())
        b = metrics(frame, frame.history_counts_aft.to_numpy())
        results[name] = {'candidate': a, 'control': b, 'passed': bool(
            a['row_mae'] < b['row_mae'] and a['block_mae'] < b['block_mae'] and
            a['tail_24h_block_mae'] <= b['tail_24h_block_mae'])}
    val.to_parquet(out / 'predictions.parquet', index=False)
    expanded.to_parquet(out / 'expanded_predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'cohorts': results, 'passed': all(x['passed'] for x in results.values()),
              'inputs_unchanged': True, 'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
