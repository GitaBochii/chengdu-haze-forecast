"""Fixed same-run f006-to-f024 changes added to the existing forecast AFT."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from haze_forecast.operational_gefs_benchmark import publication_audited_features
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    dev = base / 'reusable_gefs_aft_v1'
    out = base / 'gefs_time_changes_v1'
    if out.exists():
        raise FileExistsError(out)
    sources = [ROOT / f'data/processed/operational_gefs_audited/operational_gefs_event_features_{y}.parquet' for y in (2022, 2023, 2024, 2025)]
    paths = sources + [dev / 'protocol.json', dev / 'predictions.parquet', dev / 'expanded_predictions.parquet',
        base / 'interval_business_aft_v1/training_rows.parquet', base / 'trajectory_reset_audit_v1/origins.parquet',
        base / 'reusable_gefs_snapshots_v1/features.parquet', ROOT / 'artifacts/operational_gefs_ensemble/report.json',
        Path(__file__), ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py',
        ROOT / 'src/haze_forecast/app.py']
    before = {str(p): sha(p) for p in paths}
    archive = json.loads((ROOT / 'artifacts/operational_gefs_ensemble/report.json').read_bytes())
    fields = ['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']
    near = [f'gefs_mean_{f}_f006' for f in fields]
    parts = []
    for path in sources:
        assert sha(path) == archive['audited_gefs_sha256'][path.name]
        raw = pd.read_parquet(path)
        audited = publication_audited_features(raw)
        part = audited[['timestamp', *near]].rename(columns={'timestamp': 'known_at'})
        part['known_at'] = pd.to_datetime(part.known_at, utc=True)
        part['near_valid_at'] = pd.to_datetime(raw.gefs_forecast_valid_utc_f006, utc=True)
        part['near_run_at'] = pd.to_datetime(raw.gefs_run_utc, utc=True)
        parts.append(part)
    near_rows = pd.concat(parts, ignore_index=True)
    assert near_rows.known_at.is_unique
    forecasts = pd.read_parquet(base / 'reusable_gefs_snapshots_v1/features.parquet')
    combined = forecasts.merge(near_rows, on='known_at', how='left', validate='many_to_one')
    present = combined.known_at.notna()
    assert combined.loc[present, 'run_at'].eq(combined.loc[present, 'near_run_at']).all()
    assert (combined.loc[present, 'valid_at'] - combined.loc[present, 'near_valid_at']).eq(pd.Timedelta(hours=18)).all()
    additions = []
    for field in fields:
        name = f'gefs_same_run_change_6_to_24_{field}'
        combined[name] = combined[f'gefs_mean_{field}_f024'] - combined[f'gefs_mean_{field}_f006']
        additions.append(name)
    combined['gefs_f006_signed_lead_hours'] = (combined.near_valid_at - combined.timestamp).dt.total_seconds() / 3600
    additions.append('gefs_f006_signed_lead_hours')
    combined = combined.set_index('timestamp')
    old_names = json.loads((dev / 'protocol.json').read_bytes())['features']
    names = old_names + additions
    train = pd.read_parquet(base / 'interval_business_aft_v1/training_rows.parquet')
    history = pd.read_parquet(base / 'trajectory_reset_audit_v1/origins.parquet').set_index('timestamp')
    train[names[87:92]] = history.loc[train.timestamp, names[87:92]].to_numpy()
    train[names[92:]] = combined.loc[train.timestamp, names[92:]].to_numpy()
    val = pd.read_parquet(dev / 'predictions.parquet')
    val[additions] = combined.loc[val.timestamp, additions].to_numpy()
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'additions': additions, 'seed': 42, 'rounds': 500,
        'selection': 'single same-run change representation; no tuning',
        'limitations': ['f006 may be past relative to origin; signed lead supplied; values remain forecasts not observations',
                        'archive publication proxy retained', 'reused 2023 development data']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    combined.reset_index().to_parquet(out / 'feature_evidence.parquet', index=False)
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    pred = model.predict(val[names])
    assert np.isfinite(pred).all()
    val['time_change_aft'] = pred
    expanded = pd.read_parquet(dev / 'expanded_predictions.parquet')
    np.testing.assert_array_equal(expanded.timestamp, val.timestamp)
    expanded['time_change_aft'] = pred
    scores = {}
    for cohort, frame in [('original', val), ('expanded', expanded)]:
        frame = frame.loc[frame.score_allowed]
        a, b = [metrics(frame, frame[col].to_numpy()) for col in ('time_change_aft', 'forecast_aft')]
        scores[cohort] = {'candidate': a, 'control': b, 'passed': bool(a['row_mae'] < b['row_mae'] and
            a['block_mae'] < b['block_mae'] and a['tail_24h_block_mae'] <= b['tail_24h_block_mae'])}
    val.to_parquet(out / 'predictions.parquet', index=False)
    expanded.to_parquet(out / 'expanded_predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'cohorts': scores, 'passed': all(s['passed'] for s in scores.values()), 'inputs_unchanged': True,
        'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
