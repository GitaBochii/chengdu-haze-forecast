"""Fixed cross-year confirmation, without historical PBL or added 2021 forecasts."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.benchmark_gefs_72h_aft import attach_forecast, verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha
from scripts.build_gefs_72h_features import require


def require_development(report):
    require(report.get('passed') is True, 'development gate did not pass')


def candidate_features(old, frozen):
    kept = [name for name in old if not name.startswith('boundary_layer_height_m')]
    require(len(old) == 104 and len(set(old)) == 104 and len(kept) == 91,
            'unexpected original feature schema')
    require(kept == frozen and 'gefs_control_boundary_layer_height_m_f024' in kept,
            'frozen features changed or forecast PBL removed')
    return kept


def validate_split(train, val, year):
    cutoff = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp(f'{year + 1}-01-01', tz='Asia/Shanghai')
    require(train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all(),
            'training or label maturity crosses cutoff')
    require(val.timestamp.ge(cutoff).all() and val.timestamp.lt(end).all(),
            'inference outside target year')
    require(not set(train.continuity_block) & set(val.continuity_block), 'block leakage')


def crossyear_gate(years):
    checks = {}
    for year in ('2024', '2025'):
        scores = years[year]['common']
        candidate, old, page = (scores[a] for a in
                                ('no_history_pbl', 'forecast_aft', 'strict_prediction_hours'))
        keys = ('row_mae', 'block_mae', 'tail_24h_block_mae')
        checks[year] = bool(all(np.isfinite(v[k]) for v in (candidate, old, page) for k in keys)
                            and all(candidate[k] < page[k] for k in keys)
                            and candidate['tail_24h_block_mae'] <= old['tail_24h_block_mae'])
    return {'passed': all(checks.values()), 'year_checks': checks, 'deployment_authorized': False}


def verified_folder(folder, inputs):
    report = json.loads((folder / 'report.json').read_bytes())
    outputs = report['output_sha256']
    require('protocol.json' in outputs, 'protocol hash missing')
    require(all(Path(n).name == n for n in outputs), 'invalid output filename')
    inputs.update(verify_hashes({str(folder / n): h for n, h in outputs.items()}))
    inputs[str(folder / 'report.json')] = sha(folder / 'report.json')
    protocol = json.loads((folder / 'protocol.json').read_bytes())
    inputs.update(verify_hashes(protocol['input_sha256']))
    return report, protocol


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'original_gefs_without_pbl_crossyear_v1'
    require(not out.exists(), f'output already exists: {out}')
    inputs = {str(Path(__file__).resolve()): sha(Path(__file__))}
    dev_report, dev = verified_folder(base / 'original_gefs_without_pbl_v1', inputs)
    require_development(dev_report)
    old_dir = base / 'reusable_gefs_crossyear_v1'
    _, old = verified_folder(old_dir, inputs)
    names = old['features']
    kept = candidate_features(names, dev['features'])
    require(dev['rounds'] == 500 and dev['params']['colsample_bytree'] == 1.0,
            'frozen development settings changed')
    for source in ('src/haze_forecast/app.py', 'scripts/benchmark_gefs_72h_aft.py',
                   'scripts/benchmark_observed_business_aft.py'):
        inputs[str(ROOT / source)] = sha(ROOT / source)
    forecasts = pd.read_parquet(base / 'reusable_gefs_snapshots_v1/features.parquet')
    history = pd.read_parquet(base / 'trajectory_reset_audit_v1/origins.parquet')
    prepared = {}
    for year in (2024, 2025):
        train = pd.read_parquet(base / f'interval_crossyear_v1/{year}/training_rows.parquet')
        train = attach_forecast(train, history, names[87:92])
        train = attach_forecast(train, forecasts, names[92:])
        cohorts = {c: pd.read_parquet(old_dir / f'{year}_{c}.parquet')
                   for c in ('predictions', 'expanded', 'common')}
        val = cohorts['predictions']
        validate_split(train, val, year)
        x = pd.read_parquet(old_dir / f'{year}_features.parquet')
        # Reconstruct only to verify the saved matrix; use that saved matrix for inference.
        reference = pd.read_parquet(base / f'interval_history_crossyear_v1/{year}_inference_features.parquet')
        reference[names[92:]] = forecasts.set_index('timestamp').loc[val.timestamp, names[92:]].to_numpy()
        pd.testing.assert_frame_equal(x[names], reference[names])
        baseline = joblib.load(old_dir / f'{year}.joblib').predict(x[names])
        np.testing.assert_array_equal(baseline, val.forecast_aft.to_numpy())
        require(not val.timestamp.duplicated().any(), 'duplicate inference timestamps')
        baseline_by_time = pd.Series(baseline, index=val.timestamp)
        for frame in cohorts.values():
            require(not frame.timestamp.duplicated().any(), 'duplicate cohort timestamps')
            np.testing.assert_array_equal(baseline_by_time.loc[frame.timestamp], frame.forecast_aft)
        prepared[year] = (train, x, val, cohorts)
    verify_hashes(inputs)
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({
        'input_sha256': inputs, 'features': kept, 'params': dev['params'], 'rounds': 500,
        'selection': 'none; frozen development candidate and original forecast support',
        'gate': 'Both years common expanded cohort: all three MAEs strictly below webpage; tail no worse than old f024.',
        'deployment_authorized': False,
        'limitations': ['Reused historical confirmation, not pristine holdout.',
                        'Archive availability proxy only; unchanged webpage training policy differs.'],
    }, indent=2), encoding='utf-8')
    years = {}
    for year, (train, x, val, cohorts) in prepared.items():
        train.to_parquet(out / f'{year}_training_rows.parquet', index=False)
        x[kept].to_parquet(out / f'{year}_inference_features.parquet', index=False)
        matrix = xgb.DMatrix(train[kept].astype(float))
        matrix.set_float_info('label_lower_bound', train.lower_bound_hours.to_numpy())
        matrix.set_float_info('label_upper_bound', train.upper_bound_hours.to_numpy())
        model = xgb.train(dev['params'], matrix, num_boost_round=500)
        path = out / f'{year}.ubj'
        model.save_model(path)
        inference = xgb.DMatrix(x[kept].astype(float))
        prediction = model.predict(inference)
        require(np.isfinite(prediction).all(), 'nonfinite candidate predictions')
        replay = xgb.Booster()
        replay.load_model(path)
        np.testing.assert_array_equal(prediction, replay.predict(inference))
        by_time = pd.Series(prediction, index=val.timestamp)
        scores = {}
        for cohort, frame in cohorts.items():
            frame['no_history_pbl'] = by_time.loc[frame.timestamp].to_numpy()
            arms = ['no_history_pbl', 'forecast_aft', 'history_aft']
            if cohort == 'common':
                arms += ['strict_prediction_hours']
            scores[cohort] = {arm: metrics(frame, frame[arm].to_numpy()) for arm in arms}
            frame.to_parquet(out / f'{year}_{cohort}.parquet', index=False)
        years[str(year)] = scores
    verify_hashes(inputs)
    report = {'years': years, **crossyear_gate(years), 'inputs_unchanged': True,
              'old_model_replays_passed': True, 'candidate_replays_passed': True,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
