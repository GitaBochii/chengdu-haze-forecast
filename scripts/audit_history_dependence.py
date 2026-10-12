"""Fixed-model dependence probe, not a forecast candidate or retraining."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_temporal_calibration_data_v1'
    out = base / 'history_dependence_2022_v1'
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((source / 'report.json').read_bytes())['output_sha256']
    for name, digest in manifest.items():
        if sha(source / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    paths = [source / name for name in manifest]
    paths += [source / 'report.json', Path(__file__), ROOT / 'src/haze_forecast/app.py',
              ROOT / 'src/haze_forecast/cohort_scorecard.py']
    paths += [p for p in (ROOT / 'artifacts/cnemc_2025_deployment_candidate').iterdir()
              if p.is_file()]
    before = {str(p): sha(p) for p in paths}
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    pollutants = ('aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3')
    history = [f'{p}_{suffix}' for p in pollutants
               for suffix in ('lag_1', 'lag_3', 'lag_24', 'roll_3', 'roll_24')]
    history += ['hours_since_first_observed_high', 'observed_high_hours',
                'past_missing_hours', 'past_short_clean_hours', 'past_high_nonparticle_hours']
    assert len(history) == 40 and set(history).issubset(names)
    untouched = [name for name in names if name not in history]
    train = pd.read_parquet(source / 'training_rows.parquet')
    val = pd.read_parquet(source / 'predictions.parquet')
    cutoff = pd.Timestamp('2022-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()
    assert val.timestamp.dt.year.eq(2022).all() and val.score_allowed.all()
    assert val.exact_hours.notna().all() and val.timestamp.is_unique
    model = joblib.load(source / 'model.joblib')
    baseline = model.predict(val[names])
    np.testing.assert_array_equal(baseline, val.forecast_aft)
    # Joint donor rows preserve dependence within the substituted feature group.
    donors = np.random.default_rng(42).integers(0, len(train), size=(20, len(val)))
    protocol = {'input_sha256': before, 'history_columns': history, 'untouched_columns': untouched,
                'seed': 42, 'repetitions': 20, 'donors': 'Uniform pre-2022 training rows, with replacement',
                'purpose': 'Measure prediction dependence, not altered-input accuracy',
                'limitations': ['Donor history may conflict with retained current state and weather',
                                'Magnitude is not causal importance or an accuracy improvement',
                                '2022 reused development; no new independent test',
                                'Labels used only after perturbations to describe baseline errors'],
                'deployment_authorized': False}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    np.save(out / 'donor_indices.npy', donors)
    predictions = []
    for indices in donors:
        changed = val[names].copy()
        changed[history] = train.iloc[indices][history].to_numpy()
        pd.testing.assert_frame_equal(changed[untouched], val[untouched])
        predictions.append(model.predict(changed))
    predictions = np.asarray(predictions)
    assert np.isfinite(predictions).all() and (predictions >= 0).all()
    np.save(out / 'perturbed_predictions.npy', predictions)
    delta = predictions - baseline
    rows = val[['timestamp', 'continuity_block', 'exact_hours']].copy()
    rows['baseline_prediction'] = baseline
    rows['baseline_error'] = baseline - val.exact_hours.to_numpy()
    rows['baseline_absolute_error'] = abs(rows.baseline_error)
    rows['mean_absolute_shift'] = abs(delta).mean(axis=0)
    rows['mean_signed_shift'] = delta.mean(axis=0)
    rows['max_absolute_shift'] = abs(delta).max(axis=0)
    rows.to_parquet(out / 'origins.parquet', index=False)
    blocks = rows.groupby('continuity_block').agg(
        rows=('timestamp', 'size'), first_origin=('timestamp', 'min'),
        baseline_mae=('baseline_absolute_error', 'mean'),
        baseline_bias=('baseline_error', 'mean'),
        mean_absolute_shift=('mean_absolute_shift', 'mean'),
        mean_signed_shift=('mean_signed_shift', 'mean'))
    blocks.to_csv(out / 'blocks.csv')
    assert before == {str(p): sha(Path(p)) for p in before}
    report = {'baseline_scores': score_predictions(val, baseline),
              'mean_absolute_shift_hours': float(abs(delta).mean()),
              'block_mean_absolute_shift_hours': float(blocks.mean_absolute_shift.mean()),
              'row_mean_shift_quantiles': {str(q): float(rows.mean_absolute_shift.quantile(q))
                                           for q in (0, .25, .5, .75, 1)},
              'unchanged_origins_all_repetitions': int((abs(delta).max(axis=0) <= 1e-9).sum()),
              'top_baseline_error_blocks': blocks.nlargest(3, 'baseline_mae').reset_index().drop(
                  columns='first_origin').to_dict('records'),
              'output_sha256': {p.name: sha(p) for p in out.iterdir()},
              'deployment_authorized': False}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
