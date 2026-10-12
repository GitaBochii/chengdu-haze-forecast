"""Two predeclared historical rolling folds for AirFormer and matched AFT."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.labels import label_city_data
from haze_forecast.paired_interval_score import paired_error_summary
from haze_forecast.regional_air_experiment import (
    boundary_event_mask,
    prepare_panel,
    purge_event_windows,
    sequence_origins,
)
from haze_forecast.regional_air_scoring import active_node_mask, aggregate_active_predictions
from haze_forecast.training import AFTSurvivalRegressor
from scripts.benchmark_regional_air_models import build_model, train_model
from scripts.complete_regional_frozen_predictions import checked_positions
from scripts.evaluate_regional_common_intervals import cap_labels, curve_recovery

ROOT = Path(__file__).resolve().parents[1]


def fold_years(prediction_year):
    if prediction_year not in (2022, 2023):
        raise ValueError('Only the predeclared 2022 and 2023 folds are allowed')
    return 2014, prediction_year - 2, prediction_year - 1, prediction_year


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-year', type=int, required=True)
    cli = parser.parse_args()
    first, last, selection, evaluation = fold_years(cli.prediction_year)
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / f'regional_rolling_{evaluation}_20260916'
    if out.exists():
        raise FileExistsError(out)
    data = ROOT / 'data/processed/regional_air_models_full_history'
    label_path = base / 'interval_business_labels_v1/origins.parquet'
    files = [data / name for name in ('station_hourly.parquet', 'weather_proxy.parquet', 'city_hourly.parquet')]
    files += [label_path, Path(__file__).resolve(), ROOT / 'scripts/benchmark_regional_air_models.py',
              ROOT / 'scripts/complete_regional_frozen_predictions.py',
              ROOT / 'scripts/evaluate_regional_common_intervals.py',
              ROOT / 'src/haze_forecast/training.py', ROOT / 'src/haze_forecast/labels.py',
              ROOT / 'src/haze_forecast/cohort_scorecard.py', ROOT / 'src/haze_forecast/paired_interval_score.py']
    files += list((ROOT / 'src/haze_forecast').glob('regional_*.py'))
    files += list((ROOT / 'experiments/regional_air_models/vendor/AirFormer/src').rglob('*.py'))
    files += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if hashes[str(label_path)] != json.loads((label_path.parent / 'report.json').read_bytes())['output_sha256']:
        raise ValueError('Canonical labels changed')
    start = pd.Timestamp(f'{first}-01-01', tz='Asia/Shanghai')
    cutoff = pd.Timestamp(f'{selection}-01-01', tz='Asia/Shanghai')
    eval_start = pd.Timestamp(f'{evaluation}-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp(f'{evaluation + 1}-01-01', tz='Asia/Shanghai')
    stations = pd.read_parquet(data / 'station_hourly.parquet')
    weather = pd.read_parquet(data / 'weather_proxy.parquet')
    city = pd.read_parquet(data / 'city_hourly.parquet')
    stations = stations.loc[stations.timestamp.lt(end)]
    weather = weather.loc[weather.timestamp.lt(end)]
    city = city.loc[city.timestamp.lt(end)]
    panel = prepare_panel(stations, weather, cutoff, training_start=start, ozone_column='o3_8h')
    times = panel['timestamps']
    labels = label_city_data(city).set_index('timestamp').reindex(times)
    blocked = boundary_event_mask(labels, [start, cutoff, eval_start])
    known_weather = weather.set_index('timestamp').reindex(times)[[
        'temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']].notna().all(axis=1).to_numpy()

    def origins(year):
        selected = sequence_origins(times, history=24, horizon=72,
                                    start=pd.Timestamp(f'{year}-01-01', tz=times.tz),
                                    end=pd.Timestamp(f'{year + 1}-01-01', tz=times.tz), stride=6)
        selected = purge_event_windows(selected, blocked, history=24, horizon=72)
        selected = selected[known_weather[selected]]
        active = active_node_mask(panel['targets'], selected, lookback=24)
        return selected[active.any(axis=1).all(axis=1)]

    yearly = {year: origins(year) for year in range(first, last + 1)}
    if any(len(rows) == 0 for rows in yearly.values()):
        raise ValueError('Empty training year')
    train_indices = np.concatenate(list(yearly.values()))
    validation_indices = origins(selection)
    if not len(validation_indices) or not (times[train_indices + 72] < cutoff).all():
        raise ValueError('Training target crosses cutoff')
    if not (times[validation_indices + 72] < eval_start).all():
        raise ValueError('Checkpoint selection reads prediction year')
    canonical = pd.read_parquet(label_path)
    cohort = canonical.loc[canonical.timestamp.ge(eval_start) & canonical.timestamp.lt(end)].copy()
    cohort = cohort.reset_index(drop=True)
    positions = checked_positions(times, cohort.timestamp)
    active = active_node_mask(panel['targets'], positions, lookback=24)
    if not active.any(axis=1).all():
        raise ValueError('No active station at a requested origin; do not silently filter')
    active_history = active_node_mask(panel['targets'], np.arange(len(times)), lookback=24).any(axis=2)
    mask = np.broadcast_to(~active_history[:, :, None], panel['inputs'].shape)
    values = pd.DataFrame(np.ma.median(np.ma.array(panel['inputs'], mask=mask), axis=1).filled(0), index=times)
    aft_features = pd.concat([values.add_prefix('now_'), values.rolling(24).mean().add_prefix('mean_')], axis=1)
    remaining = labels.remaining_hours
    mature = times + pd.to_timedelta(remaining + 5, unit='h')
    aft_train = np.asarray((times >= start) & (times < cutoff) & (mature < cutoff)
                           & remaining.notna().to_numpy(), dtype=bool)
    aft_train &= ~blocked & aft_features.notna().all(axis=1).to_numpy()
    history_safe = np.zeros(len(times), dtype=bool)
    history_safe[purge_event_windows(np.arange(23, len(times)), blocked, history=24, horizon=0)] = True
    aft_train &= history_safe
    missing = np.cumsum(~np.isfinite(city.set_index('timestamp').reindex(times).aqi.to_numpy()))
    for pos in np.flatnonzero(aft_train):
        finish = pos + int(remaining.iloc[pos]) + 5
        aft_train[pos] = finish < len(times) and missing[finish] == missing[pos]
    if aft_train.sum() < 30:
        raise ValueError('Insufficient matched AFT training origins')
    args = SimpleNamespace(history=24, horizon=72, batch_size=256, learning_rate=.001,
                           seed=42, epochs=24, patience=5)
    if not torch.cuda.is_available():
        raise RuntimeError('Declared CUDA training unavailable; do not change training device silently')
    device = torch.device('cuda')
    torch.set_num_threads(2)
    protocol = {'input_sha256': hashes, 'training_years': [first, last], 'selection_year': selection,
                'prediction_year': evaluation, 'configuration': vars(args),
                'training_windows_by_year': {str(y): len(i) for y, i in yearly.items()},
                'validation_windows': len(validation_indices), 'prediction_origins': len(cohort),
                'training_last_target': str(times[train_indices + 72].max()),
                'selection_last_target': str(times[validation_indices + 72].max()),
                'matched_aft_training_rows': int(aft_train.sum()),
                'nodes': panel['nodes'], 'feature_names': panel['feature_names'],
                'scaler': {'mean': panel['scaler'].mean.tolist(), 'scale': panel['scaler'].scale.tolist()},
                'policy': 'Two fixed folds; no routing fit, no 2025 score access or parameter selection',
                'target': 'min(T,68h); concentration loss selects neural checkpoint in prior year',
                'deployment_authorized': False}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    cohort[['timestamp']].to_parquet(out / 'requested_origins.parquet', index=False)
    pd.DataFrame({'timestamp': times[train_indices]}).to_parquet(out / 'training_windows.parquet', index=False)
    pd.DataFrame({'timestamp': times[validation_indices]}).to_parquet(out / 'selection_windows.parquet', index=False)
    model = build_model('airformer', panel, args)
    training = train_model(model, panel, train_indices, validation_indices, args, out, 'airformer', device)
    # The reused trainer's legacy log key says 2024; normalize the saved trace to
    # this fold's explicit selection year. Actual selected indices were checked above.
    trace_path = out / 'airformer_training.json'
    trace = json.loads(trace_path.read_bytes())
    for row in trace:
        row['validation_loss'] = row.pop('validation_2024_loss')
        row['selection_year'] = selection
    trace_path.write_text(json.dumps(trace, indent=2), encoding='utf-8')
    best = min(trace, key=lambda row: row['validation_loss'])
    if best['validation_loss'] != training['best_validation_loss']:
        raise ValueError('Wrong checkpoint selection')
    model.load_state_dict(torch.load(out / 'airformer_state.pt', map_location=device, weights_only=True))
    model.eval()
    parts = []
    with torch.inference_mode():
        for offset in range(0, len(positions), 256):
            x = np.stack([panel['inputs'][pos - 23:pos + 1] for pos in positions[offset:offset + 256]])
            parts.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
    prediction = np.maximum(0, np.concatenate(parts) * panel['scaler'].scale[[0, 2]]
                            + panel['scaler'].mean[[0, 2]])
    curves = aggregate_active_predictions(prediction, active)
    neural_prediction = np.array([curve_recovery(curve[:, 1]) for curve in curves], dtype=float)
    aft = AFTSurvivalRegressor(nthread=1).fit(aft_features.loc[aft_train], remaining.loc[aft_train])
    joblib.dump(aft, out / 'matched_aft.joblib')
    aft_prediction = aft.predict(aft_features.iloc[positions])
    np.testing.assert_array_equal(aft_prediction, joblib.load(out / 'matched_aft.joblib').predict(aft_features.iloc[positions]))
    scored = cap_labels(cohort)
    scored['matched_aft'] = np.minimum(aft_prediction, 68.)
    scored['airformer'] = neural_prediction
    scored['score_mature'] = cohort.label_available_at.lt(end)
    mature_cohort = scored.loc[scored.score_mature]
    scores = {name: score_predictions(mature_cohort, mature_cohort[name].to_numpy())
              for name in ('matched_aft', 'airformer')}
    paired = paired_error_summary(mature_cohort, mature_cohort.matched_aft, mature_cohort.airformer)
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Input changed during training: {path}')
    np.savez_compressed(out / 'city_curves.npz', prediction=curves,
                        origin_ns=pd.DatetimeIndex(cohort.timestamp).asi8)
    scored.to_parquet(out / 'predictions.parquet', index=False)
    cohort.to_parquet(out / 'uncapped_labels.parquet', index=False)
    report = {'training': training, 'best_epoch': best['epoch'], 'scores': scores, 'paired': paired,
              'predicted_origins': len(cohort), 'scored_mature_origins': len(mature_cohort),
              'immature_rows_retained_not_scored': int((~scored.score_mature).sum()),
              'routing_fit': False, 'deployment_authorized': False,
              'limitations': ['Historical research years, not fresh holdouts',
                              'Capped duration, not full recovery time',
                              'Fold-specific training-only station selection may differ',
                              'Retrospective weather, no live availability proof',
                              'Prior-year concentration validation selects neural checkpoint',
                              'No mixer fit or new weights selected from prediction-year labels'],
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'year': evaluation, 'scores': scores, 'paired': paired}), flush=True)


if __name__ == '__main__':
    main()
