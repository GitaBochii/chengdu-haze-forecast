"""Inference only: extend frozen regional models to all 266 canonical 2025 origins."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary
from haze_forecast.regional_air_experiment import prepare_panel
from haze_forecast.regional_air_scoring import active_node_mask, aggregate_active_predictions
from scripts.benchmark_regional_air_models import build_model
from scripts.evaluate_regional_common_intervals import cap_labels, curve_recovery

ROOT = Path(__file__).resolve().parents[1]


def checked_positions(times, requested):
    requested = pd.DatetimeIndex(requested)
    if not times.is_unique or not times.is_monotonic_increasing or not requested.is_unique:
        raise ValueError('Duplicate or unordered source clock')
    positions = times.get_indexer(requested)
    if (positions < 23).any():
        raise ValueError('Missing origin or insufficient history')
    for pos in positions:
        if not np.all(np.diff(times[pos - 23:pos + 1].asi8) == 3600000000000):
            raise ValueError('Noncontiguous input history')
    return positions


def main():
    base = ROOT / 'artifacts'
    regional = base / 'regional_air_models'
    out = base / 'task01_recovery_head/regional_frozen_complete_2025_20260916_v2'
    if out.exists():
        raise FileExistsError(out)
    data = ROOT / 'data/processed/regional_air_models_full_history'
    folders = {name: regional / 'full_history_2014_2023_v1'
               for name in ('matched_aft', 'pm25_gnn', 'airformer')}
    folders['pm25_gnn_history'] = regional / 'history_structure_seed42_v1'
    reference = json.loads((folders['pm25_gnn'] / 'report.json').read_bytes())
    cohort_path = base / 'task01_recovery_head/unified_scorecard_crossyear_v1/2025_original_common.parquet'
    cohort = pd.read_parquet(cohort_path)
    old_common = base / 'task01_recovery_head/regional_common_intervals_2025_20260916'
    old_report = json.loads((old_common / 'report.json').read_bytes())
    inputs = dict(old_report['input_sha256'])
    for name, digest in reference['input_data_sha256'].items():
        inputs[str(data / name)] = digest
    for path, digest in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen dependency mismatch: {path}')
    paths = [Path(__file__).resolve(), old_common / 'report.json',
             ROOT / 'scripts/benchmark_regional_air_models.py']
    paths += list((ROOT / 'src/haze_forecast').glob('regional_*.py'))
    for name, folder in folders.items():
        paths.append(folder / ('matched_2014_2023_aft.joblib' if name == 'matched_aft'
                               else f'{name}_state.pt'))
    paths += list((ROOT / 'experiments/regional_air_models/vendor/PM25-GNN/model').glob('*.py'))
    paths += list((ROOT / 'experiments/regional_air_models/vendor/AirFormer/src').rglob('*.py'))
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in inputs and inputs[str(path)] != digest:
            raise ValueError('Source conflict')
        inputs[str(path)] = digest
    panel = prepare_panel(pd.read_parquet(data / 'station_hourly.parquet'),
                          pd.read_parquet(data / 'weather_proxy.parquet'),
                          pd.Timestamp('2024-01-01', tz='Asia/Shanghai'),
                          training_start=pd.Timestamp('2014-01-01', tz='Asia/Shanghai'),
                          ozone_column='o3_8h')
    for folder in set(folders.values()):
        report = json.loads((folder / 'report.json').read_bytes())
        if panel['nodes'] != report['nodes'] or panel['feature_names'] != report['feature_names']:
            raise ValueError('Frozen station or feature schema mismatch')
        np.testing.assert_array_equal(panel['scaler'].mean, report['scaler']['mean'])
        np.testing.assert_array_equal(panel['scaler'].scale, report['scaler']['scale'])
        np.testing.assert_array_equal(panel['coords'], report['coordinates_lon_lat'])
    positions = checked_positions(panel['timestamps'], cohort.timestamp)
    active = active_node_mask(panel['targets'], positions, lookback=24)
    if not active.any(axis=1).all():
        raise ValueError('Some origins have no observed active station; no silent filtering')
    old = pd.read_parquet(old_common / 'capped_predictions.parquet')
    replay_positions = checked_positions(panel['timestamps'], old.timestamp)
    replay_active = active_node_mask(panel['targets'], replay_positions, lookback=24)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    args = SimpleNamespace(seed=42, history=24, horizon=72)
    manifest = {'input_sha256': inputs, 'origins': len(cohort), 'replay_origins': len(old),
                'device': str(device), 'curve_replay_atol': 0.001, 'curve_replay_rtol': 1e-5,
                'policy': 'No fitting or checkpoint selection; replay old 42 before extending each model',
                'replay_batch_policy': 'Original full saved origin order and batch size 256; tolerances unchanged',
                'target': 'min(full recovery start time, 68 hours)', 'deployment_authorized': False}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    scored = cap_labels(cohort)
    results, replay = {}, {}

    def infer(model, indices, mask):
        parts = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(indices), 256):
                batch_indices = indices[start:start + 256]
                x = np.stack([panel['inputs'][i - 23:i + 1] for i in batch_indices])
                parts.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
        normalized = np.concatenate(parts)
        prediction = np.maximum(0, normalized * panel['scaler'].scale[[0, 2]]
                                + panel['scaler'].mean[[0, 2]])
        return aggregate_active_predictions(prediction, mask)

    for name, folder in folders.items():
        if name == 'matched_aft':
            active_history = active_node_mask(panel['targets'], np.arange(len(panel['timestamps'])),
                                              lookback=24).any(axis=2)
            mask = np.broadcast_to(~active_history[:, :, None], panel['inputs'].shape)
            values = pd.DataFrame(np.ma.median(np.ma.array(panel['inputs'], mask=mask),
                                               axis=1).filled(0), index=panel['timestamps'])
            features = pd.concat([values.add_prefix('now_'),
                                  values.rolling(24).mean().add_prefix('mean_')], axis=1)
            model = joblib.load(folder / 'matched_2014_2023_aft.joblib')
            old_prediction = np.minimum(model.predict(features.iloc[replay_positions]), 68.)
            np.testing.assert_array_equal(old_prediction, old[name])
            prediction = np.minimum(model.predict(features.iloc[positions]), 68.)
            replay[name] = {'exact_capped_replay': True}
        else:
            model = build_model(name, panel, args)
            model.load_state_dict(torch.load(folder / f'{name}_state.pt', map_location='cpu', weights_only=True))
            model.to(device)
            with np.load(folder / f'{name}_2025_curves.npz', allow_pickle=False) as saved:
                saved_times = pd.to_datetime(saved['origin_ns'], utc=True).tz_convert('Asia/Shanghai')
                full_replay_positions = checked_positions(panel['timestamps'], saved_times)
                full_replay_active = active_node_mask(panel['targets'], full_replay_positions, lookback=24)
                full_replay_curves = infer(model, full_replay_positions, full_replay_active)
                take = saved_times.get_indexer(old.timestamp)
                if (take < 0).any():
                    raise ValueError('Missing old origin')
                expected = saved['city_prediction']
                np.testing.assert_allclose(full_replay_curves, expected, atol=0.001, rtol=1e-5)
                max_difference = float(np.max(abs(full_replay_curves - expected)))
                replay_curves = full_replay_curves[take]
            old_prediction = np.array([curve_recovery(c[:, 1]) for c in replay_curves])
            np.testing.assert_array_equal(old_prediction, old[name])
            curves = infer(model, positions, active)
            prediction = np.array([curve_recovery(c[:, 1]) for c in curves], dtype=float)
            common_take = pd.DatetimeIndex(cohort.timestamp).get_indexer(old.timestamp)
            np.testing.assert_array_equal(prediction[common_take], old[name])
            np.savez_compressed(out / f'{name}_city_curves.npz', prediction=curves,
                                origin_ns=pd.DatetimeIndex(cohort.timestamp).asi8)
            replay[name] = {'exact_capped_replay': True, 'replayed_curve_origins': len(saved_times),
                            'max_city_curve_difference': max_difference}
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        scored[name] = prediction
        results[name] = score_predictions(scored, prediction)
        print(json.dumps({'model': name, 'replay': replay[name], 'scores': results[name]}), flush=True)
    paired = {name: paired_error_summary(scored, scored.matched_aft, scored[name])
              for name in folders if name != 'matched_aft'}
    for path, digest in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Input changed during inference: {path}')
    scored.to_parquet(out / 'capped_predictions.parquet', index=False)
    cohort.to_parquet(out / 'uncapped_labels.parquet', index=False)
    report = {'scores': results, 'paired': paired, 'replay': replay, 'rows': len(scored),
              'cohort_coverage': len(scored) / len(cohort), 'new_origins': len(scored) - len(old),
              'deployment_authorized': False, 'training_performed': False,
              'limitations': ['Historical research population, not untouched evaluation',
                              'Full coverage of 266 selected origins is not all-hour business coverage',
                              'Capped target cannot measure complete duration beyond 67 hours',
                              'Station activation and imputation follow frozen original rules',
                              'Retrospective weather, no real-time availability claim',
                              'Checkpoint hashes pinned now; old reports lack complete output manifests'],
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
