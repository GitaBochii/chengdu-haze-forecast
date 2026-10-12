"""Train both predeclared rolling folds with identical train/inference support."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary
from haze_forecast.regional_air_experiment import prepare_panel
from haze_forecast.regional_air_scoring import active_node_mask, aggregate_active_predictions
from scripts.benchmark_regional_air_models import build_model, train_model
from scripts.complete_regional_frozen_predictions import checked_positions
from scripts.diagnose_regional_curves import digest
from scripts.evaluate_regional_common_intervals import curve_recovery
from scripts.masked_airformer import MaskedAirFormer

ROOT = Path(__file__).resolve().parents[1]


def infer(model, panel, positions, active):
    pieces = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(positions), 256):
            x = np.stack([panel['inputs'][p-23:p+1] for p in positions[offset:offset+256]])
            pieces.append(model(torch.from_numpy(x).to('cuda')).cpu().numpy())
    values = np.maximum(0, np.concatenate(pieces) * panel['scaler'].scale[[0, 2]] + panel['scaler'].mean[[0, 2]])
    return aggregate_active_predictions(values, active)


def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required by fixed protocol')
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'masked_spatial_training_20260916'
    if out.exists():
        raise FileExistsError(out)
    files = [Path(__file__).resolve(), ROOT / 'scripts/masked_airformer.py', ROOT / 'masked_spatial_training_plan_20260916.md']
    hashes = {str(p): digest(p) for p in files}
    source = ROOT / 'data/processed/regional_air_models_full_history'
    stations = pd.read_parquet(source / 'station_hourly.parquet')
    weather = pd.read_parquet(source / 'weather_proxy.parquet')
    protocols = {}
    for year in (2022, 2023):
        folder = base / f'regional_rolling_{year}_20260916'
        protocol = json.loads((folder / 'protocol.json').read_bytes())
        frozen = json.loads((folder / 'report.json').read_bytes())
        for path, expected in protocol['input_sha256'].items():
            if digest(Path(path)) != expected:
                raise ValueError(f'Changed input: {path}')
            hashes[path] = expected
        for name, expected in frozen['output_sha256'].items():
            if digest(folder / name) != expected:
                raise ValueError(f'Changed output: {name}')
            hashes[str(folder / name)] = expected
        protocols[year] = protocol
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({'folds': protocols, 'input_sha256': hashes,
        'intervention': 'Origin-known AQI source support in both training and inference',
        'new_trainable_parameters': 0, 'deployment_authorized': False}, indent=2), encoding='utf-8')
    results = {}
    for year in (2022, 2023):
        folder = base / f'regional_rolling_{year}_20260916'
        target = out / str(year)
        target.mkdir()
        protocol = protocols[year]
        end = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
        cutoff = pd.Timestamp(f'{year-1}-01-01', tz='Asia/Shanghai')
        panel = prepare_panel(stations.loc[stations.timestamp.lt(end)], weather.loc[weather.timestamp.lt(end)], cutoff,
                              training_start=pd.Timestamp('2014-01-01', tz='Asia/Shanghai'), ozone_column='o3_8h')
        assert panel['nodes'] == protocol['nodes']
        np.testing.assert_array_equal(panel['scaler'].mean, protocol['scaler']['mean'])
        np.testing.assert_array_equal(panel['scaler'].scale, protocol['scaler']['scale'])
        training_times = pd.read_parquet(folder / 'training_windows.parquet').timestamp
        selection_times = pd.read_parquet(folder / 'selection_windows.parquet').timestamp
        train_positions = checked_positions(panel['timestamps'], training_times)
        selection_positions = checked_positions(panel['timestamps'], selection_times)
        assert (training_times + pd.Timedelta(hours=72) < cutoff).all()
        assert (selection_times + pd.Timedelta(hours=72) < pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')).all()
        args = SimpleNamespace(**protocol['configuration'])
        numerical_model = build_model('airformer', panel, args)
        n = sum(p.numel() for p in numerical_model.parameters())
        model = MaskedAirFormer(numerical_model)
        assert n == sum(p.numel() for p in model.parameters())
        flags = np.isfinite(panel['targets'][:, :, 1:2]).astype('float32')
        panel['inputs'] = np.concatenate([panel['inputs'], flags], axis=-1)
        training = train_model(model, panel, train_positions, selection_positions, args, target, 'masked_airformer', torch.device('cuda'))
        trace_path = target / 'masked_airformer_training.json'
        trace = json.loads(trace_path.read_bytes())
        for row in trace:
            row['validation_loss'] = row.pop('validation_2024_loss')
            row['selection_year'] = year - 1
        trace_path.write_text(json.dumps(trace, indent=2), encoding='utf-8')
        frame = pd.read_parquet(folder / 'predictions.parquet')
        positions = checked_positions(panel['timestamps'], frame.timestamp)
        active = active_node_mask(panel['targets'], positions)
        curves = infer(model, panel, positions, active)
        # Rebuild the original numerical dimension before loading the wrapper checkpoint.
        numerical_panel = {**panel, 'inputs': panel['inputs'][..., :17]}
        restored = MaskedAirFormer(build_model('airformer', numerical_panel, args)).to('cuda')
        restored.load_state_dict(torch.load(target / 'masked_airformer_state.pt', weights_only=True, map_location='cuda'))
        replay = infer(restored, panel, positions, active)
        np.testing.assert_array_equal(curves, replay)
        frame['masked_airformer'] = [curve_recovery(c[:, 1]) for c in curves]
        scored = frame.loc[frame.score_mature]
        result = {'training': training, 'best_epoch': min(trace, key=lambda r: r['validation_loss'])['epoch'],
                  'predicted_origins': len(frame), 'scored_origins': len(scored),
                  'checkpoint_replay_max_difference': float(abs(curves-replay).max()),
                  'scores': score_predictions(scored, scored.masked_airformer.to_numpy()),
                  'versus_original': paired_error_summary(scored, scored.airformer, scored.masked_airformer),
                  'versus_aft': paired_error_summary(scored, scored.matched_aft, scored.masked_airformer)}
        for path, expected in hashes.items():
            if digest(Path(path)) != expected:
                raise ValueError(f'Changed during training: {path}')
        frame.to_parquet(target / 'predictions.parquet', index=False)
        np.savez_compressed(target / 'city_curves.npz', prediction=curves, origin_ns=pd.DatetimeIndex(frame.timestamp).asi8)
        result['output_sha256'] = {p.name: digest(p) for p in target.iterdir()}
        (target / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        results[str(year)] = result
        print(json.dumps({'year': year, 'result': result}), flush=True)
        del model, restored, panel, numerical_panel, numerical_model
        torch.cuda.empty_cache()
    (out / 'report.json').write_text(json.dumps({'years': results, 'input_sha256': hashes,
        'deployed': False, 'target': 'min(T,68h)'}, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
