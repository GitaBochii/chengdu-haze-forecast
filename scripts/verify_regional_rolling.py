"""Independently replay rolling checkpoints and recalculate capped score evidence."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from haze_forecast.regional_air_experiment import prepare_panel
from haze_forecast.regional_air_scoring import active_node_mask, aggregate_active_predictions
from scripts.benchmark_regional_air_models import build_model
from scripts.complete_regional_frozen_predictions import checked_positions

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT / 'data/processed/regional_air_models_full_history'
    torch.set_num_threads(2)
    results = {}
    for year in (2022, 2023):
        folder = ROOT / f'artifacts/task01_recovery_head/regional_rolling_{year}_20260916'
        destination = folder / 'independent_verification.json'
        if destination.exists():
            raise FileExistsError(destination)
        protocol = json.loads((folder / 'protocol.json').read_bytes())
        report = json.loads((folder / 'report.json').read_bytes())
        for path, expected in protocol['input_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Input hash mismatch: {path}')
        for name, expected in report['output_sha256'].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Output hash mismatch: {name}')
        cutoff = pd.Timestamp(f'{year - 1}-01-01', tz='Asia/Shanghai')
        evaluation_start = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year + 1}-01-01', tz='Asia/Shanghai')
        training = pd.read_parquet(folder / 'training_windows.parquet').timestamp
        selection = pd.read_parquet(folder / 'selection_windows.parquet').timestamp
        assert (training + pd.Timedelta(hours=72) < cutoff).all()
        assert selection.ge(cutoff).all() and (selection + pd.Timedelta(hours=72) < evaluation_start).all()
        trace = json.loads((folder / 'airformer_training.json').read_bytes())
        assert all(row['selection_year'] == year - 1 for row in trace)
        best = min(trace, key=lambda row: row['validation_loss'])
        assert best['epoch'] == report['best_epoch']
        assert best['validation_loss'] == report['training']['best_validation_loss']
        stations = pd.read_parquet(data / 'station_hourly.parquet')
        weather = pd.read_parquet(data / 'weather_proxy.parquet')
        panel = prepare_panel(stations.loc[stations.timestamp.lt(end)],
                              weather.loc[weather.timestamp.lt(end)], cutoff,
                              training_start=pd.Timestamp('2014-01-01', tz='Asia/Shanghai'),
                              ozone_column='o3_8h')
        assert panel['nodes'] == protocol['nodes']
        np.testing.assert_array_equal(panel['scaler'].mean, protocol['scaler']['mean'])
        np.testing.assert_array_equal(panel['scaler'].scale, protocol['scaler']['scale'])
        frame = pd.read_parquet(folder / 'predictions.parquet')
        uncapped = pd.read_parquet(folder / 'uncapped_labels.parquet')
        pd.testing.assert_series_equal(frame.timestamp, uncapped.timestamp)
        np.testing.assert_array_equal(frame.lower_bound_hours, np.minimum(uncapped.lower_bound_hours, 68))
        np.testing.assert_array_equal(frame.upper_bound_hours, np.minimum(uncapped.upper_bound_hours, 68))
        np.testing.assert_array_equal(frame.score_mature, uncapped.label_available_at.lt(end))
        positions = checked_positions(panel['timestamps'], frame.timestamp)
        active = active_node_mask(panel['targets'], positions, lookback=24)
        model = build_model('airformer', panel, SimpleNamespace(**protocol['configuration']))
        model.load_state_dict(torch.load(folder / 'airformer_state.pt', weights_only=True, map_location='cpu'))
        model.to('cuda').eval()
        pieces = []
        with torch.inference_mode():
            for offset in range(0, len(positions), 256):
                x = np.stack([panel['inputs'][p - 23:p + 1] for p in positions[offset:offset + 256]])
                pieces.append(model(torch.from_numpy(x).to('cuda')).cpu().numpy())
        curves = np.maximum(0, np.concatenate(pieces) * panel['scaler'].scale[[0, 2]]
                            + panel['scaler'].mean[[0, 2]])
        curves = aggregate_active_predictions(curves, active)
        with np.load(folder / 'city_curves.npz') as saved:
            np.testing.assert_allclose(curves, saved['prediction'], rtol=1e-5, atol=.001)
            max_difference = float(abs(curves - saved['prediction']).max())
        # Independent rolling-window reconstruction, without the production decoder.
        windows = np.lib.stride_tricks.sliding_window_view(curves[:, :, 1], 6, axis=1)
        clean = (windows <= 100).all(axis=2)
        recovery = np.where(clean.any(axis=1), clean.argmax(axis=1) + 1, 68)
        np.testing.assert_array_equal(recovery, frame.airformer)
        scored = frame.loc[frame.score_mature].copy()
        exact = scored.loc[scored.exact_hours.notna()]
        for name in ('matched_aft', 'airformer'):
            error = abs(exact[name] - exact.exact_hours)
            tail = exact.exact_hours.ge(24)
            calculated = [error.mean(), error.groupby(exact.continuity_block).mean().mean(),
                          error[tail].groupby(exact.continuity_block[tail]).mean().mean()]
            np.testing.assert_allclose(calculated, [report['scores'][name][k] for k in (
                'exact_row_mae', 'exact_block_mae', 'tail24_block_mae')], rtol=0, atol=1e-12)
        lo, hi = scored.lower_bound_hours.to_numpy(), scored.upper_bound_hours.to_numpy()
        a, b = scored.matched_aft.to_numpy(), scored.airformer.to_numpy()
        truth = np.stack([lo, hi, np.clip(a, lo, hi), np.clip(b, lo, hi)])
        delta = abs(b - truth) - abs(a - truth)
        low, high = delta.min(axis=0), delta.max(axis=0)
        np.testing.assert_allclose([low.mean(), high.mean(),
            pd.Series(low).groupby(scored.continuity_block.to_numpy()).mean().mean(),
            pd.Series(high).groupby(scored.continuity_block.to_numpy()).mean().mean()],
            [report['paired'][k] for k in ('all_row_delta_lower', 'all_row_delta_upper',
                                         'all_block_delta_lower', 'all_block_delta_upper')], rtol=0, atol=1e-12)
        result = {'passed': True, 'replayed_origins': len(frame), 'scored_origins': len(scored),
                  'max_curve_difference': max_difference, 'independent_metrics': 10,
                  'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
        results[str(year)] = result
        del model, panel, stations, weather
        torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
