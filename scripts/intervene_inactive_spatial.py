"""Frozen-checkpoint experiment removing origin-known inactive spatial sources."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary
from haze_forecast.regional_air_experiment import prepare_panel
from haze_forecast.regional_air_scoring import active_node_mask, aggregate_active_predictions
from scripts.benchmark_regional_air_models import build_model
from scripts.complete_regional_frozen_predictions import checked_positions
from scripts.diagnose_regional_curves import digest
from scripts.evaluate_regional_common_intervals import curve_recovery

ROOT = Path(__file__).resolve().parents[1]


def restricted_assignment(original, active):
    active = torch.as_tensor(active, dtype=torch.bool, device=original.device)
    if active.shape != (original.shape[1],) or not active.any():
        raise ValueError('Expected nonempty source support')
    if active.all():
        return original.clone(), original.sum(dim=1).eq(0)
    kept = original * active[None, :, None]
    total = kept.sum(dim=1, keepdim=True)
    return kept / total.clamp_min(torch.finfo(kept.dtype).tiny), total.squeeze(1).eq(0)


@contextmanager
def spatial_support(model, active):
    saved = []
    try:
        for module in model.modules():
            assignment = getattr(module, 'assignment', None)
            if isinstance(assignment, torch.Tensor) and assignment.ndim == 3:
                saved.append((module, assignment, module.mask))
                module.assignment, module.mask = restricted_assignment(assignment, active)
        if not saved:
            raise ValueError('No spatial buffers found')
        yield
    finally:
        for module, assignment, mask in saved:
            module.assignment, module.mask = assignment, mask


def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Original CUDA inference device required')
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'inactive_spatial_intervention_20260916'
    if out.exists():
        raise FileExistsError(out)
    hashes = {str(p): digest(p) for p in (Path(__file__).resolve(), ROOT / 'inactive_spatial_plan_20260916.md')}
    source = ROOT / 'data/processed/regional_air_models_full_history'
    stations = pd.read_parquet(source / 'station_hourly.parquet')
    weather = pd.read_parquet(source / 'weather_proxy.parquet')
    results, predictions, curves = {}, {}, {}
    for year in (2022, 2023):
        folder = base / f'regional_rolling_{year}_20260916'
        protocol = json.loads((folder / 'protocol.json').read_bytes())
        frozen = json.loads((folder / 'report.json').read_bytes())
        for path, expected in protocol['input_sha256'].items():
            if digest(Path(path)) != expected:
                raise ValueError(f'Changed input: {path}')
            hashes[path] = expected
        for name, expected in frozen['output_sha256'].items():
            path = folder / name
            if digest(path) != expected:
                raise ValueError(f'Changed output: {path}')
            hashes[str(path)] = expected
        frame = pd.read_parquet(folder / 'predictions.parquet')
        end = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
        panel = prepare_panel(stations.loc[stations.timestamp.lt(end)], weather.loc[weather.timestamp.lt(end)],
                              pd.Timestamp(f'{year-1}-01-01', tz='Asia/Shanghai'),
                              training_start=pd.Timestamp('2014-01-01', tz='Asia/Shanghai'), ozone_column='o3_8h')
        assert panel['nodes'] == protocol['nodes']
        np.testing.assert_array_equal(panel['scaler'].mean, protocol['scaler']['mean'])
        np.testing.assert_array_equal(panel['scaler'].scale, protocol['scaler']['scale'])
        positions = checked_positions(panel['timestamps'], frame.timestamp)
        active = active_node_mask(panel['targets'], positions)
        model = build_model('airformer', panel, SimpleNamespace(**protocol['configuration']))
        model.load_state_dict(torch.load(folder / 'airformer_state.pt', weights_only=True, map_location='cpu'))
        model.to('cuda').eval()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        baseline, candidate, node_changes = [], [], []
        with torch.inference_mode():
            for offset in range(0, len(positions), 256):
                batch = positions[offset:offset+256]
                x = torch.from_numpy(np.stack([panel['inputs'][p-23:p+1] for p in batch])).to('cuda')
                original = model(x).cpu().numpy()
                changed = original.copy()
                support = active[offset:offset+len(batch), :, 1]
                for mask in np.unique(support, axis=0):
                    if mask.all():
                        continue
                    rows = (support == mask).all(axis=1)
                    with spatial_support(model, mask):
                        trial = model(x).cpu().numpy()
                    changed[rows] = trial[rows]
                baseline.append(original)
                candidate.append(changed)
                diff = abs(changed[:, :, :, 1] - original[:, :, :, 1]) * panel['scaler'].scale[2]
                node_changes.extend(np.where(support[:, None, :], diff, 0).max(axis=(1, 2)).tolist())
        for key, value in model.state_dict().items():
            assert torch.equal(state[key], value), key
        original = aggregate_active_predictions(np.maximum(0, np.concatenate(baseline) * panel['scaler'].scale[[0, 2]]
                                                           + panel['scaler'].mean[[0, 2]]), active)
        altered = aggregate_active_predictions(np.maximum(0, np.concatenate(candidate) * panel['scaler'].scale[[0, 2]]
                                                          + panel['scaler'].mean[[0, 2]]), active)
        with np.load(folder / 'city_curves.npz') as saved:
            np.testing.assert_array_equal(saved['origin_ns'], pd.DatetimeIndex(frame.timestamp).asi8)
            np.testing.assert_allclose(original, saved['prediction'], rtol=1e-5, atol=.001)
            replay_error = float(abs(original - saved['prediction']).max())
        np.testing.assert_array_equal([curve_recovery(c[:, 1]) for c in original], frame.airformer)
        frame['intervention'] = [curve_recovery(c[:, 1]) for c in altered]
        frame['inactive_nodes'] = (~active[:, :, 1]).sum(axis=1)
        frame['active_node_aqi_max_change'] = node_changes
        frame['city_aqi_max_change'] = abs(altered[:, :, 1] - original[:, :, 1]).max(axis=1)
        np.testing.assert_array_equal(altered[frame.inactive_nodes.eq(0)], original[frame.inactive_nodes.eq(0)])
        cohort = frame.loc[frame.score_mature]
        results[str(year)] = {'predicted_origins': len(frame), 'scored_origins': len(cohort),
                              'baseline_replay_max_difference': replay_error,
                              'intervened_scored_origins': int(cohort.inactive_nodes.gt(0).sum()),
                              'active_node_aqi_max_change': float(cohort.active_node_aqi_max_change.max()),
                              'changed_recovery_origins': int(cohort.intervention.ne(cohort.airformer).sum()),
                              'scores': score_predictions(cohort, cohort.intervention.to_numpy()),
                              'versus_original': paired_error_summary(cohort, cohort.airformer, cohort.intervention),
                              'versus_aft': paired_error_summary(cohort, cohort.matched_aft, cohort.intervention)}
        predictions[year] = frame
        curves[year] = altered
        print(json.dumps({str(year): results[str(year)]}), flush=True)
        del model, state, panel
        torch.cuda.empty_cache()
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Changed during experiment: {path}')
    out.mkdir()
    for year in (2022, 2023):
        predictions[year].to_parquet(out / f'predictions_{year}.parquet', index=False)
        np.savez_compressed(out / f'curves_{year}.npz', prediction=curves[year],
                            origin_ns=pd.DatetimeIndex(predictions[year].timestamp).asi8)
    report = {'years': results, 'input_sha256': hashes, 'trained': False, 'deployed': False,
              'limitations': ['Inference-time intervention differs from mask-aware retraining',
                              'Historical reused years; no independent fresh holdout', 'Capped min(T,68h) target'],
              'output_sha256': {p.name: digest(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
