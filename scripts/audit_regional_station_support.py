"""Diagnose observation support without changing frozen predictions or labels."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.regional_air_scoring import active_node_mask
from scripts.diagnose_regional_curves import curve_statistics, digest

ROOT = Path(__file__).resolve().parents[1]


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else None


def fixed_pool_truth(raw, active):
    raw, active = np.asarray(raw, dtype=float), np.asarray(active, dtype=bool)
    if raw.ndim != 3 or active.shape != (len(raw), raw.shape[2]):
        raise ValueError('Expected [origin,lead,node] and [origin,node]')
    available = np.isfinite(raw) & active[:, None, :]
    count = available.sum(axis=2)
    median = np.ma.median(np.ma.array(raw, mask=~available), axis=2).filled(np.nan)
    required = active.sum(axis=1)[:, None]
    median[(count != required) | (required == 0)] = np.nan
    return median, count


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'regional_station_support_20260916_v2'
    if out.exists():
        raise FileExistsError(out)
    source = ROOT / 'data/processed/regional_air_models_full_history'
    paths = [source / 'station_hourly.parquet', source / 'city_hourly.parquet',
             Path(__file__).resolve(), ROOT / 'scripts/diagnose_regional_curves.py',
             ROOT / 'src/haze_forecast/regional_air_scoring.py']
    hashes = {str(path): digest(path) for path in paths}
    stations = pd.read_parquet(paths[0])
    city = pd.read_parquet(paths[1]).set_index('timestamp').aqi.sort_index()
    table = stations.pivot(index='timestamp', columns='node_id', values='aqi').sort_index()
    np.testing.assert_allclose(table.median(axis=1), city.reindex(table.index), equal_nan=True)
    reports, frames = {}, {}
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
                raise ValueError(f'Changed frozen output: {name}')
            hashes[str(path)] = expected
        hashes[str(folder / 'report.json')] = digest(folder / 'report.json')
        frame = pd.read_parquet(folder / 'predictions.parquet')
        with np.load(folder / 'city_curves.npz') as saved:
            np.testing.assert_array_equal(saved['origin_ns'], pd.DatetimeIndex(frame.timestamp).asi8)
            prediction = saved['prediction'][:, :, 1].copy()
        # Reindex to a complete hourly clock; never forward-fill observed targets.
        times = pd.date_range(table.index.min(), table.index.max(), freq='h')
        selected = table.reindex(index=times, columns=protocol['nodes'])
        positions = times.get_indexer(frame.timestamp)
        if (positions < 23).any() or (positions + 72 >= len(times)).any():
            raise ValueError('Origin outside history/future archive')
        active = active_node_mask(selected.to_numpy()[:, :, None], positions)[:, :, 0]
        future_positions = positions[:, None] + np.arange(1, 73)[None, :]
        raw = selected.to_numpy()[future_positions]
        fixed, count = fixed_pool_truth(raw, active)
        city_truth = city.reindex(times).to_numpy()[future_positions]
        city_stats = curve_statistics(prediction, city_truth)
        fixed_stats = curve_statistics(prediction, fixed)
        np.testing.assert_array_equal(city_stats['predicted_recovery'], frame.airformer)
        valid = np.isfinite(fixed) & np.isfinite(city_truth)
        delta = np.where(valid, abs(fixed - city_truth), 0)
        n = valid.sum(axis=1)
        detail = frame[['timestamp', 'continuity_block', 'score_mature', 'exact_hours', 'airformer']].copy()
        detail['active_nodes'] = active.sum(axis=1)
        detail['fully_observed_pool_hours'] = np.isfinite(fixed).sum(axis=1)
        detail['comparable_city_hours'] = n
        detail['support_aqi_mae'] = np.divide(delta.sum(axis=1), n, out=np.full(len(frame), np.nan), where=n > 0)
        detail['support_threshold_flips'] = (valid & ((fixed <= 100) != (city_truth <= 100))).sum(axis=1)
        for name, truth in (('city', city_truth), ('fixed', fixed)):
            error = np.where(valid, abs(prediction - truth), 0).sum(axis=1)
            detail[f'{name}_paired_hour_mae'] = np.divide(error, n, out=np.full(len(frame), np.nan), where=n > 0)
        for name, stats in (('city', city_stats), ('fixed', fixed_stats)):
            for key in ('aqi_mae', 'truth_recovery', 'false_predicted_window'):
                detail[f'{name}_{key}'] = stats[key]
        mature = detail.loc[detail.score_mature]
        windows = mature.loc[mature.city_false_predicted_window.notna() & mature.fixed_false_predicted_window.notna()]
        complete = mature.loc[mature.city_truth_recovery.notna() & mature.fixed_truth_recovery.notna()]
        bad_city = windows.city_false_predicted_window.eq(1)
        good_fixed = windows.fixed_false_predicted_window.eq(0)
        reports[str(year)] = {
            'predicted_origins': len(frame), 'mature_origins': len(mature),
            'training_selected_nodes': protocol['nodes'], 'archive_nodes': len(table.columns),
            'active_pool_sizes': {str(k): int(v) for k, v in mature.active_nodes.value_counts().items()},
            'fully_observed_pool_hour_cells': int(mature.fully_observed_pool_hours.sum()),
            'comparable_city_hour_cells': int(mature.comparable_city_hours.sum()),
            'support_aqi_mae_origin_mean': finite_mean(mature.support_aqi_mae),
            'support_aqi_mae_block_mean': finite_mean(mature.groupby('continuity_block').support_aqi_mae.mean()),
            'threshold_flip_cells': int(mature.support_threshold_flips.sum()),
            'both_complete_72h_origins': len(complete),
            'both_complete_recovery_differences': int(complete.city_truth_recovery.ne(complete.fixed_truth_recovery).sum()),
            'both_complete_city_prediction_aqi_mae': finite_mean(complete.city_aqi_mae),
            'both_complete_fixed_prediction_aqi_mae': finite_mean(complete.fixed_aqi_mae),
            'paired_hour_city_prediction_aqi_mae': finite_mean(mature.city_paired_hour_mae),
            'paired_hour_fixed_prediction_aqi_mae': finite_mean(mature.fixed_paired_hour_mae),
            'both_verifiable_predicted_windows': len(windows),
            'city_invalid_windows_in_paired_subset': int(bad_city.sum()),
            'city_invalid_but_fixed_valid': int((bad_city & good_fixed).sum()),
            'city_valid_but_fixed_invalid': int((~bad_city & ~good_fixed).sum()),
            'both_invalid_windows': int((bad_city & ~good_fixed).sum()),
            'predicted_windows_not_paired': int(mature.airformer.lt(68).sum() - len(windows)),
        }
        frames[year] = detail
        # Save observation-only support arrays for independent replay of this audit.
        frames[f'arrays_{year}'] = {'fixed_truth': fixed, 'city_truth': city_truth,
                                  'active': active, 'available_count': count}
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Changed during audit: {path}')
    out.mkdir()
    for year in (2022, 2023):
        frames[year].to_parquet(out / f'origins_{year}.parquet', index=False)
        np.savez_compressed(out / f'observation_support_{year}.npz', **frames[f'arrays_{year}'])
    report = {'years': reports, 'input_sha256': hashes,
              'policy': 'Observation-only diagnostic; no future-selected prediction mask; no target replacement',
              'limitations': ['Fixed-pool truth requires every selected station observed; incomplete hours remain unknown',
                              'Paired subsets are not the complete business scoring cohort',
                              'Repeated forecast origins and hours are not independent events'],
              'output_sha256': {p.name: digest(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
