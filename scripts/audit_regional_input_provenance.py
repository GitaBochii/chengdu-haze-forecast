"""Trace existing input fills; do not train or change missing-data policy."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.paired_interval_score import paired_error_summary
from haze_forecast.regional_air_experiment import prepare_panel
from haze_forecast.regional_air_scoring import active_node_mask
from scripts.diagnose_regional_curves import digest

ROOT = Path(__file__).resolve().parents[1]


def fill_provenance(values):
    values = np.asarray(values, dtype=float)
    shape = values.shape
    flat = values.reshape(len(values), -1)
    if np.isinf(flat).any():
        raise ValueError('Infinite raw input')
    available = pd.DataFrame(flat).ffill(limit=6).notna().to_numpy()
    return np.where(np.isfinite(flat), 0, np.where(available, 1, 2)).astype('uint8').reshape(shape)


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'regional_input_provenance_20260916'
    if out.exists():
        raise FileExistsError(out)
    source = ROOT / 'data/processed/regional_air_models_full_history'
    stations = pd.read_parquet(source / 'station_hourly.parquet')
    weather = pd.read_parquet(source / 'weather_proxy.parquet')
    hashes = {str(Path(__file__).resolve()): digest(Path(__file__).resolve())}
    results, details = {}, {}
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
        frame = pd.read_parquet(folder / 'predictions.parquet')
        end = pd.Timestamp(f'{year + 1}-01-01', tz='Asia/Shanghai')
        cutoff = pd.Timestamp(f'{year - 1}-01-01', tz='Asia/Shanghai')
        station = stations.loc[stations.timestamp.lt(end)]
        met = weather.loc[weather.timestamp.lt(end)]
        panel = prepare_panel(station, met, cutoff,
                              training_start=pd.Timestamp('2014-01-01', tz='Asia/Shanghai'), ozone_column='o3_8h')
        assert panel['nodes'] == protocol['nodes']
        np.testing.assert_array_equal(panel['scaler'].mean, protocol['scaler']['mean'])
        times, nodes = panel['timestamps'], panel['nodes']
        raw = np.stack([station.pivot(index='timestamp', columns='node_id', values=key)
                        .reindex(index=times, columns=nodes).to_numpy()
                        for key in panel['feature_names'][:7]], axis=2)
        weather_raw = met.set_index('timestamp').reindex(times)[panel['feature_names'][7:11]].to_numpy()
        raw = np.concatenate([raw, np.repeat(weather_raw[:, None], len(nodes), axis=1)], axis=2)
        status = fill_provenance(raw)
        # Reproduce filled numeric inputs independently of causal_fill.
        filled = pd.DataFrame(raw.reshape(len(raw), -1)).ffill(limit=6).to_numpy().reshape(raw.shape)
        filled = np.where(np.isfinite(filled), filled, panel['scaler'].mean[:11]).astype('float32')
        normalized = ((filled - panel['scaler'].mean[:11]) / panel['scaler'].scale[:11]).astype('float32')
        np.testing.assert_array_equal(normalized, panel['inputs'][:, :, :11])
        positions = times.get_indexer(frame.timestamp)
        if (positions < 23).any():
            raise ValueError('Missing input history')
        active = active_node_mask(panel['targets'], positions)[:, :, 1]
        windows = np.stack([status[p-23:p+1] for p in positions])
        detail = frame.copy()
        detail['inactive_aqi_nodes'] = len(nodes) - active.sum(axis=1)
        for name, mask in (
            ('all_pollution', np.ones_like(windows[:, :, :, :7], dtype=bool)),
            ('active_pollution', np.broadcast_to(active[:, None, :, None], windows[:, :, :, :7].shape)),
            ('active_aqi', np.broadcast_to(active[:, None, :, None], windows[:, :, :, 2:3].shape)),
            ('weather', np.ones_like(windows[:, :, :1, 7:11], dtype=bool))):
            values = windows[:, :, :1, 7:11] if name == 'weather' else (
                windows[:, :, :, 2:3] if name == 'active_aqi' else windows[:, :, :, :7])
            denominator = mask.sum(axis=(1, 2, 3))
            for code, label in enumerate(('observed', 'forward', 'fallback')):
                detail[f'{name}_{label}_fraction'] = ((values == code) & mask).sum(axis=(1, 2, 3)) / denominator
        mature = detail.loc[detail.score_mature]
        groups = {}
        # Categories are fixed by the actual fill mechanism, not by errors or optimized cutoffs.
        for key in ('active_aqi_fallback_fraction', 'active_pollution_fallback_fraction', 'weather_fallback_fraction'):
            groups[key] = {}
            for label, mask in (('none', mature[key].eq(0)), ('any', mature[key].gt(0))):
                subset = mature.loc[mask]
                groups[key][label] = {'rows': len(subset), 'blocks': int(subset.continuity_block.nunique()),
                                     'paired': paired_error_summary(subset, subset.matched_aft, subset.airformer) if len(subset) else None}
        fractions = {key: float(mature[key].mean()) for key in detail if key.endswith('_fraction')}
        results[str(year)] = {'predicted_origins': len(detail), 'scored_origins': len(mature),
                              'mean_origin_fractions': fractions,
                              'inactive_aqi_node_counts': {str(k): int(v) for k, v in mature.inactive_aqi_nodes.value_counts().items()},
                              'groups': groups, 'numeric_input_reproduction': 'exact'}
        details[year] = detail
        del panel, raw, status, windows, filled, normalized
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Changed during audit: {path}')
    out.mkdir()
    for year, detail in details.items():
        detail.to_parquet(out / f'origins_{year}.parquet', index=False)
    report = {'years': results, 'input_sha256': hashes,
              'limitations': ['Observed means present in processed source, not proven real-time availability',
                              'Groups are descriptive and may share continuity blocks; not causal experiments',
                              'Past-only AQI support is used for active-station descriptive breakdown',
                              'First 11 source features audited; calendar and derived wind channels excluded from fractions'],
              'output_sha256': {p.name: digest(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
