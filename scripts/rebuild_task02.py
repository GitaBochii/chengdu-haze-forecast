"""Reconstruct the missing weather triplet with explicit new proxy assumptions."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.noaa_isd import prepare_noaa_hourly
from haze_forecast.cnemc_weather import add_noaa_weather
from haze_forecast.operational_gefs_benchmark import safe_event_year_split
from haze_forecast.training import AFTSurvivalRegressor
from haze_forecast.weather_source_ablation import WEATHER_BASE, WEATHER_FIELDS, feature_arms, uniform_weather_rows
from scripts.benchmark_operational_gefs_ensemble import _prepare
from scripts.benchmark_weather_source_ablation import _sha

ROOT = Path(__file__).resolve().parents[1]


def delayed_grid(hourly, times):
    if hourly.timestamp.duplicated().any():
        raise ValueError('Ambiguous weather hour')
    grid = hourly.set_index('timestamp')[list(WEATHER_BASE)].copy()
    grid.index = pd.DatetimeIndex(grid.index).tz_convert('UTC') + pd.Timedelta(hours=72)
    result = grid.reindex(times).rename_axis('timestamp').reset_index()
    return result


def main():
    out = ROOT / 'artifacts/task02_rebuild_20260910_v1'
    if out.exists():
        raise FileExistsError(out)
    processed = ROOT / 'data/processed'
    sources = {'prefix_2014': processed / 'cnemc_chengdu_2014_training_features.parquet',
               'base': processed / 'training_features.parquet',
               **{f'external_{y}': processed / f'cnemc_chengdu_{y}_training_features.parquet'
                  for y in range(2022, 2026)}}
    weather_path = processed / 'regional_air_models_full_history/weather_proxy.parquet'
    raw_paths = [ROOT / f'data/raw/noaa_isd/{y}/56294099999.csv' for y in range(2014, 2026)]
    deployed = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    metadata = json.loads((deployed / 'model_metadata.json').read_bytes())
    names = feature_arms(metadata['feature_names'])['mixed_four']
    assert len(names) == 74
    paths = [*sources.values(), weather_path, *raw_paths, *deployed.glob('*'),
             Path(__file__).resolve(), ROOT / 'src/haze_forecast/noaa_isd.py',
             ROOT / 'src/haze_forecast/weather_source_ablation.py',
             ROOT / 'src/haze_forecast/features.py', ROOT / 'src/haze_forecast/training.py',
             ROOT / 'src/haze_forecast/cnemc_weather.py',
             ROOT / 'src/haze_forecast/operational_gefs_benchmark.py',
             ROOT / 'src/haze_forecast/temporal_validation.py',
             ROOT / 'scripts/benchmark_operational_gefs_ensemble.py']
    hashes = {str(p): _sha(p) for p in paths if p.is_file()}
    legacy = json.loads((ROOT / 'mision/20260908_task02_final_delivery/audit.json').read_bytes())
    for p in [*sources.values(), weather_path]:
        if _sha(p) != legacy['key_input_sha256'][p.name]:
            raise ValueError(f'Historical shared input changed: {p}')
    rows = pd.concat([_prepare(pd.read_parquet(p), key) for key, p in sources.items()],
                     ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    if rows.timestamp.duplicated().any():
        raise ValueError('Overlapping original origins')
    era5 = pd.read_parquet(weather_path)
    times = pd.DatetimeIndex(era5.timestamp).tz_convert('UTC')
    raw = pd.concat([prepare_noaa_hourly(pd.read_csv(p, low_memory=False)) for p in raw_paths],
                    ignore_index=True)
    raw = add_noaa_weather(raw[['timestamp']], raw)
    grids = {'offline_era5': era5[['timestamp', *WEATHER_BASE]].copy(),
             'strict_abstention': era5[['timestamp', *WEATHER_BASE]].copy(),
             'proxy_72h': delayed_grid(raw, times)}
    grids['strict_abstention'][list(WEATHER_BASE)] = np.nan
    frames = {key: uniform_weather_rows(rows, grid) for key, grid in grids.items()}
    untouched = [c for c in rows if c not in WEATHER_FIELDS]
    for frame in frames.values():
        pd.testing.assert_frame_equal(frame[untouched], rows[untouched])
    protocol = {
        'identity': 'New reconstruction, not byte-identical reproduction of lost Task02 phases',
        'features': names, 'years': [2022, 2023, 2024, 2025], 'seed': 42, 'rounds': 500,
        'input_sha256': hashes, 'arms': list(frames),
        'proxy_policy': 'ISD 56294099999 only; quality parser; ceil observation hour then +72h; exact hour lookup; no fill/interpolation',
        'strict_policy': 'All four weather values absent; no replacement with retrospective data',
        'training_policy': 'Existing old-label event-isolated annual split, 2014 onwards; mature labels before cutoff',
        'selection': 'Evaluate all 12 fixed year/arm combinations, no tuning or deployment',
        'limitations': ['72h is an assumption, not proof of historical publication latency',
                        'Annual ISD files may contain retrospective revisions',
                        'ERA5 retrospective; evaluation years already explored',
                        'Original proxy source mixing/fill rules were not delivered',
                        'Old exact-label 74-feature protocol, not new interval-label GEFS scorecard'],
        'deployment_authorized': False,
    }
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    for key in frames:
        grids[key].to_parquet(out / f'{key}_hourly.parquet', index=False)
        frames[key].to_parquet(out / f'{key}_features.parquet', index=False)
    scores = {}
    for year in protocol['years']:
        scores[str(year)] = {}
        reference = None
        for arm, frame in frames.items():
            train, test = safe_event_year_split(frame, year)
            keys = ['timestamp', 'event_key', 'remaining_hours']
            if reference is None:
                reference = (train[keys].copy(), test[keys].copy())
            else:
                pd.testing.assert_frame_equal(train[keys], reference[0])
                pd.testing.assert_frame_equal(test[keys], reference[1])
            model = AFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit(
                train[names], train.remaining_hours)
            checkpoint = out / f'{year}_{arm}.joblib'
            joblib.dump(model, checkpoint)
            prediction = model.predict(test[names])
            np.testing.assert_array_equal(prediction, joblib.load(checkpoint).predict(test[names]))
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite prediction')
            saved = test[keys].copy()
            saved['prediction'] = prediction
            saved.to_parquet(out / f'{year}_{arm}_predictions.parquet', index=False)
            error = abs(saved.prediction - saved.remaining_hours)
            weather_fields = [c for c in names if c in WEATHER_FIELDS]
            tail = saved.remaining_hours.ge(24)
            scores[str(year)][arm] = {
                'row_mae': float(error.mean()),
                'event_equal_mae': float(error.groupby(saved.event_key).mean().mean()),
                'tail24_event_mae': float(error[tail].groupby(saved.loc[tail, 'event_key']).mean().mean()),
                'train_rows': len(train), 'test_rows': len(test),
                'test_events': int(test.event_key.nunique()), 'prediction_coverage': 1.,
                'weather_value_coverage': float(test[weather_fields].notna().to_numpy().mean()),
            }
            print(json.dumps({'year': year, 'arm': arm, **scores[str(year)][arm]}), flush=True)
    if hashes != {p: _sha(Path(p)) for p in hashes}:
        raise ValueError('Inputs or deployment changed')
    report = {'scores': scores, 'deployment_authorized': False,
              'output_sha256': {p.name: _sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
