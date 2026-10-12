"""Independent replay of materialized business states and recovery bounds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.features import MEASUREMENT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/task01_recovery_head/observed_business_dataset_v1'
SOURCES = {
    '2014': 'data/interim/cnemc_chengdu_2014_weather.parquet',
    '2015_2021': 'data/processed/city_labeled.parquet',
    '2022': 'data/interim/cnemc_chengdu_2022_weather.parquet',
    '2023': 'data/interim/cnemc_chengdu_2023_weather.parquet',
    '2024_2025': 'data/interim/cnemc_chengdu_2024_2025_weather.parquet',
}


def main():
    full = pd.read_parquet(OUT / 'features.parquet')
    origins = pd.read_parquet(OUT / 'origins.parquet')
    assert full.timestamp.is_unique
    assert full.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    raw = []
    for name, path in SOURCES.items():
        frame = pd.read_parquet(ROOT / path)[['timestamp', *MEASUREMENT_COLUMNS]]
        frame['source'] = name
        raw.append(frame)
    raw = pd.concat(raw).set_index('timestamp').sort_index()
    expected = raw.reindex(pd.DatetimeIndex(full.timestamp))
    pd.testing.assert_frame_equal(
        full[list(MEASUREMENT_COLUMNS)].reset_index(drop=True),
        expected[list(MEASUREMENT_COLUMNS)].reset_index(drop=True), check_dtype=False,
    )
    assert full.input_hour_observed.to_numpy().tolist() == full.timestamp.isin(raw.index).tolist()
    assert full.source.fillna('').tolist() == expected.source.fillna('').tolist()
    pd.testing.assert_frame_equal(
        full.loc[full.eligible].reset_index(drop=True),
        origins[full.columns].reset_index(drop=True),
    )
    aq = full.aqi.to_numpy()
    p25, p10 = full.pm2_5_iaqi.to_numpy(), full.pm10_iaqi.to_numpy()
    states, episode_ids, blocks, eligible = [], [], [], []
    episode, serial, block = 0, 0, 1
    for i in range(len(full)):
        clean_now = np.isfinite(aq[i]) and aq[i] <= 100
        if aq[i] > 100 and (p25[i] > 100 or p10[i] > 100):
            state = 'confirmed_pollution'
        elif clean_now or (aq[i] > 100 and p25[i] <= 100 and p10[i] <= 100):
            state = 'confirmed_not_pollution'
        else:
            state = 'unknown'
        states.append(state)
        six_clean = i >= 5 and np.all(np.isfinite(aq[i-5:i+1]) & (aq[i-5:i+1] <= 100))
        first_six = six_clean and (i == 5 or not (np.isfinite(aq[i-6]) and aq[i-6] <= 100))
        if not np.isfinite(aq[i]) or six_clean:
            episode = 0
        elif episode == 0 and i >= 2 and states[-3:] == ['confirmed_pollution'] * 3:
            serial += 1
            episode = serial
        episode_ids.append(episode)
        blocks.append(block)
        eligible.append(episode > 0 and state == 'confirmed_pollution')
        if first_six:
            block += 1
    assert states == full.pollution_state.tolist()
    assert episode_ids == full.episode_id.tolist()
    assert blocks == full.continuity_block.tolist()
    assert eligible == full.eligible.tolist()
    # Scan directly instead of calling the optimized production label generator.
    for row in origins.itertuples(index=False):
        i = int(full.timestamp.searchsorted(row.timestamp))
        end = i + 1
        while end < len(aq) and np.isfinite(aq[end]):
            end += 1
        candidates = [j for j in range(i + 1, end - 5) if np.all(aq[j:j+6] <= 100)]
        if candidates:
            j = candidates[0]
            assert row.exact_hours == row.lower_bound_hours == row.upper_bound_hours == j - i
            assert row.reason == 'exact' and row.informative
            assert row.label_available_at == full.timestamp.iloc[j+5]
        else:
            lower = max(0, end - i - 6) + 1
            assert row.lower_bound_hours == lower and np.isposinf(row.upper_bound_hours)
            assert np.isnan(row.exact_hours) and row.informative == (lower > 1)
            assert row.reason == ('gap' if end < len(aq) else 'end_of_data')
            assert row.label_available_at == full.timestamp.iloc[min(end, len(aq)-1)]
        assert row.observed_prefix_hours == end - i - 1
    result = {
        'status': 'PASS', 'grid_rows': len(full), 'origins': len(origins),
        'checks': ['raw_measurements', 'source_provenance', 'hour_grid', 'all_eligible_origins',
                   'independent_states_and_groups', 'all_bounds_direct_scan'],
        'verified_file_hashes': {
            name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
            for name in ('features.parquet', 'origins.parquet', 'report.json')
        },
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / 'independent_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
