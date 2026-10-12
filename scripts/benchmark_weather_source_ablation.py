"""Run matched AFT meteorology ablations without changing targets or deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.operational_gefs_benchmark import safe_event_year_split
from haze_forecast.weather_source_ablation import (
    WEATHER_FIELDS,
    feature_arms,
    uniform_weather_rows,
)
from scripts.benchmark_operational_gefs_ensemble import _prepare, _score

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_hashes(path):
    return {p.name: _sha(p) for p in sorted(path.glob('*')) if p.is_file()}


def _write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/weather_source_ablation_v1')
    parser.add_argument('--weather', type=Path,
                        default=ROOT / 'data/processed/regional_air_models_full_history/weather_proxy.parquet')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new output directory; evidence is immutable')
    deployed = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    before = _directory_hashes(deployed)
    metadata = json.loads((deployed / 'model_metadata.json').read_text(encoding='utf-8'))
    groups = feature_arms(metadata['feature_names'])
    processed = ROOT / 'data/processed'
    sources = {'prefix_2014': processed / 'cnemc_chengdu_2014_training_features.parquet',
               'base': processed / 'training_features.parquet',
               **{f'external_{y}': processed / f'cnemc_chengdu_{y}_training_features.parquet'
                  for y in range(2022, 2026)}}
    input_paths = [*sources.values(), args.weather]
    snapshot = {str(p): _sha(p) for p in input_paths}
    rows = pd.concat([_prepare(pd.read_parquet(path), name) for name, path in sources.items()],
                     ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    if rows.duplicated(['timestamp', 'event_key']).any():
        raise ValueError('duplicate original event keys')
    uniform = uniform_weather_rows(rows, pd.read_parquet(args.weather))
    pd.testing.assert_frame_equal(rows.drop(columns=list(WEATHER_FIELDS), errors='ignore'),
                                  uniform.drop(columns=list(WEATHER_FIELDS), errors='ignore'))
    args.output.mkdir(parents=True)
    predictions = args.output / 'predictions'
    predictions.mkdir()
    protocol = {'arms': groups, 'development_years': [2022, 2023], 'confirmation_years': [2024, 2025],
                'selection_rule': 'minimum mean development-year row MAE; baseline wins exact ties',
                'training': '2014 onwards expanding windows; confirmed labels before Shanghai year boundary',
                'input_sha256': snapshot, 'deployment_before': before,
                'limitations': ['All evaluation years previously explored, not pristine holdouts.',
                                'Uniform ERA5 is a retrospective city proxy, not real-time weather.',
                                'Common four-weather arms omit all PBL fields and indicators.',
                                'Source replacement also changes geographic representativeness.',
                                'AFT full-duration metrics are not neural 68-hour restricted metrics.',
                                'AFT predictions are saved, not experiment model weights.']}
    _write(args.output / 'protocol.json', protocol)
    scores, coverage, selected, frozen_hash = {}, {}, None, None
    for year in (2022, 2023, 2024, 2025):
        train, test = safe_event_year_split(rows, year)
        uniform_train, uniform_test = safe_event_year_split(uniform, year)
        for left, right in ((train, uniform_train), (test, uniform_test)):
            pd.testing.assert_frame_equal(left[['timestamp', 'event_key', 'remaining_hours']],
                                          right[['timestamp', 'event_key', 'remaining_hours']])
        names = list(groups) if year <= 2023 else list(dict.fromkeys(['baseline', selected]))
        scores[str(year)] = {}
        for name in names:
            tr, te = (uniform_train, uniform_test) if name == 'uniform_four' else (train, test)
            scores[str(year)][name] = _score(tr, te, groups[name], predictions / f'{name}_{year}.parquet')
            print(json.dumps({'year': year, 'arm': name, **scores[str(year)][name]}), flush=True)
        fields = [f for f in groups['mixed_four'] if f in WEATHER_FIELDS]
        coverage[str(year)] = {name: float(frame[fields].notna().to_numpy().mean())
                               for name, frame in (('mixed_four', test), ('uniform_four', uniform_test))}
        if year == 2023:
            means = {name: float(np.mean([scores[str(y)][name]['row_mae'] for y in (2022, 2023)]))
                     for name in groups}
            selected = min(means, key=means.__getitem__)
            _write(args.output / 'selection.json', {'selected_arm': selected, 'development_mean_mae': means,
                                                    'scores': scores.copy()})
            frozen_hash = _sha(args.output / 'selection.json')
            print(f'Selection frozen: {selected}', flush=True)
        elif year > 2023 and _sha(args.output / 'selection.json') != frozen_hash:
            raise RuntimeError('development selection changed during confirmation')
    if snapshot != {str(p): _sha(p) for p in input_paths}:
        raise RuntimeError('benchmark inputs changed after loading')
    after = _directory_hashes(deployed)
    if before != after:
        raise RuntimeError('deployment artifacts changed')
    passed = all(scores[str(y)][selected][metric] < scores[str(y)]['baseline'][metric]
                 for y in (2024, 2025) for metric in ('row_mae', 'event_equal_mae'))
    _write(args.output / 'report.json', {**protocol, 'scores': scores, 'selected_arm': selected,
                                       'selection_sha256': frozen_hash, 'weather_value_coverage': coverage,
                                       'deployment_after': after,
                                       'decision': {'status': 'research_candidate_only' if passed else 'reject',
                                                    'metric_gate_passed': passed,
                                                    'deployment_artifact_changed': False}})
    print(f'Report: {args.output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
