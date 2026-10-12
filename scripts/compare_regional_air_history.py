"""Independently rescore saved historical experiments on identical verified origins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from haze_forecast.regional_air_experiment import recovery_metrics


def aligned_duration_scores(frames: dict[str, pd.DataFrame], horizon: int = 72) -> dict:
    selected = {}
    for name, frame in frames.items():
        if frame.timestamp.duplicated().any():
            raise ValueError('duplicate forecast origins')
        selected[name] = frame.loc[frame.label_verified_in_scoring_window
                                    & frame.actual_hours.notna()].set_index('timestamp')
    common = next(iter(selected.values())).index
    for frame in selected.values():
        common = common.intersection(frame.index)
    common = common.sort_values()
    if not len(common):
        raise ValueError('no common verified forecast origins')
    truth = next(iter(selected.values())).loc[common, ['event_id', 'actual_hours']]
    scores = {}
    for name, frame in selected.items():
        aligned = frame.loc[common]
        if not aligned[['event_id', 'actual_hours']].equals(truth):
            raise ValueError('truth differs between experiments')
        scores[name] = recovery_metrics(aligned, horizon)
    return {'common_event_rows': len(common), 'common_events': int(truth.event_id.nunique()),
            'common_timestamps': [time.isoformat() for time in common], 'models': scores}


def validate_reports(reports: dict) -> None:
    if any(r['deployment_artifact_changed'] for r in reports.values()):
        raise ValueError('deployment artifacts changed')
    if reports['full']['input_data_sha256'] != reports['recent']['input_data_sha256']:
        raise ValueError('experiments must use the same input dataset')
    for arm, expected_start in [('full', 2014), ('recent', 2023)]:
        config = reports[arm]['configuration']
        if (config['train_start_year'], config['train_end_year']) != (expected_start, 2023):
            raise ValueError('unexpected training years')
        if config['horizon'] != 72 or config['max_train'] or config['max_eval']:
            raise ValueError('comparison requires 72-hour forecasts and uncapped sample counts')
    for key in ('history', 'horizon', 'ozone_column', 'validation_stride', 'eval_stride',
                'seed', 'epochs', 'patience', 'learning_rate', 'batch_size', 'train_stride',
                'max_train', 'max_eval'):
        if reports['full']['configuration'][key] != reports['recent']['configuration'][key]:
            raise ValueError(f'evaluation protocol differs: {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', type=Path, required=True)
    parser.add_argument('--recent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = {name: json.loads((path / 'report.json').read_text(encoding='utf-8'))
               for name, path in [('full', args.full), ('recent', args.recent)]}
    validate_reports(reports)
    result = {'years': {}, 'deployment_artifact_changed': False,
              'nodes': {name: r['nodes'] for name, r in reports.items()},
              'training_windows': {name: r['training_windows_by_year'] for name, r in reports.items()},
              'limits': ['68-hour restricted duration, not full-duration MAE',
                         'Node selection may differ; not a pure history-length ablation',
                         'Both evaluation periods previously inspected; no automatic promotion',
                         ('This comparison aligns duration only; per-arm concentration metrics '
                          'are supplementary, not an aligned concentration comparison')]}
    models = ['matched_aft', 'persistence', 'gru', 'pm25_gnn', 'airformer']
    for year in (2025, 2026):
        frames = {f'{arm}/{model}': pd.read_parquet(folder / f'{model}_{year}_recovery.parquet')
                  for arm, folder in [('full', args.full), ('recent', args.recent)] for model in models}
        result['years'][str(year)] = aligned_duration_scores(frames,
                                                reports['full']['configuration']['horizon'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
