"""Independent source-to-metric audit; never imports the training implementation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ('pm2_5', 'pm10', 'aqi')
ARMS = ('persistence', 'daily_repeat', 'pollution_only', 'all_conditions')
PARTS = ('dev', 'test', 'recent_audit')
CONDITIONS = ('all', 'clean_origin', 'polluted_origin')
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'data/processed/regional_air_models_full_history/city_hourly.parquet'
SOURCE_SHA = 'b16c1941da64f5ef19bc5856cd1215851c4130ffb80b18ca0c16af10f118ab0e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(frame):
    times = pd.DatetimeIndex(frame['timestamp'])
    assert times.tz is not None and times.is_unique and not times.hasnans
    times = times.tz_convert('Asia/Shanghai')
    assert np.all(np.diff(times.asi8) == 3_600_000_000_000)
    raw = frame[list(CHANNELS)].to_numpy(dtype=np.float64)
    indices = []
    for i in range(23, len(frame) - 24):
        segment = raw[i-23:i+25]
        years = times[[i-23, i, i+24]].year
        partitions = [0 if y <= 2023 else 1 if y == 2024 else 2 if y == 2025 else 3 for y in years]
        if len(set(partitions)) == 1 and np.isfinite(segment).all() and (segment >= 0).all():
            indices.append(i)
    indices = np.asarray(indices, dtype=np.int64)
    history = raw[indices[:, None] + np.arange(-23, 1)].astype(np.float32)
    target = raw[indices[:, None] + np.arange(1, 25)].astype(np.float32)
    year = times[indices].year
    meta = pd.DataFrame({'timestamp': times[indices],
        'partition': ['train' if y <= 2023 else 'dev' if y == 2024 else 'test' if y == 2025 else 'recent_audit' for y in year],
        'origin_aqi': history[:, -1, 2], 'polluted': history[:, -1, 2] > 100,
        'source_index': indices})
    return meta, history, target


def independent_scores(target, pred, days):
    assert target.shape == pred.shape and target.ndim == 3 and target.shape[1:] == (24, 3)
    assert len(target) == len(days) and len(target) > 0
    assert np.isfinite(target).all() and np.isfinite(pred).all() and (pred >= 0).all()
    days = np.asarray(days)
    result = {}
    for j, channel in enumerate(CHANNELS):
        errors = pred[:, :, j].astype(np.float64) - target[:, :, j].astype(np.float64)
        absolute = np.abs(errors)
        daily = [absolute[days == day].mean() for day in np.unique(days)]
        result[channel] = {'mae': float(absolute.mean()), 'rmse': float(np.sqrt(np.square(errors).mean())),
            'day_equal_mae': float(np.mean(daily)),
            **{f'mae_h{h}': float(absolute[:, h-1].mean()) for h in (1, 6, 12, 24)}}
    return result


def compare_tree(expected, actual):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), 'Metric keys differ'
        for key in expected:
            compare_tree(expected[key], actual[key])
    elif isinstance(expected, int):
        assert expected == actual
    else:
        assert np.isclose(expected, actual, rtol=1e-10, atol=1e-10), (expected, actual)


def audit(run):
    report_path = run / 'report.json'
    report_sha = sha(report_path)
    report = json.loads(report_path.read_text('utf-8'))
    assert report['status'] == 'completed_fixed_protocol'
    assert report['trajectory_target_order'] == list(CHANNELS)
    assert report['forecast_leads'] == list(range(1, 25))
    assert sha(SOURCE) == SOURCE_SHA
    expected_sources = {str(SOURCE), str(HERE/'trajectory.py'), str(HERE/'PROTOCOL.md')}
    assert set(report['source_sha256']) == expected_sources
    for path, digest in report['source_sha256'].items():
        assert sha(path) == digest, path
    expected_models = {f'{arm}_{channel}_h{h:02}.txt' for arm in ARMS[2:] for channel in CHANNELS for h in range(1,25)}
    assert set(report['models']) == expected_models
    for name, digest in report['models'].items():
        assert sha(run/name) == digest, name
    expected_artifacts = {'all_eligible_origins.parquet', 'evaluation_origins.parquet', 'targets.npy',
                          *(arm+'.npy' for arm in ARMS)}
    assert set(report['artifact_sha256']) == expected_artifacts
    for name, digest in report['artifact_sha256'].items():
        assert sha(run/name) == digest, name
    meta, histories, targets = reconstruct(pd.read_parquet(SOURCE))
    pd.testing.assert_frame_equal(meta, pd.read_parquet(run/'all_eligible_origins.parquet'))
    counts = {f'{p}/{c}': int(n) for (p, c), n in meta.groupby(['partition', 'polluted']).size().items()}
    assert counts == report['counts']
    evaluation = meta.partition.ne('train').to_numpy()
    eval_meta = meta.loc[evaluation].reset_index(drop=True)
    pd.testing.assert_frame_equal(eval_meta, pd.read_parquet(run/'evaluation_origins.parquet').reset_index(drop=True))
    targets = targets[evaluation]
    histories = histories[evaluation]
    assert np.array_equal(targets, np.load(run/'targets.npy', allow_pickle=False))
    days = eval_meta.timestamp.dt.strftime('%Y-%m-%d').to_numpy()
    metrics = {}
    for arm in ARMS:
        pred = np.load(run/(arm+'.npy'), allow_pickle=False)
        assert pred.shape == targets.shape and np.isfinite(pred).all() and (pred >= 0).all()
        if arm == 'persistence':
            assert np.array_equal(pred, np.broadcast_to(histories[:, -1:, :], pred.shape))
        elif arm == 'daily_repeat':
            assert np.array_equal(pred, histories)
        metrics[arm] = {}
        for part in PARTS:
            for condition in CONDITIONS:
                mask = eval_meta.partition.eq(part).to_numpy()
                if condition != 'all':
                    mask &= eval_meta.polluted.to_numpy() == (condition == 'polluted_origin')
                metrics[arm][part+'/'+condition] = {'origins': int(mask.sum()), 'days': int(len(np.unique(days[mask]))),
                    'scores': independent_scores(targets[mask], pred[mask], days[mask])}
    compare_tree(metrics, report['metrics'])
    differences = {}
    for arm in ARMS[:-1]:
        differences[arm] = {group: {channel: {key: score - metrics[arm][group]['scores'][channel][key]
            for key, score in values.items()} for channel, values in block['scores'].items()}
            for group, block in metrics['all_conditions'].items()}
    assert sha(report_path) == report_sha
    for path, digest in report['source_sha256'].items():
        assert sha(path) == digest
    return {'status': 'passed_independent_source_and_metric_audit', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'report_sha256': report_sha, 'auditor_sha256': sha(Path(__file__)), 'models_verified': len(expected_models),
        'artifacts_verified': len(expected_artifacts), 'eligible_origins_verified': len(meta),
        'evaluation_origins_verified': len(eval_meta), 'metrics': metrics,
        'paired_all_conditions_minus_control': differences,
        'difference_sign': 'negative is lower error for all_conditions; no significance claim',
        'limits': ['No causal improvement or multi-seed stability claim.',
                   'AQI trajectory error is not dissipation-time error.',
                   'Models hash-verified; independent model inference is not repeated here.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    (args.run/'independent_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False), 'utf-8')
    print(json.dumps({key: value for key, value in result.items() if key not in ('metrics', 'paired_all_conditions_minus_control')}))


if __name__ == '__main__':
    main()
