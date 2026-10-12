"""Independently replay frozen LSTM/GRU runs; write only verification.json."""

import hashlib
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/task01_recovery_head'
OUT = BASE / 'lstm_direct_crossyear_20261012'
OLD = BASE / 'direct_history_crossyear_20260918'
YEARS = (2024, 2025)
SEEDS = (24, 42, 2026)
ARMS = ('gru_history', 'lstm_history', 'lstm_repeat')
SUMMARY_METRICS = (
    'exact_row_mae', 'exact_block_mae', 'tail24_block_mae',
    'all_row_error_lower', 'all_row_error_upper',
    'all_block_error_lower', 'all_block_error_upper',
)
REPLAY_RTOL = 1e-7
REPLAY_ATOL = 1e-7


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def compare(actual, expected, location='root'):
    if isinstance(actual, dict):
        assert isinstance(expected, dict) and actual.keys() == expected.keys(), location
        for name, value in actual.items():
            compare(value, expected[name], f'{location}.{name}')
    elif isinstance(actual, list):
        assert isinstance(expected, list) and len(actual) == len(expected), location
        for index, value in enumerate(actual):
            compare(value, expected[index], f'{location}[{index}]')
    elif actual is None or isinstance(actual, (bool, str)):
        assert actual == expected, (location, actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10,
                                   err_msg=location)


def check_hashes(hashes):
    assert isinstance(hashes, dict) and hashes
    for path, expected in hashes.items():
        assert digest(path) == expected, f'Changed frozen input: {path}'


def average(values, mask, blocks=None):
    selected = np.asarray(values)[mask]
    if not len(selected):
        return None
    if blocks is None:
        return float(selected.mean())
    return float(pd.DataFrame({'v': selected, 'b': np.asarray(blocks)[mask]})
                 .groupby('b').v.mean().mean())


def independent_scores(frame, prediction):
    prediction = np.asarray(prediction, dtype=float)
    lo = frame.lower_bound_hours.to_numpy(float)
    hi = frame.upper_bound_hours.to_numpy(float)
    exact = frame.exact_hours.notna().to_numpy()
    blocks = frame.continuity_block.to_numpy()
    assert prediction.shape == lo.shape
    assert np.isfinite(prediction).all() and (prediction > 0).all()
    assert np.isfinite(lo).all() and (lo > 0).all()
    assert ((np.isfinite(hi) | np.isposinf(hi)) & (hi >= lo)).all()
    np.testing.assert_array_equal(exact, lo == hi)
    np.testing.assert_array_equal(frame.exact_hours.to_numpy()[exact], lo[exact])
    minimum = np.maximum(lo - prediction, np.maximum(prediction - hi, 0.))
    maximum = np.maximum(np.abs(prediction - lo), np.abs(prediction - hi))
    all_rows = np.ones(len(frame), dtype=bool)
    uncertain, tail = ~exact, exact & (lo >= 24.)
    return {
        'rows': len(frame), 'blocks': len(set(blocks)), 'prediction_coverage': 1.,
        'exact_rows': int(exact.sum()), 'exact_blocks': len(set(blocks[exact])),
        'finite_interval_rows': int((uncertain & np.isfinite(hi)).sum()),
        'right_censored_rows': int(np.isposinf(hi).sum()),
        'exact_row_mae': average(minimum, exact),
        'exact_block_mae': average(minimum, exact, blocks),
        'tail24_rows': int(tail.sum()), 'tail24_block_mae': average(minimum, tail, blocks),
        'all_row_error_lower': average(minimum, all_rows),
        'all_row_error_upper': average(maximum, all_rows) if np.isfinite(hi).all() else None,
        'all_block_error_lower': average(minimum, all_rows, blocks),
        'all_block_error_upper': average(maximum, all_rows, blocks) if np.isfinite(hi).all() else None,
        'uncertain_blocks': len(set(blocks[uncertain])),
        'uncertain_row_distance': average(minimum, uncertain),
        'uncertain_block_distance': average(minimum, uncertain, blocks),
        'uncertain_below_fraction': average(prediction < lo, uncertain),
    }


def independent_paired(frame, baseline, candidate):
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    assert len(baseline) == len(candidate) == len(frame)
    assert np.isfinite(baseline).all() and np.isfinite(candidate).all()
    assert (baseline > 0).all() and (candidate > 0).all()
    bounds = []
    # Absolute-error differences are piecewise linear with knots at both predictions.
    for a, b, lo, hi in zip(baseline, candidate, frame.lower_bound_hours,
                           frame.upper_bound_hours, strict=True):
        points = [lo, max(lo, min(a, hi)), max(lo, min(b, hi))]
        differences = [abs(b - y) - abs(a - y) for y in points]
        differences.append(abs(b - hi) - abs(a - hi) if np.isfinite(hi) else a - b)
        bounds.append((min(differences), max(differences)))
    lower, upper = np.asarray(bounds).T
    blocks = frame.continuity_block.to_numpy()
    exact = frame.exact_hours.notna().to_numpy()
    hi, uncertain = frame.upper_bound_hours.to_numpy(), ~exact
    result = {
        'rows': len(frame), 'blocks': len(set(blocks)), 'prediction_coverage': 1.,
        'exact_rows': int(exact.sum()), 'exact_blocks': len(set(blocks[exact])),
        'uncertain_rows': int(uncertain.sum()), 'uncertain_blocks': len(set(blocks[uncertain])),
        'finite_interval_rows': int((uncertain & np.isfinite(hi)).sum()),
        'right_censored_rows': int(np.isposinf(hi).sum()),
        'exact_row_delta': average(lower, exact),
        'exact_block_delta': average(lower, exact, blocks),
    }
    for subgroup, mask in [('all', np.ones(len(frame), bool)), ('uncertain', uncertain)]:
        for side, values in [('lower', lower), ('upper', upper)]:
            result[f'{subgroup}_row_delta_{side}'] = average(values, mask)
            result[f'{subgroup}_block_delta_{side}'] = average(values, mask, blocks)
    return result


def standardize(values, members):
    values = np.asarray(values, dtype=np.float64)
    members = np.asarray(members, dtype=bool)
    assert members.shape == (len(values),) and members.any()
    assert not np.isinf(values).any()
    fitting = values[members].reshape(-1, values.shape[-1])
    means, scales, support = [], [], []
    for column in fitting.T:
        observed = column[~np.isnan(column)]
        means.append(float(observed.mean()) if len(observed) else 0.)
        std = float(observed.std(ddof=0)) if len(observed) else 1.
        scales.append(std if std > 1e-6 else 1.)
        support.append(bool(len(observed)))
    mean, scale, supported = np.array(means), np.array(scales), np.array(support)
    numeric = np.where(np.isfinite(values) & supported, (values - mean) / scale, 0.)
    transformed = np.concatenate((numeric, np.isnan(values).astype(float)), axis=-1)
    return transformed.astype(np.float32), mean, scale, supported


class IndependentNetwork(nn.Module):
    def __init__(self, arm):
        super().__init__()
        assert arm in ARMS
        self.is_lstm = arm.startswith('lstm')
        recurrent = nn.LSTM if self.is_lstm else nn.GRU
        self.history = recurrent(24, 16, num_layers=1, batch_first=True,
                                 bidirectional=False)
        self.snapshot = nn.Sequential(nn.Linear(208, 32), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(48, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, history, snapshot):
        _, state = self.history(history)
        hidden = state[0] if self.is_lstm else state
        return self.output(torch.cat((hidden[-1], self.snapshot(snapshot)), dim=1)).squeeze(1)


def replay_weights(path, arm, arrays, indices, preserve_slice_layout=False):
    indices = np.asarray(indices, dtype=int)
    assert indices.ndim == 1 and len(indices)
    if preserve_slice_layout:
        np.testing.assert_array_equal(indices, np.arange(indices[0], indices[0] + len(indices)))
    model = IndependentNetwork(arm).cuda().eval()
    parameters = sum(value.numel() for value in model.parameters())
    assert parameters == (9505 if arm == 'gru_history' else 10177)
    model.load_state_dict(torch.load(path, weights_only=True, map_location='cuda'), strict=True)
    prediction = []
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            rows = indices[start:start + 128]
            # Final inference uses slices; fancy indexing changes snapshot strides.
            selection = slice(rows[0], rows[-1] + 1) if preserve_slice_layout else rows
            history = torch.from_numpy(arrays['history'][selection]).cuda()
            snapshot = torch.from_numpy(arrays['snapshot'][selection]).cuda()
            prediction.append(torch.exp(model(history, snapshot).double()).cpu().numpy())
    prediction = np.concatenate(prediction)
    assert np.isfinite(prediction).all() and (prediction > 0).all()
    return prediction, parameters


def validate_selection(run):
    trace, refit = run['selection_trace'], run['refit_trace']
    assert 1 <= len(trace) <= 60
    assert [row['epoch'] for row in trace] == list(range(1, len(trace) + 1))
    assert all(np.isfinite([row['training_nll'], row['selection_block_mae']]).all()
               for row in trace)
    best = min(trace, key=lambda row: row['selection_block_mae'])['epoch']
    assert best == run['selected_epoch']
    assert [row['epoch'] for row in refit] == list(range(1, best + 1))
    assert all(np.isfinite(row['training_nll']) for row in refit)
    best_at, best_value = 0, float('inf')
    for row in trace:
        if row['selection_block_mae'] < best_value:
            best_at, best_value = row['epoch'], row['selection_block_mae']
        if row['epoch'] < len(trace):
            assert row['epoch'] - best_at < 8
    assert len(trace) == 60 or len(trace) - best_at == 8
    return best


def validate_normalization(folder, history, snapshot, fitting, full_training):
    results, arrays = {}, {}
    assert history.shape[1:] == (24, 12) and snapshot.shape[1:] == (104,)
    for mode in ('history', 'repeat'):
        values = history if mode == 'history' else np.repeat(history[:, -1:], 24, axis=1)
        for stage, members in [('final', full_training), ('selection', fitting)]:
            name = f'{mode}_inputs.npz' if stage == 'final' else f'selection_{mode}_inputs.npz'
            with np.load(folder / name, allow_pickle=False) as saved:
                actual = {key: saved[key] for key in saved.files}
            expected_keys = {name + suffix for name in ('history', 'snapshot')
                             for suffix in ('', '_mean', '_scale', '_supported')}
            assert actual.keys() == expected_keys, (name, actual.keys())
            result = {}
            for key, raw in [('history', values), ('snapshot', snapshot)]:
                data, mean, scale, supported = standardize(raw, members)
                assert actual[key].dtype == np.float32
                assert actual[key + '_mean'].dtype == np.float64
                assert actual[key + '_scale'].dtype == np.float64
                assert actual[key + '_supported'].dtype == np.bool_
                np.testing.assert_allclose(actual[key + '_mean'], mean, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(actual[key + '_scale'], scale, rtol=1e-10, atol=1e-10)
                np.testing.assert_array_equal(actual[key + '_supported'], supported)
                np.testing.assert_allclose(actual[key], data, rtol=1e-6, atol=1e-6)
                assert np.isfinite(actual[key]).all()
                np.testing.assert_array_equal(actual[key][..., raw.shape[-1]:], np.isnan(raw))
                result[key] = {'supported_features': int(supported.sum()),
                               'fitting_rows': int(members.sum())}
            arrays[(mode, stage)] = actual
            results[f'{mode}_{stage}'] = result
    return arrays, results


def summaries(scores):
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for metric in SUMMARY_METRICS:
            values = np.asarray([scores[f'{arm}_seed{seed}'][metric] for seed in SEEDS])
            assert np.isfinite(values).all()
            result[arm][metric] = {'mean': float(values.mean()), 'min': float(values.min()),
                                   'max': float(values.max())}
    return result


def exact_advantage(native, scores, summary):
    metrics = ('exact_row_mae', 'exact_block_mae')
    mean = summary['lstm_history']
    improves = sum(all(scores[f'lstm_history_seed{seed}'][key] < native[key]
                       for key in metrics) for seed in SEEDS)
    return bool(all(mean[key]['mean'] < native[key] for key in metrics)
                and mean['tail24_block_mae']['mean'] <= native['tail24_block_mae']
                and improves >= 2)


def verify_year(year, protocol, hourly, canonical):
    folder, old_folder = OUT / str(year), OLD / str(year)
    report = read_json(folder / 'report.json')
    train = pd.read_parquet(folder / 'training.parquet')
    development = pd.read_parquet(folder / 'selection_rows.parquet')
    selection_train = pd.read_parquet(folder / 'selection_training.parquet')
    for name, actual in [('training', train), ('selection_rows', development),
                         ('selection_training', selection_train)]:
        pd.testing.assert_frame_equal(actual, pd.read_parquet(old_folder / f'{name}.parquet'))
    original_train = pd.read_parquet(BASE / f'original_gefs_without_pbl_crossyear_v1/{year}_training_rows.parquet')
    pd.testing.assert_frame_equal(train, original_train)
    features = pd.read_parquet(BASE / f'reusable_gefs_crossyear_v1/{year}_features.parquet')
    original_predictions = pd.read_parquet(BASE / f'reusable_gefs_crossyear_v1/{year}_predictions.parquet')
    predictions = pd.read_parquet(folder / 'all_predictions.parquet')
    scored = pd.read_parquet(folder / 'scored_predictions.parquet')
    prefixes = [f'{arm}_seed{seed}' for arm in ARMS for seed in SEEDS]
    assert list(predictions.columns[:2]) == ['timestamp', 'native']
    assert len(predictions.columns) == 11 and set(predictions.columns) == {'timestamp', 'native', *prefixes}
    assert set(report['runs']) == set(prefixes)
    np.testing.assert_array_equal(predictions.timestamp, original_predictions.timestamp)
    assert predictions.timestamp.is_unique and train.timestamp.is_unique and scored.timestamp.is_unique
    cohort = canonical[canonical.timestamp.dt.year.eq(year)].reset_index(drop=True)
    pd.testing.assert_frame_equal(scored[canonical.columns].reset_index(drop=True), cohort)
    old_scored = pd.read_parquet(old_folder / 'scored_predictions.parquet')
    preserved = [column for column in old_scored.columns if column not in ('snapshot', 'history')]
    pd.testing.assert_frame_equal(scored[preserved], old_scored[preserved])
    assert set(predictions.timestamp) == set(scored.timestamp)
    for column in ('lower_bound_hours', 'upper_bound_hours', 'exact_hours',
                   'label_available_at', 'continuity_block'):
        np.testing.assert_array_equal(train[column], canonical.set_index('timestamp').loc[train.timestamp, column])
    cutoff = pd.Timestamp(f'{year - 1}-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(end).all() and train.label_available_at.lt(end).all()
    assert scored.label_available_at.lt(end + pd.DateOffset(years=1)).all()
    assert not set(train.continuity_block) & set(scored.continuity_block)
    selection = train.timestamp.ge(cutoff) & train.label_available_at.lt(end)
    fitting = (train.timestamp.lt(cutoff) & train.label_available_at.lt(cutoff)
               & ~train.continuity_block.isin(train.loc[selection, 'continuity_block']))
    pd.testing.assert_frame_equal(development, train.loc[selection].reset_index(drop=True))
    pd.testing.assert_frame_equal(selection_train, train.loc[fitting].reset_index(drop=True))
    assert not set(selection_train.continuity_block) & set(development.continuity_block)
    assert selection_train.label_available_at.lt(cutoff).all()
    expected = {2024: (6770, 396, 7166, 335, 222, 335),
                2025: (7166, 335, 7624, 266, 335, 243)}[year]
    assert (int(fitting.sum()), int(selection.sum()), len(train), len(scored),
            int(development.exact_hours.notna().sum()), int(scored.exact_hours.notna().sum())) == expected
    deferred = ~(fitting | selection)
    assert int(deferred.sum()) == (123 if year == 2025 else 0)
    if deferred.any():
        assert train.loc[deferred].timestamp.dt.year.eq(2023).all()
        assert train.loc[deferred].label_available_at.ge(cutoff).all()

    with np.load(folder / 'raw_inputs.npz', allow_pickle=False) as raw:
        assert set(raw.files) == {'history', 'snapshot', 'timestamps'}
        raw = {name: raw[name] for name in raw.files}
    with np.load(old_folder / 'raw_inputs.npz', allow_pickle=False) as old_raw:
        for name, values in raw.items():
            np.testing.assert_array_equal(values, old_raw[name])
    times = pd.DatetimeIndex(pd.concat([train.timestamp, predictions.timestamp], ignore_index=True))
    np.testing.assert_array_equal(times.asi8, raw['timestamps'])
    source = hourly.set_index('timestamp')
    assert source.index.is_unique and source.index.is_monotonic_increasing
    history = np.stack([source.loc[pd.to_datetime(times.asi8 - lag * 3_600_000_000_000, utc=True)
                                  .tz_convert('Asia/Shanghai'), protocol['history_features']].to_numpy(float)
                        for lag in range(23, -1, -1)], axis=1)
    np.testing.assert_array_equal(history, raw['history'])
    snapshot = np.concatenate([train[protocol['features']].to_numpy(float),
                               features[protocol['features']].to_numpy(float)])
    np.testing.assert_array_equal(snapshot, raw['snapshot'])
    current = np.concatenate([train[protocol['history_features']].to_numpy(float),
                              features[protocol['history_features']].to_numpy(float)])
    np.testing.assert_array_equal(history[:, -1], current)
    native = joblib.load(BASE / f'reusable_gefs_crossyear_v1/{year}.joblib').predict(features[protocol['features']])
    np.testing.assert_array_equal(native, original_predictions.forecast_aft)
    np.testing.assert_array_equal(native, predictions.native)
    full_members = np.arange(len(times)) < len(train)
    selection_members = np.r_[fitting.to_numpy(), np.zeros(len(predictions), dtype=bool)]
    arrays, normalization = validate_normalization(folder, history, snapshot,
                                                    selection_members, full_members)
    for column in ('native', *prefixes):
        np.testing.assert_array_equal(predictions.set_index('timestamp').loc[scored.timestamp, column],
                                       scored[column])
    native_score = independent_scores(scored, scored.native.to_numpy())
    compare(native_score, report['native'], f'{year}.native')
    compare(native_score, read_json(old_folder / 'report.json')['scores']['native'], f'{year}.old_native')
    scores, paired, runs = {}, {}, {}
    for arm in ARMS:
        mode = 'repeat' if arm == 'lstm_repeat' else 'history'
        for seed in SEEDS:
            prefix = f'{arm}_seed{seed}'
            run = read_json(folder / f'{prefix}_run.json')
            compare(run, report['runs'][prefix], f'{year}.{prefix}.stored_run')
            assert (run['year'], run['arm'], run['seed']) == (year, arm, seed)
            best = validate_selection(run)
            replay, parameters = replay_weights(folder / f'{prefix}.pt', arm,
                                                 arrays[(mode, 'final')],
                                                 np.arange(len(train), len(times)),
                                                 preserve_slice_layout=True)
            np.testing.assert_allclose(replay, predictions[prefix], rtol=REPLAY_RTOL,
                                       atol=REPLAY_ATOL, err_msg=f'{year}.{prefix}.final_replay')
            assert parameters == run['parameters']
            selection_replay, _ = replay_weights(folder / f'{prefix}_selection.pt', arm,
                                                  arrays[(mode, 'selection')],
                                                  np.flatnonzero(selection.to_numpy()))
            development_mae = independent_scores(development, selection_replay)['exact_block_mae']
            np.testing.assert_allclose(development_mae,
                                       run['selection_trace'][best - 1]['selection_block_mae'],
                                       rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
            np.testing.assert_allclose(development_mae, run['development_replay_mae'],
                                       rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
            scores[prefix] = independent_scores(scored, scored[prefix].to_numpy())
            paired[prefix] = independent_paired(scored, scored.native.to_numpy(),
                                                scored[prefix].to_numpy())
            compare(scores[prefix], run['scores'], f'{year}.{prefix}.scores')
            compare(paired[prefix], run['paired_native'], f'{year}.{prefix}.paired')
            runs[prefix] = {'selected_epoch': best, 'selection_epochs': len(run['selection_trace']),
                            'parameters': parameters, 'development_replay_mae': development_mae,
                            'maximum_absolute_replay_difference': float(np.abs(replay - predictions[prefix]).max())}
    summary = summaries(scores)
    compare(summary, report['summaries'], f'{year}.summaries')
    gate = exact_advantage(native_score, scores, summary)
    assert gate == report['exact_advantage'] and report['year'] == year
    assert report['all_input_hashes_unchanged'] is True and report['deployed'] is False
    repeat_paired = {str(seed): independent_paired(scored, scored[f'lstm_repeat_seed{seed}'],
                                                   scored[f'lstm_history_seed{seed}']) for seed in SEEDS}
    compare(repeat_paired, report['lstm_history_vs_repeat'], f'{year}.history_vs_repeat')
    return {'passed': True, 'training_rows': len(train), 'selection_training_rows': int(fitting.sum()),
            'selection_rows': int(selection.sum()), 'deferred_rows': int(deferred.sum()),
            'prediction_rows': len(predictions), 'scoring_rows': len(scored),
            'normalization': normalization, 'runs': runs, 'native': native_score,
            'scores': scores, 'paired_native': paired, 'history_vs_repeat': repeat_paired,
            'summaries': summary, 'exact_advantage': gate}


def verify_all():
    protocol, old_protocol = read_json(OUT / 'protocol.json'), read_json(OLD / 'protocol.json')
    audit = read_json(OUT / 'upstream_audit.json')
    assert protocol['years'] == list(YEARS) and protocol['seeds'] == list(SEEDS)
    assert protocol['arms'] == list(ARMS) and Path(protocol['old_archive']).resolve() == OLD.resolve()
    assert protocol['features'] == old_protocol['features'] and len(protocol['features']) == 104
    assert protocol['history_features'] == old_protocol['history_features']
    assert len(protocol['history_features']) == 12
    assert (protocol['batch_size'], protocol['scale'], protocol['max_epochs'],
            protocol['patience'], protocol['device']) == (128, 1.2, 60, 8, 'cuda')
    assert not protocol.get('deployment_authorized', False)
    assert protocol['deployed'] is False
    assert torch.cuda.is_available(), 'CUDA replay is mandatory; CPU replay is not a substitute.'
    for old in ('grouped_forecast_aft_2023_20260916', 'direct_history_aft_2023_20260918'):
        check_hashes(read_json(BASE / old / 'protocol.json')['input_sha256'])
    check_hashes(old_protocol['input_sha256'])
    check_hashes(protocol['input_sha256'])
    check_hashes(protocol['deployment_sha256'])
    check_hashes(audit['input_sha256'])
    assert audit['generated_after_training_start'] is True
    assert audit['old_archive_unchanged_during_generation'] is True
    assert Path(audit['old_archive']).resolve() == OLD.resolve()
    assert audit['old_protocol_input_files'] == len(old_protocol['input_sha256'])
    assert audit['expected_sources'].keys() == audit['input_sha256'].keys()
    assert old_protocol['input_sha256'].items() <= audit['input_sha256'].items()
    joint_hashes = dict(protocol['input_sha256'])
    for path, expected in audit['input_sha256'].items():
        assert path not in joint_hashes or joint_hashes[path] == expected, path
        joint_hashes[path] = expected
    archived_verification = read_json(OLD / 'verification.json')
    assert archived_verification['passed'] is True
    for name, expected in archived_verification['artifact_sha256'].items():
        assert audit['input_sha256'][str(OLD / name)] == expected, name
    assert audit['input_sha256'][str(ROOT / 'scripts/verify_direct_history_crossyear.py')] == archived_verification['verifier_sha256']
    old_files = [path for path in OLD.rglob('*') if path.is_file()]
    assert len(old_files) == audit['old_archive_files']
    for path in old_files:
        assert str(path) in joint_hashes, f'Old archive not protected: {path}'
    for year in YEARS:
        old_report = read_json(OLD / str(year) / 'report.json')
        for name, expected in old_report['output_sha256'].items():
            assert digest(OLD / str(year) / name) == expected, name
    before = {str(path.relative_to(OUT)): digest(path) for path in OUT.rglob('*')
              if path.is_file() and path.name != 'verification.json'}
    hourly = pd.read_parquet(BASE / 'observed_business_dataset_v1/features.parquet')
    repair = read_json(BASE / 'ghcnh_weather_repair_v1/report.json')
    weather = pd.read_parquet(BASE / 'ghcnh_weather_repair_v1/quality_filtered_weather.parquet').set_index('timestamp')
    patched = hourly[['timestamp', *protocol['history_features']]].copy()
    counts = {}
    for column in ('temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms'):
        aligned = weather[column].reindex(pd.DatetimeIndex(hourly.timestamp)).to_numpy()
        mask = (hourly.timestamp.gt(pd.Timestamp(repair['patch_after'])).to_numpy()
                & hourly[column].isna().to_numpy() & np.isfinite(aligned))
        patched.loc[mask, column] = aligned[mask]
        counts[column] = int(mask.sum())
    compare(counts, repair['patched_grid_cells'], 'weather_repair')
    compare(counts, old_protocol['patched_cells'], 'frozen_weather_repair')
    pd.testing.assert_frame_equal(patched, pd.read_parquet(OLD / 'hourly_inputs.parquet'))
    canonical = pd.read_parquet(BASE / 'interval_business_labels_v1/origins.parquet')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    results = {str(year): verify_year(year, protocol, hourly if year == 2024 else patched, canonical)
               for year in YEARS}
    assert sum(result['scoring_rows'] for result in results.values()) == 601
    report = read_json(OUT / 'report.json')
    assert set(report['years']) == set(results)
    for year in YEARS:
        compare(read_json(OUT / str(year) / 'report.json'), report['years'][str(year)],
                f'root.years.{year}')
    both = all(result['exact_advantage'] for result in results.values())
    assert report['both_years_exact_advantage'] == both and report['deployed'] is False
    assert report['all_input_hashes_unchanged'] is True
    expected_artifacts = {name: value for name, value in before.items() if name != 'report.json'}
    compare(expected_artifacts, report['output_sha256'], 'root.output_sha256')
    check_hashes(protocol['input_sha256'])
    check_hashes(protocol['deployment_sha256'])
    check_hashes(audit['input_sha256'])
    after = {str(path.relative_to(OUT)): digest(path) for path in OUT.rglob('*')
             if path.is_file() and path.name != 'verification.json'}
    assert before == after, 'Artifacts changed during independent verification.'
    return {'passed': True, 'years': results, 'both_years_exact_advantage': both,
            'deployed': False, 'final_weights_replayed': 18, 'development_weights_replayed': 18,
            'input_hashes_checked': len(protocol['input_sha256']),
            'supplemental_input_hashes_checked': len(audit['input_sha256']),
            'upstream_audit_sha256': digest(OUT / 'upstream_audit.json'),
            'deployment_hashes_checked': len(protocol['deployment_sha256']),
            'artifact_sha256': before, 'verifier_sha256': digest(__file__),
            'replay_tolerance': {'rtol': REPLAY_RTOL, 'atol': REPLAY_ATOL},
            'limitations': ['Retrospective numerical verification is not evidence of real-time availability.',
                            'Supplemental upstream protection started after training launched; it is not original preflight coverage.',
                            'The prior verification.json is protected from the supplemental snapshot, not by an original self-hash.',
                            'Epoch traces and selected checkpoints are checked; every optimizer update is not independently retrained.']}


def main():
    needed = [OUT / 'protocol.json', OUT / 'report.json',
              *(OUT / str(year) / 'report.json' for year in YEARS)]
    pending = [str(path) for path in needed if not path.exists()]
    if pending:
        print(json.dumps({'status': 'waiting_for_reports', 'pending': pending}))
        return
    try:
        result = verify_all()
    except Exception as exc:  # noqa: BLE001 - Persist diagnostics and still exit with failure.
        result = {'passed': False, 'error_type': type(exc).__name__, 'error': str(exc),
                  'traceback': traceback.format_exc(), 'verifier_sha256': digest(__file__)}
    destination = OUT / 'verification.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'output': str(destination),
                      **({'error': result['error']} if not result['passed'] else {})}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
