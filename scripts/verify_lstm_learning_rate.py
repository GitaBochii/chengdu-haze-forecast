"""Independently replay the fixed learning-rate search without retraining."""

import json
import math
import os
import traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.verify_lstm_direct_crossyear import (
    REPLAY_ATOL,
    REPLAY_RTOL,
    SUMMARY_METRICS,
    check_hashes,
    compare,
    digest,
    independent_paired,
    independent_scores,
    read_json,
    replay_weights,
    standardize,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/task01_recovery_head'
OUT = BASE / 'lstm_lr_tuning_20261012'
PREVIOUS = BASE / 'lstm_direct_crossyear_20261012'
OLDER = BASE / 'direct_history_crossyear_20260918'
YEARS = (2024, 2025)
SEEDS = (24, 42, 2026)
LEARNING_RATES = (.0003, .001, .003)
ARM = 'lstm_history'
LABEL_COLUMNS = (
    'lower_bound_hours', 'upper_bound_hours', 'exact_hours',
    'label_available_at', 'continuity_block',
)


def learning_rate_id(value):
    return 'lr_' + str(value).replace('.', 'p')


def verify_trace(trace, selected_epoch, refit_trace=None):
    assert isinstance(trace, list) and 1 <= len(trace) <= 60
    assert [row['epoch'] for row in trace] == list(range(1, len(trace) + 1))
    assert all(np.isfinite([row['training_nll'], row['selection_block_mae']]).all()
               for row in trace)
    assert all(row['selection_block_mae'] >= 0 for row in trace)
    best_at, best_value = 0, math.inf
    for row in trace:
        if row['selection_block_mae'] < best_value:
            best_at, best_value = row['epoch'], row['selection_block_mae']
        if row['epoch'] < len(trace):
            assert row['epoch'] - best_at < 8, 'Trace continued past patience limit'
    assert selected_epoch == best_at, 'Selected epoch must be earliest optimum'
    assert len(trace) == 60 or len(trace) - best_at == 8
    if refit_trace is not None:
        assert [row['epoch'] for row in refit_trace] == list(range(1, best_at + 1))
        assert all(np.isfinite(row['training_nll']) for row in refit_trace)
    return {'selected_epoch': best_at, 'selection_epochs': len(trace),
            'best_trace_mae': best_value, 'reached_epoch_budget': len(trace) == 60,
            'best_at_budget_boundary': best_at == 60}


def verify_state_equal(candidate, previous):
    left = torch.load(candidate, weights_only=True, map_location='cpu')
    right = torch.load(previous, weights_only=True, map_location='cpu')
    assert left.keys() == right.keys()
    for key in left:
        assert left[key].dtype == right[key].dtype and left[key].shape == right[key].shape
        assert torch.equal(left[key], right[key]), f'Original-rate checkpoint differs: {key}'


def metric_summary(scores):
    result = {}
    for metric in SUMMARY_METRICS:
        values = np.asarray([scores[str(seed)][metric] for seed in SEEDS], dtype=float)
        assert np.isfinite(values).all(), metric
        result[metric] = {'mean': float(values.mean()), 'min': float(values.min()),
                          'max': float(values.max())}
    return result


def advantage_gate(native, scores, summary):
    metrics = ('exact_row_mae', 'exact_block_mae')
    improvements = sum(all(scores[str(seed)][key] < native[key] for key in metrics)
                       for seed in SEEDS)
    return bool(all(summary[key]['mean'] < native[key] for key in metrics)
                and summary['tail24_block_mae']['mean'] <= native['tail24_block_mae']
                and improvements >= 2)


def normalization_arrays(folder, history, snapshot, fitting, full):
    arrays, result = {}, {}
    expected_keys = {key + suffix for key in ('history', 'snapshot')
                     for suffix in ('', '_mean', '_scale', '_supported')}
    for stage, members in [('selection', fitting), ('final', full)]:
        name = 'selection_history_inputs.npz' if stage == 'selection' else 'history_inputs.npz'
        with np.load(folder / name, allow_pickle=False) as saved:
            actual = {key: saved[key] for key in saved.files}
        assert actual.keys() == expected_keys
        result[stage] = {}
        for key, raw in [('history', history), ('snapshot', snapshot)]:
            data, mean, scale, supported = standardize(raw, members)
            assert actual[key].dtype == np.float32
            assert actual[key + '_mean'].dtype == np.float64
            assert actual[key + '_scale'].dtype == np.float64
            assert actual[key + '_supported'].dtype == np.bool_
            np.testing.assert_allclose(actual[key + '_mean'], mean, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(actual[key + '_scale'], scale, rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(actual[key + '_supported'], supported)
            np.testing.assert_allclose(actual[key], data, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(actual[key][..., raw.shape[-1]:], np.isnan(raw))
            assert np.isfinite(actual[key]).all()
            result[stage][key] = {'fitting_rows': int(members.sum()),
                                  'supported_features': int(supported.sum())}
        arrays[stage] = actual
    return arrays, result


def source_year(year, protocol, canonical, hourly):
    folder = PREVIOUS / str(year)
    training = pd.read_parquet(folder / 'training.parquet')
    development = pd.read_parquet(folder / 'selection_rows.parquet')
    fitting_rows = pd.read_parquet(folder / 'selection_training.parquet')
    old_predictions = pd.read_parquet(folder / 'all_predictions.parquet')
    old_scored = pd.read_parquet(folder / 'scored_predictions.parquet')
    for name, actual in [('training', training), ('selection_rows', development),
                         ('selection_training', fitting_rows)]:
        pd.testing.assert_frame_equal(actual, pd.read_parquet(OLDER / str(year) / f'{name}.parquet'))
    original_training = pd.read_parquet(
        BASE / f'original_gefs_without_pbl_crossyear_v1/{year}_training_rows.parquet')
    pd.testing.assert_frame_equal(training, original_training)
    cohort = canonical[canonical.timestamp.dt.year.eq(year)].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_scored[canonical.columns].reset_index(drop=True), cohort)
    assert training.timestamp.is_unique and old_predictions.timestamp.is_unique
    assert old_scored.timestamp.is_unique
    assert set(old_predictions.timestamp) == set(cohort.timestamp)
    assert not set(training.continuity_block) & set(cohort.continuity_block)
    lookup = canonical.set_index('timestamp')
    for column in LABEL_COLUMNS:
        np.testing.assert_array_equal(training[column], lookup.loc[training.timestamp, column])
    cutoff = pd.Timestamp(f'{year - 1}-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
    assert training.timestamp.lt(end).all() and training.label_available_at.lt(end).all()
    assert cohort.label_available_at.lt(end + pd.DateOffset(years=1)).all()
    selection = training.timestamp.ge(cutoff) & training.label_available_at.lt(end)
    fitting = (training.timestamp.lt(cutoff) & training.label_available_at.lt(cutoff)
               & ~training.continuity_block.isin(training.loc[selection, 'continuity_block']))
    pd.testing.assert_frame_equal(development, training.loc[selection].reset_index(drop=True))
    pd.testing.assert_frame_equal(fitting_rows, training.loc[fitting].reset_index(drop=True))
    assert not set(fitting_rows.continuity_block) & set(development.continuity_block)
    assert fitting_rows.label_available_at.lt(cutoff).all()
    expected = {2024: (6770, 396, 7166, 335, 222, 335),
                2025: (7166, 335, 7624, 266, 335, 243)}[year]
    assert (int(fitting.sum()), int(selection.sum()), len(training), len(cohort),
            int(development.exact_hours.notna().sum()), int(cohort.exact_hours.notna().sum())) == expected
    deferred = ~(fitting | selection)
    assert int(deferred.sum()) == (123 if year == 2025 else 0)
    if deferred.any():
        assert training.loc[deferred].timestamp.dt.year.eq(2023).all()
        assert training.loc[deferred].label_available_at.ge(cutoff).all()
    with np.load(folder / 'raw_inputs.npz', allow_pickle=False) as saved:
        assert set(saved.files) == {'history', 'snapshot', 'timestamps'}
        raw = {name: saved[name] for name in saved.files}
    times = pd.DatetimeIndex(pd.concat([training.timestamp, old_predictions.timestamp], ignore_index=True))
    np.testing.assert_array_equal(times.asi8, raw['timestamps'])
    source = hourly.set_index('timestamp')
    assert source.index.is_unique and source.index.is_monotonic_increasing
    history = np.stack([source.loc[pd.to_datetime(times.asi8 - lag * 3_600_000_000_000, utc=True)
                                  .tz_convert('Asia/Shanghai'), protocol['history_features']].to_numpy(float)
                        for lag in range(23, -1, -1)], axis=1)
    np.testing.assert_array_equal(history, raw['history'])
    features = pd.read_parquet(BASE / f'reusable_gefs_crossyear_v1/{year}_features.parquet')
    snapshot = np.concatenate([training[protocol['features']].to_numpy(float),
                               features[protocol['features']].to_numpy(float)])
    np.testing.assert_array_equal(snapshot, raw['snapshot'])
    assert history.shape == (len(times), 24, 12) and snapshot.shape == (len(times), 104)
    np.testing.assert_array_equal(history[:, -1], snapshot[:, [protocol['features'].index(name)
                                                            for name in protocol['history_features']]])
    native = joblib.load(BASE / f'reusable_gefs_crossyear_v1/{year}.joblib').predict(
        features[protocol['features']])
    np.testing.assert_array_equal(native, old_predictions.native)
    original_predictions = pd.read_parquet(BASE / f'reusable_gefs_crossyear_v1/{year}_predictions.parquet')
    np.testing.assert_array_equal(native, original_predictions.forecast_aft)
    np.testing.assert_array_equal(old_predictions.timestamp, original_predictions.timestamp)
    full_members = np.arange(len(times)) < len(training)
    selection_members = np.r_[fitting.to_numpy(), np.zeros(len(old_predictions), dtype=bool)]
    arrays, normalization = normalization_arrays(folder, history, snapshot,
                                                 selection_members, full_members)
    return {'training': training, 'development': development, 'cohort': cohort,
            'old_predictions': old_predictions, 'old_scored': old_scored,
            'selection_indices': np.flatnonzero(selection.to_numpy()),
            'final_indices': np.arange(len(training), len(times)), 'arrays': arrays,
            'normalization': normalization, 'fitting_rows': int(fitting.sum()),
            'deferred_rows': int(deferred.sum())}


def verify_candidates(source, canonical):
    folder = OUT / 'candidates'
    report = read_json(folder / 'report.json')
    predictions = pd.read_parquet(folder / 'development_predictions.parquet')
    prefixes = [f'{learning_rate_id(rate)}_seed{seed}' for rate in LEARNING_RATES for seed in SEEDS]
    assert set(report['runs']) == set(prefixes)
    assert predictions.columns.is_unique and predictions.timestamp.is_unique
    development = source['development']
    assert list(predictions.columns) == [*development.columns, *prefixes]
    pd.testing.assert_frame_equal(predictions[development.columns], development)
    expected = canonical.set_index('timestamp', drop=False).loc[development.timestamp].reset_index(drop=True)
    pd.testing.assert_frame_equal(predictions[canonical.columns], expected)
    assert len(predictions) == 396 and predictions.timestamp.dt.year.eq(2023).all()
    assert predictions.exact_hours.notna().sum() == 222
    scores, checks = {}, {}
    for rate in LEARNING_RATES:
        for seed in SEEDS:
            prefix = f'{learning_rate_id(rate)}_seed{seed}'
            run = read_json(folder / f'{prefix}_run.json')
            compare(run, report['runs'][prefix], f'candidate.{prefix}.stored_run')
            assert (run['seed'], run['learning_rate'], run['lr_id']) == (seed, rate, learning_rate_id(rate))
            selection = verify_trace(run['selection_trace'], run['selected_epoch'])
            assert run['budget_reached'] == selection['reached_epoch_budget']
            replay, parameters = replay_weights(folder / f'{prefix}.pt', ARM,
                                                 source['arrays']['selection'],
                                                 source['selection_indices'])
            np.testing.assert_allclose(replay, predictions[prefix], rtol=REPLAY_RTOL,
                                       atol=REPLAY_ATOL, err_msg=f'candidate.{prefix}.replay')
            assert parameters == run['parameters'] == 10177
            score = independent_scores(predictions, predictions[prefix].to_numpy())
            compare(score, run['scores'], f'candidate.{prefix}.scores')
            replay_mae = independent_scores(predictions, replay)['exact_block_mae']
            np.testing.assert_allclose(replay_mae, run['development_replay_mae'],
                                       rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
            np.testing.assert_allclose(replay_mae, selection['best_trace_mae'],
                                       rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
            if rate == .001:
                assert run['original_configuration_reproduced'] is True
                old_prefix = f'lstm_history_seed{seed}'
                old_run = read_json(PREVIOUS / '2024' / f'{old_prefix}_run.json')
                compare(run['selection_trace'], old_run['selection_trace'],
                        f'candidate.{prefix}.original_trace')
                assert run['selected_epoch'] == old_run['selected_epoch']
                compare(run['development_replay_mae'], old_run['development_replay_mae'],
                        f'candidate.{prefix}.original_development_mae')
                verify_state_equal(folder / f'{prefix}.pt',
                                   PREVIOUS / '2024' / f'{old_prefix}_selection.pt')
            scores[prefix] = score
            checks[prefix] = {**selection, 'scores': score, 'parameters': parameters,
                              'development_replay_mae': replay_mae,
                              'original_rate_checkpoint_reproduced': rate == .001,
                              'maximum_absolute_replay_difference': float(np.abs(replay - predictions[prefix]).max())}
    means = {learning_rate_id(rate): float(np.mean([scores[f'{learning_rate_id(rate)}_seed{seed}']['exact_block_mae']
                                                  for seed in SEEDS])) for rate in LEARNING_RATES}
    selected = min(LEARNING_RATES, key=lambda rate: (means[learning_rate_id(rate)],
                                                   abs(math.log(rate / .001)), rate))
    decision = read_json(OUT / 'decision.json')
    assert decision['metric'] == 'exact_block_mae'
    assert decision['learning_rates'] == list(LEARNING_RATES) and decision['seeds'] == list(SEEDS)
    assert decision['selected_learning_rate'] == selected
    assert decision['selected_lr_id'] == learning_rate_id(selected)
    assert decision['before_evaluation'] is True
    assert decision['retrospective_exploration'] is True
    compare(means, decision['development_mean_scores'], 'decision.development_mean_scores')
    epochs = {str(seed): report['runs'][f'{learning_rate_id(selected)}_seed{seed}']['selected_epoch']
              for seed in SEEDS}
    compare(epochs, decision['selected_epochs'], 'decision.selected_epochs')
    return decision, {'passed': True, 'runs': checks, 'development_mean_scores': means,
                       'selected_learning_rate': selected}, report['runs']


def verify_year(year, source, decision, candidates, canonical):
    folder = OUT / str(year)
    report = read_json(folder / 'report.json')
    predictions = pd.read_parquet(folder / 'all_predictions.parquet')
    scored = pd.read_parquet(folder / 'scored_predictions.parquet')
    tuned = [f'lstm_tuned_seed{seed}' for seed in SEEDS]
    original = [f'original_lstm_seed{seed}' for seed in SEEDS]
    assert list(predictions.columns) == ['timestamp', 'native', *original, *tuned]
    assert predictions.timestamp.is_unique and scored.timestamp.is_unique and scored.columns.is_unique
    assert set(scored.columns) == {*canonical.columns, 'native', *original, *tuned}
    np.testing.assert_array_equal(predictions.timestamp, source['old_predictions'].timestamp)
    pd.testing.assert_frame_equal(scored[canonical.columns].reset_index(drop=True), source['cohort'])
    assert set(predictions.timestamp) == set(scored.timestamp)
    np.testing.assert_array_equal(predictions.native, source['old_predictions'].native)
    for seed in SEEDS:
        np.testing.assert_array_equal(predictions[f'original_lstm_seed{seed}'],
                                      source['old_predictions'][f'lstm_history_seed{seed}'])
    for column in ('native', *original, *tuned):
        np.testing.assert_array_equal(predictions.set_index('timestamp').loc[scored.timestamp, column], scored[column])
    native = independent_scores(scored, scored.native.to_numpy())
    compare(native, report['native'], f'{year}.native')
    scores, old_scores, checks, paired_native, paired_original = {}, {}, {}, {}, {}
    assert set(report['runs']) == set(tuned)
    assert set(report['original_lstm']) == {str(seed) for seed in SEEDS}
    for seed in SEEDS:
        prefix = f'lstm_tuned_seed{seed}'
        run = read_json(folder / f'{prefix}_run.json')
        compare(run, report['runs'][prefix], f'{year}.{prefix}.stored_run')
        assert (run['year'], run['seed'], run['learning_rate']) == (year, seed, decision['selected_learning_rate'])
        selection = verify_trace(run['selection_trace'], run['selected_epoch'], run['refit_trace'])
        if year == 2024:
            candidate = f"{decision['selected_lr_id']}_seed{seed}"
            assert run['selection_source'] == f'candidates/{candidate}.pt'
            compare(run['selection_trace'], candidates[candidate]['selection_trace'],
                    f'{year}.{prefix}.candidate_trace')
            assert run['selected_epoch'] == decision['selected_epochs'][str(seed)]
            compare(run['development_replay_mae'], candidates[candidate]['development_replay_mae'],
                    f'{year}.{prefix}.candidate_mae')
            development_mae = candidates[candidate]['development_replay_mae']
        else:
            assert run['selection_source'] == f'{year}/{prefix}_selection.pt'
            development_prediction, _ = replay_weights(folder / f'{prefix}_selection.pt', ARM,
                                                        source['arrays']['selection'],
                                                        source['selection_indices'])
            development_mae = independent_scores(source['development'], development_prediction)['exact_block_mae']
            np.testing.assert_allclose(development_mae, run['development_replay_mae'],
                                       rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
            np.testing.assert_allclose(development_mae, selection['best_trace_mae'],
                                       rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
        replay, parameters = replay_weights(folder / f'{prefix}.pt', ARM,
                                             source['arrays']['final'], source['final_indices'],
                                             preserve_slice_layout=True)
        np.testing.assert_allclose(replay, predictions[prefix], rtol=REPLAY_RTOL,
                                   atol=REPLAY_ATOL, err_msg=f'{year}.{prefix}.final_replay')
        assert parameters == run['parameters'] == 10177
        score = independent_scores(scored, scored[prefix].to_numpy())
        old_score = independent_scores(scored, scored[f'original_lstm_seed{seed}'].to_numpy())
        paired_n = independent_paired(scored, scored.native.to_numpy(), scored[prefix].to_numpy())
        paired_o = independent_paired(scored, scored[f'original_lstm_seed{seed}'].to_numpy(), scored[prefix].to_numpy())
        compare(score, run['scores'], f'{year}.{prefix}.scores')
        compare(old_score, report['original_lstm'][str(seed)], f'{year}.{prefix}.original_scores')
        compare(paired_n, run['paired_native'], f'{year}.{prefix}.paired_native')
        compare(paired_o, run['paired_original_lstm'], f'{year}.{prefix}.paired_original_lstm')
        if decision['selected_learning_rate'] == .001:
            assert run['original_configuration_reproduced'] is True
            old_prefix = f'lstm_history_seed{seed}'
            old_run = read_json(PREVIOUS / str(year) / f'{old_prefix}_run.json')
            compare(run['selection_trace'], old_run['selection_trace'], f'{year}.{prefix}.original_trace')
            compare(run['refit_trace'], old_run['refit_trace'], f'{year}.{prefix}.original_refit')
            verify_state_equal(folder / f'{prefix}.pt', PREVIOUS / str(year) / f'{old_prefix}.pt')
            np.testing.assert_array_equal(predictions[prefix], predictions[f'original_lstm_seed{seed}'])
            if year == 2025:
                verify_state_equal(folder / f'{prefix}_selection.pt',
                                   PREVIOUS / str(year) / f'{old_prefix}_selection.pt')
        scores[str(seed)], old_scores[str(seed)] = score, old_score
        paired_native[str(seed)], paired_original[str(seed)] = paired_n, paired_o
        checks[prefix] = {**selection, 'parameters': parameters, 'development_replay_mae': development_mae,
                          'maximum_absolute_replay_difference': float(np.abs(replay - predictions[prefix]).max())}
    summary, original_summary = metric_summary(scores), metric_summary(old_scores)
    compare(summary, report['summary'], f'{year}.summary')
    compare(original_summary, report['original_summary'], f'{year}.original_summary')
    gate = advantage_gate(native, scores, summary)
    assert report['year'] == year and report['exact_advantage'] == gate and report['deployed'] is False
    return {'passed': True, 'training_rows': len(source['training']),
            'selection_training_rows': source['fitting_rows'], 'selection_rows': len(source['development']),
            'deferred_rows': source['deferred_rows'], 'scoring_rows': len(scored),
            'normalization': source['normalization'], 'runs': checks, 'native': native,
            'scores': scores, 'original_lstm': old_scores, 'summary': summary,
            'original_summary': original_summary, 'paired_native': paired_native,
            'paired_original_lstm': paired_original, 'exact_advantage': gate}


def verify_protocol(protocol):
    previous_protocol = read_json(PREVIOUS / 'protocol.json')
    previous_verification = read_json(PREVIOUS / 'verification.json')
    assert previous_verification['passed'] is True
    assert Path(protocol['previous_archive']).resolve() == PREVIOUS.resolve()
    assert Path(protocol['old_archive']).resolve() == OLDER.resolve()
    assert protocol['years'] == list(YEARS) and protocol['seeds'] == list(SEEDS)
    assert protocol['lr_values'] == list(LEARNING_RATES)
    assert len(previous_protocol['features']) == 104
    assert len(previous_protocol['history_features']) == 12
    assert protocol['features'] == previous_protocol['features']
    assert protocol['history_features'] == previous_protocol['history_features']
    older_protocol = read_json(OLDER / 'protocol.json')
    assert previous_protocol['features'] == older_protocol['features']
    assert previous_protocol['history_features'] == older_protocol['history_features']
    assert (protocol['batch_size'], protocol['scale'], protocol['max_epochs'],
            protocol['patience'], protocol['device']) == (128, 1.2, 60, 8, 'cuda')
    assert protocol['weight_decay'] == .01 and protocol['gradient_clip'] == 5.
    assert protocol['parameter_count'] == 10177
    assert protocol['retrospective_exploration'] is True
    assert protocol['extra_development_budget'] is True
    assert protocol['deployed'] is False and not protocol.get('deployment_authorized', False)
    hashes = protocol['input_sha256']
    check_hashes(hashes)
    check_hashes(protocol['deployment_sha256'])
    check_hashes(previous_protocol['input_sha256'])
    check_hashes(previous_protocol['deployment_sha256'])
    previous_audit = read_json(PREVIOUS / 'upstream_audit.json')
    check_hashes(previous_audit['input_sha256'])
    for upstream in (previous_protocol['input_sha256'], previous_audit['input_sha256']):
        assert upstream.items() <= hashes.items(), 'Frozen upstream source omitted from preflight'
    assert protocol['deployment_sha256'] == previous_protocol['deployment_sha256']
    for archive in (PREVIOUS, OLDER):
        for path in archive.rglob('*'):
            if path.is_file():
                assert str(path) in hashes and digest(path) == hashes[str(path)], f'Unprotected archive: {path}'
    for name, expected in previous_verification['artifact_sha256'].items():
        assert hashes[str(PREVIOUS / name)] == expected, name
    assert hashes[str(ROOT / 'scripts/verify_lstm_direct_crossyear.py')] == previous_verification['verifier_sha256']
    for relative in ('scripts/tune_lstm_learning_rate.py',
                     'scripts/lstm_lr_training.py', 'docs/superpowers/specs/2026-10-12-lstm-learning-rate-design.md'):
        assert str(ROOT / relative) in hashes, f'New source omitted from preflight: {relative}'
    return previous_protocol


def verify_all():
    assert torch.cuda.is_available(), 'CUDA replay is mandatory; CPU is not an acceptable substitute.'
    protocol = read_json(OUT / 'protocol.json')
    previous_protocol = verify_protocol(protocol)
    before = {str(path.relative_to(OUT)): digest(path) for path in OUT.rglob('*')
              if path.is_file() and path.name != 'verification.json'}
    decision_before = digest(OUT / 'decision.json')
    hourly = pd.read_parquet(BASE / 'observed_business_dataset_v1/features.parquet')
    repair = read_json(BASE / 'ghcnh_weather_repair_v1/report.json')
    weather = pd.read_parquet(BASE / 'ghcnh_weather_repair_v1/quality_filtered_weather.parquet').set_index('timestamp')
    patched = hourly[['timestamp', *previous_protocol['history_features']]].copy()
    counts = {}
    for column in ('temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms'):
        aligned = weather[column].reindex(pd.DatetimeIndex(hourly.timestamp)).to_numpy()
        mask = (hourly.timestamp.gt(pd.Timestamp(repair['patch_after'])).to_numpy()
                & hourly[column].isna().to_numpy() & np.isfinite(aligned))
        patched.loc[mask, column] = aligned[mask]
        counts[column] = int(mask.sum())
    compare(counts, repair['patched_grid_cells'], 'weather_repair')
    older_protocol = read_json(OLDER / 'protocol.json')
    compare(counts, older_protocol['patched_cells'], 'frozen_weather_repair')
    pd.testing.assert_frame_equal(patched, pd.read_parquet(OLDER / 'hourly_inputs.parquet'))
    canonical = pd.read_parquet(BASE / 'interval_business_labels_v1/origins.parquet')
    assert canonical.timestamp.is_unique
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source_2024 = source_year(2024, previous_protocol, canonical, hourly)
    decision, candidates, candidate_runs = verify_candidates(source_2024, canonical)
    compare(decision['source_sha256'], protocol['input_sha256'], 'decision.source_sha256')
    assert decision['protocol_sha256'] == digest(OUT / 'protocol.json')
    results = {'2024': verify_year(2024, source_2024, decision, candidate_runs, canonical)}
    del source_2024
    source_2025 = source_year(2025, previous_protocol, canonical, patched)
    results['2025'] = verify_year(2025, source_2025, decision, candidate_runs, canonical)
    assert sum(result['scoring_rows'] for result in results.values()) == 601
    report = read_json(OUT / 'report.json')
    assert set(report['years']) == set(results)
    for year in YEARS:
        compare(read_json(OUT / str(year) / 'report.json'), report['years'][str(year)], f'root.years.{year}')
    both = all(result['exact_advantage'] for result in results.values())
    assert report['both_years_exact_advantage'] == both
    assert report['selected_learning_rate'] == decision['selected_learning_rate']
    assert report['deployed'] is False and report['input_hashes_unchanged'] is True
    assert report['decision_sha256'] == decision_before
    assert decision['before_evaluation'] is True
    expected_weights = {
        *(f'candidates/{learning_rate_id(rate)}_seed{seed}.pt'
          for rate in LEARNING_RATES for seed in SEEDS),
        *(f'2025/lstm_tuned_seed{seed}_selection.pt' for seed in SEEDS),
        *(f'{year}/lstm_tuned_seed{seed}.pt' for year in YEARS for seed in SEEDS),
    }
    assert {name.replace('\\', '/') for name in before if name.endswith('.pt')} == expected_weights
    compare({name: value for name, value in before.items() if name != 'report.json'},
            report['output_sha256'], 'root.output_sha256')
    check_hashes(protocol['input_sha256'])
    check_hashes(protocol['deployment_sha256'])
    check_hashes(previous_protocol['input_sha256'])
    assert digest(OUT / 'decision.json') == decision_before, 'Selection changed during verification'
    after = {str(path.relative_to(OUT)): digest(path) for path in OUT.rglob('*')
             if path.is_file() and path.name != 'verification.json'}
    assert before == after, 'Artifacts changed during independent verification'
    return {'passed': True, 'candidates': candidates, 'decision': decision, 'decision_sha256': decision_before,
            'years': results, 'both_years_exact_advantage': both, 'deployed': False,
            'development_weights_replayed': 12, 'final_weights_replayed': 6,
            'scoring_rows': 601, 'input_hashes_checked': len(protocol['input_sha256']),
            'deployment_hashes_checked': len(protocol['deployment_sha256']),
            'artifact_sha256': before, 'verifier_sha256': digest(__file__),
            'replay_tolerance': {'rtol': REPLAY_RTOL, 'atol': REPLAY_ATOL},
            'retrospective_exploration': True, 'extra_development_budget': True,
            'limitations': ['This additional search is retrospective exploratory offline validation, not a new blind test.',
                            'The tuned LSTM receives extra development search budget; it is not an equal-compute architecture comparison.',
                            'Pre-evaluation decision ordering is recorded by the protocol, not an externally immutable timestamp.',
                            'Checkpoints and epoch traces are checked; every optimizer update is not independently retrained.',
                            'Historical numerical replay does not demonstrate real-time data availability or deployment performance.']}


def main():
    needed = [OUT / 'protocol.json', OUT / 'decision.json', OUT / 'report.json',
              OUT / 'candidates/report.json', *(OUT / str(year) / 'report.json' for year in YEARS)]
    pending = [str(path) for path in needed if not path.exists()]
    if pending:
        print(json.dumps({'status': 'waiting_for_reports', 'pending': pending}))
        return
    try:
        result = verify_all()
    except Exception as exc:  # noqa: BLE001 - Persist diagnostics and retain a failing exit code.
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
