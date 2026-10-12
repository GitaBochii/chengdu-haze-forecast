"""Read-only independent annual reconstruction; write only verification.json."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/task01_recovery_head'
OUT = BASE / 'direct_history_crossyear_20260918'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual, expected):
    assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
    for key, value in actual.items():
        if isinstance(value, dict):
            compare(value, expected[key])
        elif value is None or isinstance(value, (bool, str)):
            assert value == expected[key], (key, value, expected[key])
        else:
            np.testing.assert_allclose(value, expected[key], rtol=1e-12, atol=1e-10, err_msg=key)


def average(values, mask, blocks=None):
    selected = np.asarray(values)[mask]
    if not len(selected):
        return None
    if blocks is None:
        return float(selected.mean())
    return float(pd.DataFrame({'v': selected, 'b': blocks[mask]}).groupby('b').v.mean().mean())


def independent_scores(frame, prediction):
    lo, hi = frame.lower_bound_hours.to_numpy(), frame.upper_bound_hours.to_numpy()
    exact = frame.exact_hours.notna().to_numpy()
    blocks = frame.continuity_block.to_numpy()
    assert np.isfinite(prediction).all() and (prediction > 0).all()
    assert np.array_equal(exact, lo == hi)
    np.testing.assert_array_equal(frame.exact_hours.to_numpy()[exact], lo[exact])
    minimum = np.array([0. if l <= p <= u else min(abs(p-l), abs(p-u))
                        for p, l, u in zip(prediction, lo, hi, strict=True)])
    maximum = np.array([max(abs(p-l), abs(p-u))
                        for p, l, u in zip(prediction, lo, hi, strict=True)])
    all_rows, uncertain, tail = np.ones(len(frame), bool), ~exact, exact & (lo >= 24)
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
    bounds = []
    for a, b, lo, hi in zip(baseline, candidate, frame.lower_bound_hours,
                           frame.upper_bound_hours, strict=True):
        points = [lo, max(lo, min(a, hi)), max(lo, min(b, hi))]
        differences = [abs(b-y) - abs(a-y) for y in points]
        differences.append(abs(b-hi) - abs(a-hi) if np.isfinite(hi) else a-b)
        bounds.append((min(differences), max(differences)))
    lower, upper = np.asarray(bounds).T
    blocks, exact = frame.continuity_block.to_numpy(), frame.exact_hours.notna().to_numpy()
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
    fitting = values[members].reshape(-1, values.shape[-1])
    means, scales, support = [], [], []
    for column in fitting.T:
        assert not np.isinf(column).any()
        observed = column[~np.isnan(column)]
        means.append(observed.mean() if len(observed) else 0.)
        std = observed.std(ddof=0) if len(observed) else 1.
        scales.append(std if std > 1e-6 else 1.)
        support.append(bool(len(observed)))
    assert not np.isinf(values).any()
    numeric = (values - means) / scales
    numeric = np.where(np.isfinite(values) & np.asarray(support), numeric, 0.)
    transformed = np.concatenate((numeric, np.isnan(values).astype(float)), axis=-1).astype(np.float32)
    return transformed, np.array(means), np.array(scales), np.array(support)


class IndependentNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.history = nn.GRU(24, 16, batch_first=True)
        self.snapshot = nn.Sequential(nn.Linear(208, 32), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(48, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, history, snapshot):
        _, hidden = self.history(history)
        return self.output(torch.cat((hidden[-1], self.snapshot(snapshot)), dim=1)).squeeze(1)


def validate_selection(selection, fitting_trace):
    trace = selection['trace']
    assert 1 <= len(trace) <= 60
    assert [r['epoch'] for r in trace] == list(range(1, len(trace)+1))
    assert all(np.isfinite([r['training_nll'], r['selection_block_mae']]).all() for r in trace)
    best = min(trace, key=lambda r: r['selection_block_mae'])['epoch']
    assert best == selection['selected_epoch'] == fitting_trace['epochs']
    assert [r['epoch'] for r in fitting_trace['trace']] == list(range(1, best+1))
    assert all(np.isfinite(r['training_nll']) for r in fitting_trace['trace'])
    best_at, best_value = 0, float('inf')
    for row in trace:
        if row['selection_block_mae'] < best_value:
            best_at, best_value = row['epoch'], row['selection_block_mae']
        if row['epoch'] < len(trace):
            assert row['epoch'] - best_at < 8
    assert len(trace) == 60 or len(trace) - best_at == 8
    return best


def verify_year(year, protocol, hourly, canonical):
    folder = OUT / str(year)
    report = json.loads((folder / 'report.json').read_bytes())
    train = pd.read_parquet(folder / 'training.parquet')
    original_train = pd.read_parquet(BASE / f'original_gefs_without_pbl_crossyear_v1/{year}_training_rows.parquet')
    pd.testing.assert_frame_equal(train, original_train)
    features = pd.read_parquet(BASE / f'reusable_gefs_crossyear_v1/{year}_features.parquet')
    original_predictions = pd.read_parquet(BASE / f'reusable_gefs_crossyear_v1/{year}_predictions.parquet')
    predictions = pd.read_parquet(folder / 'all_predictions.parquet')
    scored = pd.read_parquet(folder / 'scored_predictions.parquet')
    np.testing.assert_array_equal(predictions.timestamp, original_predictions.timestamp)
    assert predictions.timestamp.is_unique and train.timestamp.is_unique and scored.timestamp.is_unique
    expected_cohort = canonical[canonical.timestamp.dt.year.eq(year)]
    pd.testing.assert_frame_equal(scored[canonical.columns].reset_index(drop=True),
                                  expected_cohort.reset_index(drop=True))
    assert set(predictions.timestamp) == set(scored.timestamp)
    for column in ('lower_bound_hours', 'upper_bound_hours', 'exact_hours', 'label_available_at', 'continuity_block'):
        np.testing.assert_array_equal(train[column], canonical.set_index('timestamp').loc[train.timestamp, column])
    cut = pd.Timestamp(f'{year-1}-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(end).all() and train.label_available_at.lt(end).all()
    assert scored.label_available_at.lt(end + pd.DateOffset(years=1)).all()
    assert not set(train.continuity_block) & set(scored.continuity_block)
    selection = train.timestamp.ge(cut) & train.label_available_at.lt(end)
    fitting = (train.timestamp.lt(cut) & train.label_available_at.lt(cut)
               & ~train.continuity_block.isin(train.loc[selection, 'continuity_block']))
    pd.testing.assert_frame_equal(pd.read_parquet(folder / 'selection_rows.parquet'), train.loc[selection].reset_index(drop=True))
    pd.testing.assert_frame_equal(pd.read_parquet(folder / 'selection_training.parquet'), train.loc[fitting].reset_index(drop=True))
    expected = {2024: (6770, 396, 7166, 335, 222, 335), 2025: (7166, 335, 7624, 266, 335, 243)}[year]
    assert (fitting.sum(), selection.sum(), len(train), len(scored),
            train.loc[selection, 'exact_hours'].notna().sum(), scored.exact_hours.notna().sum()) == expected
    deferred = ~(fitting | selection)
    assert deferred.sum() == (123 if year == 2025 else 0)
    if deferred.any():
        assert train.loc[deferred].timestamp.dt.year.eq(2023).all()
        assert train.loc[deferred].label_available_at.ge(cut).all()

    raw = np.load(folder / 'raw_inputs.npz')
    times = pd.DatetimeIndex(pd.concat([train.timestamp, predictions.timestamp], ignore_index=True))
    np.testing.assert_array_equal(times.asi8, raw['timestamps'])
    source = hourly.set_index('timestamp')
    assert source.index.is_unique and source.index.is_monotonic_increasing
    history = np.stack([source.loc[pd.to_datetime(times.asi8 - lag * 3_600_000_000_000, utc=True)
                                  .tz_convert('Asia/Shanghai'), protocol['history_features']].to_numpy(float)
                        for lag in range(23, -1, -1)], axis=1)
    np.testing.assert_array_equal(raw['history'], history)
    snapshot = np.concatenate([train[protocol['features']].to_numpy(float), features[protocol['features']].to_numpy(float)])
    np.testing.assert_array_equal(raw['snapshot'], snapshot)
    current = np.concatenate([train[protocol['history_features']].to_numpy(float),
                              features[protocol['history_features']].to_numpy(float)])
    np.testing.assert_array_equal(history[:, -1], current)
    native = joblib.load(BASE / f'reusable_gefs_crossyear_v1/{year}.joblib').predict(features[protocol['features']])
    np.testing.assert_array_equal(native, original_predictions.forecast_aft)
    np.testing.assert_array_equal(native, predictions.native)

    epochs, normalization = {}, {}
    members = np.arange(len(times)) < len(train)
    selection_members = np.r_[fitting.to_numpy(), np.zeros(len(predictions), dtype=bool)]
    for arm in ('snapshot', 'history'):
        selected = json.loads((folder / f'{arm}_selection.json').read_bytes())
        epochs[arm] = validate_selection(selected, report['training'][arm])
        actual = np.load(folder / f'{arm}_inputs.npz')
        arm_history = history if arm == 'history' else np.repeat(history[:, -1:], 24, axis=1)
        normalization[arm] = {}
        for name, values in [('history', arm_history), ('snapshot', snapshot)]:
            transformed, mean, scale, support = standardize(values, members)
            np.testing.assert_allclose(mean, actual[name + '_mean'], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(scale, actual[name + '_scale'], rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(support, actual[name + '_supported'])
            np.testing.assert_allclose(transformed, actual[name], rtol=1e-6, atol=1e-6)
            selection_data, _, _, selection_support = standardize(values, selection_members)
            assert np.isfinite(selection_data).all()
            normalization[arm][name] = {'refit_supported': int(support.sum()),
                                        'selection_supported': int(selection_support.sum())}
        model = IndependentNetwork().cuda().eval()
        assert sum(p.numel() for p in model.parameters()) == 9505
        model.load_state_dict(torch.load(folder / f'{arm}.pt', weights_only=True, map_location='cuda'))
        replay = []
        with torch.inference_mode():
            for start in range(len(train), len(times), 128):
                h = torch.from_numpy(actual['history'][start:start+128]).cuda()
                x = torch.from_numpy(actual['snapshot'][start:start+128]).cuda()
                replay.append(torch.exp(model(h, x).double()).cpu().numpy())
        np.testing.assert_array_equal(np.concatenate(replay), predictions[arm])
    for arm in ('native', 'snapshot', 'history'):
        np.testing.assert_array_equal(predictions.set_index('timestamp').loc[scored.timestamp, arm], scored[arm])
    scores = {arm: independent_scores(scored, scored[arm].to_numpy()) for arm in ('native', 'snapshot', 'history')}
    paired = {arm: independent_paired(scored, scored.native, scored[arm]) for arm in ('snapshot', 'history')}
    paired['history_vs_snapshot'] = independent_paired(scored, scored.snapshot, scored.history)
    compare(scores, report['scores'])
    compare(paired, report['paired'])
    gates = {arm: all(scores[arm][k] < scores['native'][k] for k in ('exact_row_mae', 'exact_block_mae'))
             and scores[arm]['tail24_block_mae'] <= scores['native']['tail24_block_mae']
             and paired[arm]['all_row_delta_lower'] <= 0 and paired[arm]['all_block_delta_lower'] <= 0
             for arm in ('snapshot', 'history')}
    compare(gates, report['gates'])
    assert report['deployed'] is False and report['year'] == year
    for name, expected_hash in report['output_sha256'].items():
        assert digest(folder / name) == expected_hash, name
    return {'passed': True, 'training_rows': len(train), 'selection_training_rows': int(fitting.sum()),
            'selection_rows': int(selection.sum()), 'deferred_rows': int(deferred.sum()),
            'prediction_rows': len(predictions), 'scoring_rows': len(scored),
            'raw_history_windows': len(times), 'selected_epochs': epochs, 'normalization': normalization,
            'scores': scores, 'paired': paired, 'gates': gates,
            'output_files_checked': len(report['output_sha256'])}


def main():
    needed = [OUT / 'report.json', *(OUT / str(y) / 'report.json' for y in (2024, 2025))]
    pending = [str(path) for path in needed if not path.exists()]
    if pending:
        print(json.dumps({'status': 'waiting_for_reports', 'pending': pending}))
        return
    protocol = json.loads((OUT / 'protocol.json').read_bytes())
    assert protocol['years'] == [2024, 2025] and not protocol['deployment_authorized']
    for old in ['grouped_forecast_aft_2023_20260916', 'direct_history_aft_2023_20260918']:
        original = json.loads((BASE / old / 'protocol.json').read_bytes())
        for path, expected in original['input_sha256'].items():
            assert digest(path) == expected, path
    for path, expected in protocol['input_sha256'].items():
        assert digest(path) == expected, path
    before = {str(path.relative_to(OUT)): digest(path) for path in OUT.rglob('*')
              if path.is_file() and path.name != 'verification.json'}
    hourly = pd.read_parquet(BASE / 'observed_business_dataset_v1/features.parquet')
    repair = json.loads((BASE / 'ghcnh_weather_repair_v1/report.json').read_bytes())
    weather = pd.read_parquet(BASE / 'ghcnh_weather_repair_v1/quality_filtered_weather.parquet').set_index('timestamp')
    patched = hourly[['timestamp', *protocol['history_features']]].copy()
    counts = {}
    for column in ('temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms'):
        aligned = weather[column].reindex(pd.DatetimeIndex(hourly.timestamp)).to_numpy()
        mask = hourly.timestamp.gt(pd.Timestamp(repair['patch_after'])).to_numpy() & hourly[column].isna().to_numpy() & np.isfinite(aligned)
        patched.loc[mask, column] = aligned[mask]
        counts[column] = int(mask.sum())
    compare(counts, repair['patched_grid_cells'])
    compare(counts, protocol['patched_cells'])
    pd.testing.assert_frame_equal(patched, pd.read_parquet(OUT / 'hourly_inputs.parquet'))
    canonical = pd.read_parquet(BASE / 'interval_business_labels_v1/origins.parquet')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    results = {str(year): verify_year(year, protocol, hourly if year == 2024 else patched, canonical)
               for year in (2024, 2025)}
    expected_root = {'year_gates': {year: result['gates'] for year, result in results.items()},
                     'both_years_pass': {arm: all(r['gates'][arm] for r in results.values())
                                         for arm in ('snapshot', 'history')}, 'deployed': False}
    compare(expected_root, json.loads((OUT / 'report.json').read_bytes()))
    for path, expected in protocol['input_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in before.items():
        assert digest(OUT / name) == expected, name
    result = {'passed': True, 'years': results, 'summary': expected_root,
              'input_hashes_checked': len(protocol['input_sha256']), 'artifact_sha256': before,
              'verifier_sha256': digest(__file__),
              'limitations': [('Selection-stage transformed arrays and checkpoints were not saved; membership, '
                              'independent training-only normalization reconstruction, and trace selection were checked, '
                              'but selection losses were not replayed.'),
                              'Historical numerical verification is not proof of real-time data availability.']}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'passed': True, 'years': {y: r['scoring_rows'] for y, r in results.items()},
                      'output': str(OUT / 'verification.json')}))


if __name__ == '__main__':
    main()
