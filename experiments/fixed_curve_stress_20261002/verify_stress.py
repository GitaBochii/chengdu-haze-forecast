"""Independent reconstruction and audit of the frozen cross-year curve run.

Only the already independent 323-feature reconstruction is reused. Qualification,
processes, labels, metrics, gates and inference are independent of stress_runner.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT/'data/processed/regional_air_models_full_history/city_hourly.parquet'
SOURCE_SHA = 'b16c1941da64f5ef19bc5856cd1215851c4130ffb80b18ca0c16af10f118ab0e'
RUN76 = ROOT/'experiments/learned_path_weights_20261002/runs/20261002T130057090429Z'
RUN323 = ROOT/'experiments/empirical_paths_20261002/runs/20261002T121844976298Z'
CHEMISTRY = ROOT/'experiments/continuous_trajectory_20260929'
YEARS = (2021, 2022, 2023, 2024, 2025)
HOUR = 3_600_000_000_000
CHANNELS = ('pm2_5', 'pm10', 'aqi')
SPECIES = ('so2', 'no2', 'co', 'o3_8h')
ARMS = ('raw76', 'conservative76', 'raw323', 'conservative323')
_RECONSTRUCT = None


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for piece in iter(lambda: source.read(1024*1024), b''):
            result.update(piece)
    return result.hexdigest()


def freeze(paths):
    return {str(Path(path).resolve()): sha(path) for path in paths}


def assert_unchanged(frozen):
    for path, digest in frozen.items():
        assert sha(path) == digest, f'Frozen input changed: {path}'


def dates_of(raw):
    dates = pd.DatetimeIndex(raw.timestamp)
    if dates.tz is None:
        raise ValueError('Source timestamps require an explicit timezone')
    dates = dates.tz_convert('Asia/Shanghai')
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Source timestamps must be finite, unique and sorted')
    return dates


def validate_source(raw, complete_years=()):
    dates = dates_of(raw)
    if not (dates.asi8 % HOUR == 0).all():
        raise ValueError('Source timestamps must lie on exact hours')
    if not (np.diff(dates.asi8) == HOUR).all():
        raise ValueError('Source is not a consecutive hourly grid')
    for year in complete_years:
        expected = pd.date_range(f'{year}-01-01', f'{year+1}-01-01',
                                 freq='h', inclusive='left', tz='Asia/Shanghai')
        if not np.array_equal(dates[dates.year == year].asi8, expected.asi8):
            raise ValueError(f'Incomplete calendar year: {year}')
    return dates


def assign_groups(dates, aqi):
    dates, aqi = pd.DatetimeIndex(dates), np.asarray(aqi)
    if dates.tz is None or len(dates) != len(aqi):
        raise ValueError('Timezone-aware dates and equally sized AQI are required')
    result = np.full(len(dates), '', dtype='U96')
    epoch = pd.Timestamp('2014-01-01', tz='Asia/Shanghai').value
    segment, serial, active, clean_hours = -1, 0, None, 0
    for row, (date, value) in enumerate(zip(dates, aqi)):
        split = row == 0 or not np.isfinite(aqi[row-1])
        split |= row > 0 and (date.year != dates[row-1].year or date.value-dates[row-1].value != HOUR)
        if split:
            segment += 1
            active, clean_hours = None, 0
        if not np.isfinite(value):
            active, clean_hours = None, 0
            continue
        if active is None and value > 100:
            serial += 1
            active = f'{date.year}:s{segment}:p{serial}'
        if active is None:
            block = (date.value-epoch)//(72*HOUR)
            result[row] = f'{date.year}:s{segment}:c{block}'
        else:
            result[row] = active
            clean_hours = clean_hours+1 if value <= 100 else 0
            if clean_hours == 6:
                active, clean_hours = None, 0
    return result


def starts(curves):
    values = np.asarray(curves)
    if values.ndim != 2 or values.shape[1] != 24 or not np.isfinite(values).all():
        raise ValueError('Finite complete 24-hour curves required')
    result = np.full(len(values), 20, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(values, 6, axis=1)
    recovered = (windows <= 100).all(axis=2)
    any_recovery = recovered.any(axis=1)
    result[any_recovery] = recovered[any_recovery].argmax(axis=1)+1
    return result


def chemistry_reconstruction(raw, dates, indices):
    global _RECONSTRUCT
    if _RECONSTRUCT is None:
        sys.path.insert(0, str(CHEMISTRY))
        spec = importlib.util.spec_from_file_location('independent_frozen_chemistry', CHEMISTRY/'verify_delayed_chemistry.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RECONSTRUCT = module.reconstruct
    return _RECONSTRUCT(raw, dates, indices, 2019, 1)


def rebuild_year(raw, year):
    dates = dates_of(raw)
    selected = np.flatnonzero(dates.year == year)
    raw = raw.iloc[selected].reset_index(drop=True)
    dates = dates[selected]
    if len(raw) == 0:
        raise ValueError(f'No rows for {year}')
    core = raw[list(CHANNELS)].to_numpy(np.float64)
    full_groups = assign_groups(dates, core[:, 2])
    reasons = np.full(len(raw), 'eligible', dtype='U24')
    eligible = []
    for row in range(len(raw)):
        if row < 24 or row+24 >= len(raw):
            reasons[row] = 'year_edge'
        elif not (np.diff(dates.asi8[row-24:row+25]) == HOUR).all():
            reasons[row] = 'hour_gap'
        elif not np.isfinite(core[row, 2]):
            reasons[row] = 'origin_aqi_nonfinite'
        elif not np.isfinite(core[row-24:row]).all():
            reasons[row] = 'history_nonfinite'
        elif not np.isfinite(core[row+1:row+25]).all():
            reasons[row] = 'future_nonfinite'
        else:
            eligible.append(row)
    indices = np.asarray(eligible, dtype=np.int64)
    history = core[indices[:, None]+np.arange(-24, 0)]
    paths = core[indices[:, None]+np.arange(1, 25)]
    calendar = np.column_stack((np.sin(2*np.pi*dates[indices].hour/24),
        np.cos(2*np.pi*dates[indices].hour/24), np.sin(2*np.pi*dates[indices].month/12),
        np.cos(2*np.pi*dates[indices].month/12)))
    features76 = np.column_stack((history.reshape(len(indices), 72), calendar)).astype(np.float64)
    if len(indices):
        features323, chemistry_history = chemistry_reconstruction(raw, dates, indices)
    else:
        features323 = np.empty((0, 323), np.float32)
        chemistry_history = np.empty((0, 24, 4), np.float64)
    data = dict(indices=indices, timestamps=dates.asi8[indices], groups=full_groups[indices],
        origin_aqi=core[indices, 2], features76=features76, features323=features323,
        paths=paths, labels=starts(paths[:, :, 2]))
    return dates, data, reasons


def fraction(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def group_mean(values, groups):
    values, groups = np.asarray(values), np.asarray(groups)
    if len(values) == 0:
        return None
    return float(np.mean([np.mean(values[groups == key]) for key in np.unique(groups)]))


def binary_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, bool), np.asarray(predicted, bool)
    tp = int(np.sum(actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    return dict(n=len(actual), positive_n=int(actual.sum()), negative_n=int((~actual).sum()),
        predicted_positive=int(predicted.sum()), confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn),
        recall=fraction(tp, tp+fn), precision=fraction(tp, tp+fp), fpr=fraction(fp, fp+tn))


def mean_rates(records, keys):
    return {key: dict(value=float(np.mean(values)) if values else None,
                     valid_groups=len(values), total_groups=len(records))
            for key in keys for values in [[row[key] for row in records if row[key] is not None]]}


def recovery_metrics(actual, predicted, groups):
    actual, predicted, groups = map(np.asarray, (actual, predicted, groups))
    errors = np.abs(actual.astype(np.float64)-predicted.astype(np.float64))
    early = int(np.sum((predicted < actual) & (predicted < 20)))
    binary = binary_metrics(actual < 20, predicted < 20)
    result = {key: binary[key] for key in ('n', 'confusion', 'recall', 'precision', 'fpr')}
    group_records = []
    for group in np.unique(groups):
        mask = groups == group
        one = binary_metrics(actual[mask] < 20, predicted[mask] < 20)
        one['mae'] = float(errors[mask].mean())
        one['early_fraction'] = float(((predicted[mask] < actual[mask]) & (predicted[mask] < 20)).mean())
        group_records.append(one)
    result.update(mae=float(errors.mean()) if len(errors) else None,
        group_equal_mae=group_mean(errors, groups), groups=int(len(np.unique(groups))),
        actual_right_censored=int(np.sum(actual == 20)), predicted_recovery=int(np.sum(predicted < 20)),
        early_count=early, early_fraction=fraction(early, len(actual)),
        group_equal=mean_rates(group_records, ('mae', 'recall', 'precision', 'fpr', 'early_fraction')))
    return result


def concentration_metrics(actual, predicted, groups):
    actual, predicted = np.asarray(actual, np.float64), np.asarray(predicted, np.float64)
    errors = np.abs(actual-predicted)
    return dict(n=len(actual), hour_cells=int(errors.size), mae=float(errors.mean()) if len(errors) else None,
        group_equal_mae=group_mean(errors.mean(axis=1), groups), units='AQI',
        lead_mae=[float(errors[:, lead].mean()) if len(errors) else None for lead in range(24)],
        group_equal_lead_mae=[group_mean(errors[:, lead], groups) for lead in range(24)])


def year_equal(records):
    if not records:
        raise ValueError('At least one year required')
    if isinstance(records[0], dict):
        return {key: year_equal([record[key] for record in records]) for key in records[0]}
    if isinstance(records[0], list):
        return [year_equal([record[index] for record in records]) for index in range(len(records[0]))]
    if isinstance(records[0], str):
        assert all(record == records[0] for record in records)
        return records[0]
    if isinstance(records[0], bool):
        return dict(value=bool(all(records)), valid_years=len(records), total_years=len(records))
    valid = [float(record) for record in records if record is not None]
    return dict(value=float(np.mean(valid)) if valid else None,
                valid_years=len(valid), total_years=len(records))


def evaluate(data):
    target = data['paths'][:, :, 2]
    truth = data['labels']
    groups = data['groups']
    pollution = data['origin_aqi'] > 100
    common = data['covered76'] & data['covered323']
    result = dict(recovery={}, concentration={}, clean_onset={})
    for arm in ARMS:
        suffix = '76' if arm.endswith('76') else '323'
        coverage = data['covered'+suffix]
        prefix = 'raw_curve' if arm.startswith('raw') else 'conservative_curve'
        curve = data[prefix+suffix]
        labels = np.full(len(curve), 20, dtype=np.int64)
        labels[coverage] = starts(curve[coverage])
        masks = dict(all=coverage, polluted=coverage & pollution, clean=coverage & ~pollution,
            common=common, common_polluted=common & pollution, common_clean=common & ~pollution)
        result['recovery'][arm] = {name: recovery_metrics(truth[mask], labels[mask], groups[mask])
                                  for name, mask in masks.items()}
        result['concentration'][arm] = {name: concentration_metrics(target[mask], curve[mask], groups[mask])
                                       for name, mask in masks.items()}
        clean = coverage & ~pollution
        actual_onset = (target > 100).any(axis=1)
        predicted_onset = (curve > 100).any(axis=1)
        onset = binary_metrics(actual_onset[clean], predicted_onset[clean])
        onset['group_equal'] = mean_rates([binary_metrics(actual_onset[clean & (groups == group)],
            predicted_onset[clean & (groups == group)]) for group in np.unique(groups[clean])],
            ('recall', 'precision', 'fpr'))
        result['clean_onset'][arm] = onset
    return result


def population_counts(data, reasons):
    n = len(data['indices'])
    pollution = data['origin_aqi'] > 100
    positive = data['labels'] < 20
    result = dict(candidate_hours=len(reasons), excluded_hours=int(np.sum(reasons != 'eligible')),
        eligible_origins=n, polluted=int(pollution.sum()), clean=int((~pollution).sum()),
        polluted_positive=int(np.sum(pollution & positive)), polluted_negative=int(np.sum(pollution & ~positive)),
        covered76=int(data['covered76'].sum()), covered323=int(data['covered323'].sum()),
        common_covered=int((data['covered76'] & data['covered323']).sum()),
        coverage76=fraction(data['covered76'].sum(), n), coverage323=fraction(data['covered323'].sum(), n),
        eligible_fraction_of_calendar=fraction(n, len(reasons)))
    return result


def gates(annual, pooled, counts):
    checks = {}
    for year in YEARS:
        key = str(year)
        checks[key+':coverage76_100'] = counts[key]['coverage76'] == 1.
        checks[key+':coverage323_100'] = counts[key]['coverage323'] == 1.
        checks[key+':polluted_positive_and_negative'] = counts[key]['polluted_positive'] > 0 and counts[key]['polluted_negative'] > 0
    for key, record in list(annual.items())+[('pooled', pooled)]:
        candidate, control = (record['recovery'][arm]['polluted'] for arm in ('conservative76', 'conservative323'))
        for metric in ('mae', 'group_equal_mae', 'recall'):
            a, b = candidate[metric], control[metric]
            checks[key+':'+metric+'_not_worse'] = a is not None and b is not None and (a >= b if metric == 'recall' else a <= b)
        checks[key+':fpr_5pct'] = candidate['fpr'] is not None and candidate['fpr'] <= .05
    stable = dict(checks=checks, passed=bool(all(checks.values())),
                  failure_reasons=[key for key, value in checks.items() if not value])
    candidate, control = (pooled['recovery'][arm]['polluted'] for arm in ('conservative76', 'conservative323'))
    relative = fraction(control['mae']-candidate['mae'], control['mae']) if None not in (candidate['mae'], control['mae']) else None
    recall_delta = candidate['recall']-control['recall'] if None not in (candidate['recall'], control['recall']) else None
    checks = dict(stability_passed=stable['passed'], pooled_mae_10pct=relative is not None and control['mae'] > 0
        and candidate['mae'] <= .9*control['mae'],
        pooled_recall_5pp=recall_delta is not None and candidate['recall'] >= control['recall']+.05)
    meaningful = dict(checks=checks, passed=bool(all(checks.values())),
                      failure_reasons=[key for key, value in checks.items() if not value],
                      relative_pooled_mae_reduction=relative, pooled_recall_difference=recall_delta)
    return stable, meaningful


def predict_fixed(heads, features, buffer):
    if len(heads) != 24 or buffer not in (14., 18.):
        raise ValueError('Exactly 24 frozen heads and a frozen buffer are required')
    raw = np.empty((0, 24), dtype=np.float32) if not len(features) else np.column_stack([
        np.asarray(head.predict(features, num_threads=4)).clip(min=0).astype(np.float32) for head in heads])
    conservative = raw.astype(np.float64)+buffer
    covered = np.isfinite(raw).all(axis=1)
    return raw, conservative, covered


def curve_for(data, arm):
    suffix = '76' if arm.endswith('76') else '323'
    prefix = 'raw_curve' if arm.startswith('raw') else 'conservative_curve'
    return data[prefix+suffix], data['covered'+suffix]


def row_frame(data, year):
    rows = pd.DataFrame(dict(year=np.full(len(data['indices']), year),
        year_row=np.arange(len(data['indices'])), source_index=data['indices'],
        timestamp=pd.to_datetime(data['timestamps'], utc=True).tz_convert('Asia/Shanghai'),
        group=data['groups'], polluted=data['origin_aqi'] > 100,
        origin_aqi=data['origin_aqi'],
        covered76=data['covered76'], covered323=data['covered323'], true_start=data['labels'],
        true_onset=(data['paths'][:, :, 2] > 100).any(axis=1)))
    for arm in ARMS:
        curves, coverage = curve_for(data, arm)
        predicted = np.full(len(curves), -1, dtype=np.int64)
        predicted[coverage] = starts(curves[coverage])
        name = arm.replace('76', '_start76').replace('323', '_start323')
        rows[name] = predicted
        errors = np.abs(predicted-data['labels']).astype(np.float64)
        errors[~coverage] = np.nan
        rows[arm+'_absolute_error'] = errors
        rows[arm+'_onset'] = pd.array(np.where(coverage, (curves > 100).any(axis=1), None), dtype='boolean')
        concentration_error = np.abs(curves.astype(np.float64)-data['paths'][:, :, 2]).mean(axis=1)
        concentration_error[~coverage] = np.nan
        rows[arm+'_concentration_error'] = concentration_error
    return rows


def group_frame(data, year):
    records = []
    pollution = data['origin_aqi'] > 100
    for polluted in (True, False):
        for group in np.unique(data['groups'][pollution == polluted]):
            mask = (pollution == polluted) & (data['groups'] == group)
            record = dict(year=year, polluted=bool(polluted), group=group, n=int(mask.sum()))
            for arm in ARMS:
                curve, coverage = curve_for(data, arm)
                keep = mask & coverage
                predicted = starts(curve[keep])
                record[arm+'_n'] = int(keep.sum())
                record[arm+'_mae'] = float(np.abs(predicted-data['labels'][keep]).mean()) if keep.any() else None
                record[arm+'_concentration_mae'] = float(np.abs(curve[keep].astype(np.float64)-
                    data['paths'][keep, :, 2]).mean()) if keep.any() else None
            records.append(record)
    return pd.DataFrame(records).sort_values(['year', 'polluted', 'group']).reset_index(drop=True)


def chemical_coverage(data):
    history = data['features323'][:, 139:235].reshape(-1, 24, 4)
    pollution = data['origin_aqi'] > 100
    result = {}
    for name, mask in (('all', np.ones(len(history), bool)), ('polluted', pollution), ('clean', ~pollution)):
        record = dict(n=int(mask.sum()))
        for channel, species in enumerate(SPECIES):
            good = np.isfinite(history[mask, :, channel])
            record[species] = dict(finite_cells=int(good.sum()), total_cells=int(good.size),
                history_fraction=fraction(good.sum(), good.size), origins_all24=int(good.all(axis=1).sum()),
                origins_any=int(good.any(axis=1).sum()))
        result[name] = record
    return result


def negative_cells(values, channels):
    values = np.asarray(values)
    return {name: dict(finite_cells=int(np.isfinite(values[..., channel]).sum()),
        negative_cells=int((np.isfinite(values[..., channel]) & (values[..., channel] < 0)).sum()),
        nonfinite_cells=int((~np.isfinite(values[..., channel])).sum())) for channel, name in enumerate(channels)}


def negative_diagnostics(raw, data):
    core = raw[list(CHANNELS)].to_numpy(np.float64)
    idx = data['indices']
    return dict(raw=negative_cells(raw[list(CHANNELS+SPECIES)].to_numpy(np.float64), CHANNELS+SPECIES),
        eligible_history=negative_cells(core[idx[:, None]+np.arange(-24, 0)], CHANNELS),
        eligible_future=negative_cells(data['paths'], CHANNELS), issue=negative_cells(core[idx], CHANNELS))


def sum_counts(records):
    if isinstance(records[0], dict):
        return {key: sum_counts([record[key] for record in records]) for key in records[0]}
    return sum(records)


def stack_years(bundles):
    keys = bundles[0].keys()
    assert all(bundle.keys() == keys for bundle in bundles)
    return {key: np.concatenate([bundle[key] for bundle in bundles]) for key in keys}


def read_json(path):
    return json.loads(Path(path).read_text('utf-8'))


def dependency_paths():
    paths = [HERE/'PROTOCOL.md', Path(__file__), HERE/'test_verify_stress.py',
             CHEMISTRY/'verify_delayed_chemistry.py', CHEMISTRY/'verify_issue_delay.py', Path(sys.executable)]
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'lightgbm', 'pyarrow'):
        distribution = importlib.metadata.distribution(name)
        paths.extend(Path(distribution.locate_file(file)) for file in distribution.files or []
                     if str(file).endswith('.dist-info/RECORD'))
    return paths


def versions():
    return dict(python=sys.version, executable=sys.executable,
        **{name: importlib.metadata.version(name) for name in
           ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'lightgbm', 'pyarrow')})


def prior_inputs():
    paths = []
    for directory in (RUN76, RUN323):
        report, audit = read_json(directory/'report.json'), read_json(directory/'verification.json')
        assert audit['status'] == 'passed', f'Prior run lacks passed independent verification: {directory}'
        assert_unchanged(report['frozen_sha256'])
        assert_unchanged(audit['independent_frozen_sha256'])
        paths.extend(Path(path) for path in report['frozen_sha256'])
        paths.extend(Path(path) for path in audit['independent_frozen_sha256'])
        paths.extend((directory/'report.json', directory/'verification.json'))
        for name, digest in report['artifact_sha256'].items():
            path = directory/name
            assert sha(path) == digest, path
            paths.append(path)
    return paths


def load_heads():
    result = {}
    for width, directory, prefix in ((76, RUN76, 'lgbm76'), (323, RUN323, 'chemistry')):
        result[str(width)] = []
        for lead in range(1, 25):
            model = lgb.Booster(model_str=(directory/f'{prefix}_aqi_h{lead:02}.txt').read_text('utf-8'))
            assert model.num_feature() == width and model.current_iteration() == 300
            assert model.params['objective'] == 'regression_l1' and model.params['num_leaves'] == 24
            assert model.params['min_data_in_leaf'] == 30 and np.isclose(float(model.params['learning_rate']), .04)
            assert model.params['deterministic'] and model.params['force_col_wise']
            result[str(width)].append(model)
    return result


def add_predictions(data, heads):
    result = dict(data)
    for suffix, buffer in (('76', 14.), ('323', 18.)):
        raw, conservative, covered = predict_fixed(heads[suffix], data['features'+suffix], buffer)
        result['raw_curve'+suffix] = raw
        result['conservative_curve'+suffix] = conservative
        result['covered'+suffix] = covered
    return result


def verify_2020(raw, heads):
    dates, data, reasons = rebuild_year(raw, 2020)
    reference = dict(np.load(RUN323/'dev.npz', allow_pickle=False))
    assert len(data['indices']) == 7618
    for key, old in (('timestamps', 'timestamps'), ('features76', 'features'), ('paths', 'paths'),
                     ('origin_aqi', 'origin_aqi'), ('labels', 'labels')):
        np.testing.assert_array_equal(data[key], reference[old], err_msg='2020/'+key)
    np.testing.assert_array_equal(pd.factorize(data['groups'])[0], pd.factorize(reference['groups'])[0])
    first = int(np.flatnonzero(dates_of(raw).year == 2020)[0])
    np.testing.assert_array_equal(data['indices']+first, reference['indices'])
    previous_chemistry = np.load(RUN323/'chemistry_features.npy', mmap_mode='r')
    np.testing.assert_array_equal(data['features323'], previous_chemistry[-7618:])
    predicted = add_predictions(data, heads)
    old_prediction = dict(np.load(RUN76/'prediction.npz', allow_pickle=False))
    for suffix in ('76', '323'):
        for kind in ('raw_curve', 'conservative_curve'):
            np.testing.assert_array_equal(predicted[kind+suffix], old_prediction[kind+suffix], err_msg='2020/'+kind+suffix)
        assert predicted['covered'+suffix].all()
    old_excluded = pd.read_parquet(RUN323/'exclusions.parquet')
    old_excluded = old_excluded.loc[old_excluded.partition == 'dev']
    excluded = np.flatnonzero(reasons != 'eligible')
    np.testing.assert_array_equal(excluded+first, old_excluded.source_index)
    np.testing.assert_array_equal(dates.asi8[excluded], pd.DatetimeIndex(old_excluded.timestamp).asi8)
    return dict(status='passed', origins=7618, excluded_hours=len(excluded))


def verify_frame(actual, expected, label):
    assert set(actual.columns) == set(expected.columns), (label, actual.columns, expected.columns)
    actual = actual[expected.columns].reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    if 'group' in expected:
        np.testing.assert_array_equal(pd.factorize(actual.group)[0], pd.factorize(expected.group)[0], err_msg=label+'/groups')
        assert all(str(group).startswith(str(int(year))+':') for group, year in zip(actual.group, actual.year)), label
        actual = actual.drop(columns=['group'])
        expected = expected.drop(columns=['group'])
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12, obj=label)


def verify_bundle(actual, expected, year):
    assert actual.keys() == expected.keys(), (year, actual.keys(), expected.keys())
    for key in expected:
        assert actual[key].shape == expected[key].shape, (year, key, actual[key].shape, expected[key].shape)
        if key == 'groups':
            assert all(str(value).startswith(str(year)+':') for value in actual[key])
            np.testing.assert_array_equal(pd.factorize(actual[key], sort=False)[0],
                                          pd.factorize(expected[key], sort=False)[0], err_msg=f'{year}/{key}')
        else:
            assert actual[key].dtype == expected[key].dtype, (year, key, actual[key].dtype, expected[key].dtype)
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=f'{year}/{key}')


def eligibility_frame(data, year):
    return pd.DataFrame(dict(year=np.full(len(data['indices']), year), source_index=data['indices'],
        timestamp=pd.to_datetime(data['timestamps'], utc=True).tz_convert('Asia/Shanghai'),
        group=data['groups'], origin_aqi=data['origin_aqi'], true_start=data['labels'],
        covered76=data['covered76'], covered323=data['covered323']))


def exclusion_frame(dates, reasons, year):
    rows = np.flatnonzero(reasons != 'eligible')
    return pd.DataFrame(dict(year=np.full(len(rows), year), source_index=rows,
                            timestamp=dates[rows], reason=reasons[rows]))


def verify(run, report, audit):
    assert report['schema_version'] == 1 and report['status'] == 'pending_independent_verification'
    assert report['years'] == list(YEARS) and report['train_years'] == list(range(2014, 2020))
    assert report['selected_year'] == 2020 and report['buffers'] == dict(lgbm76=14, lgbm323=18)
    assert report['historical_stress_test'] and not any(report[key] for key in ('blind_test', 'refitted', 'deployed'))
    assert Path(report['source']).resolve() == SOURCE.resolve() and report['source_sha256'] == SOURCE_SHA
    assert report['timestamp_timezone'] == 'Asia/Shanghai'
    assert report['source_index_scope'] == 'year-local unfiltered source row'
    assert report['source_runs'] == dict(lgbm76=str(RUN76), lgbm323=str(RUN323))
    assert report['feature_dimensions'] == dict(lgbm76=76, lgbm323=323)
    model_paths = {str(directory/f'{prefix}_aqi_h{lead:02}.txt') for directory, prefix in
                   ((RUN76, 'lgbm76'), (RUN323, 'chemistry')) for lead in range(1, 25)}
    assert set(report['model_sha256']) == model_paths
    assert_unchanged(report['model_sha256'])
    assert_unchanged(report['reuse_verified_sha256'])
    assert_unchanged(report['frozen_sha256'])
    assert sha(SOURCE) == SOURCE_SHA
    expected_artifacts = {f'bundle_{year}.npz' for year in YEARS} | {
        'eligible.parquet', 'exclusions.parquet', 'row_metrics.parquet', 'group_metrics.parquet', 'annual_metrics.json'}
    assert set(report['artifact_sha256']) == expected_artifacts
    for name, digest in report['artifact_sha256'].items():
        assert sha(run/name) == digest, name
    audit['checks'].append('Protocol, pending immutable report, source and both independently passed model provenance chains; all artifact hashes')

    raw = pd.read_parquet(SOURCE)
    all_dates = validate_source(raw, complete_years=YEARS)
    assert len(raw) == 107928 and all_dates[0].year == 2014 and all_dates[-1].year == 2026
    heads = load_heads()
    compatibility = verify_2020(raw, heads)
    for key, value in compatibility.items():
        assert report['compatibility2020'][key] == value, ('compatibility2020', key)
    assert len(report['compatibility2020']['checks']) == 6
    audit['compatibility2020'] = compatibility
    audit['checks'].append('Independent 2020 qualification, year-local/source indices, exclusions, process equality, features and all 48 saved-head curves are exactly compatible')
    annual, counts, chemistry, negatives = {}, {}, {}, {}
    bundles, eligible, excluded, rows, groups = [], [], [], [], []
    for year in YEARS:
        dates, base, reasons = rebuild_year(raw, year)
        expected = add_predictions(base, heads)
        saved = dict(np.load(run/f'bundle_{year}.npz', allow_pickle=False))
        verify_bundle(saved, expected, year)
        assert np.isfinite(base['features76']).all() and not np.isinf(base['features323']).any()
        assert (dates.year.to_numpy()[base['indices'][:, None]+np.arange(-24, 25)] == year).all()
        key = str(year)
        annual[key] = evaluate(expected)
        counts[key] = population_counts(expected, reasons)
        chemistry[key] = chemical_coverage(expected)
        annual_raw = raw.loc[all_dates.year == year].reset_index(drop=True)
        negatives[key] = negative_diagnostics(annual_raw, expected)
        assert_close(annual[key], report['metrics']['annual'][key], key+'/metrics')
        assert_close(counts[key], report['counts']['annual'][key], key+'/counts')
        assert_close(chemistry[key], report['metrics']['chemical_coverage']['annual'][key], key+'/chemical_coverage')
        assert_close(negatives[key], report['negative_diagnostics']['annual'][key], key+'/negative_diagnostics')
        eligible.append(eligibility_frame(expected, year))
        excluded.append(exclusion_frame(dates, reasons, year))
        rows.append(row_frame(expected, year))
        groups.append(group_frame(expected, year))
        bundles.append(expected)
        print(json.dumps(dict(stage='independent_year_passed', year=year, origins=len(base['indices']))), flush=True)
    pooled_bundle = stack_years(bundles)
    pooled = evaluate(pooled_bundle)
    total_candidates = sum(record['candidate_hours'] for record in counts.values())
    total_excluded = sum(record['excluded_hours'] for record in counts.values())
    pooled_reasons = np.full(total_candidates, 'eligible', dtype='U16')
    pooled_reasons[:total_excluded] = 'excluded'
    pooled_counts = population_counts(pooled_bundle, pooled_reasons)
    assert_close(pooled, report['metrics']['pooled'], 'pooled/metrics')
    assert_close(pooled_counts, report['counts']['pooled'], 'pooled/counts')
    assert_close(year_equal(list(annual.values())), report['metrics']['year_equal'], 'year_equal')
    assert_close(chemical_coverage(pooled_bundle), report['metrics']['chemical_coverage']['pooled'], 'pooled/chemical_coverage')
    assert_close(sum_counts(list(negatives.values())), report['negative_diagnostics']['pooled'], 'pooled/negative_diagnostics')
    assert_close(annual, read_json(run/'annual_metrics.json'), 'annual_metrics artifact')
    for name, tables in (('eligible', eligible), ('exclusions', excluded), ('row_metrics', rows), ('group_metrics', groups)):
        actual = pd.read_parquet(run/f'{name}.parquet')
        expected = pd.concat(tables, ignore_index=True)
        verify_frame(actual, expected, name)
    stability, meaningful = gates(annual, pooled, counts)
    assert_close(stability, report['stability_gates'], 'stability_gates')
    assert_close(meaningful, report['meaningful_gates'], 'meaningful_gates')
    model_sizes = {name: sum((directory/f'{prefix}_aqi_h{lead:02}.txt').stat().st_size for lead in range(1, 25))
        for name, directory, prefix in (('lgbm76', RUN76, 'lgbm76'), ('lgbm323', RUN323, 'chemistry'))}
    model_sizes['total'] = sum(model_sizes.values())
    assert_close(model_sizes, report['model_disk_bytes'], 'model_disk_bytes')
    audit.update(independently_reinferred_models=48, reloaded_head_prediction_calls=48*6,
        years=list(YEARS), counts=dict(annual=counts, pooled=pooled_counts),
        stability_gates=stability, meaningful_gates=meaningful, model_disk_bytes=model_sizes)
    audit['checks'].extend([
        'Every unfiltered yearly hour independently qualified and excluded; all-year grouping before filtering, year namespace, full year boundaries',
        'All annual indices/timestamps/origin AQI/paths/labels and float64 monthly 76 features; independent float32 dayofyear 323 features',
        'All 48 saved heads reloaded and each 2021-2025 prediction exactly reproduced after clip/float32/fixed float64 buffer; coverage independently rebuilt',
        'Every row/group table, annual/pooled/year-equal recovery and concentration metric, each lead and common population, separate clean-onset confusion',
        'Every chemistry valid-coverage and raw negative-value count; model byte sizes and feature widths',
        'Both preregistered gates recomputed, including null denominator failures; audit passing does not mean either scientific gate passed'])


def main(run):
    run = Path(run).resolve()
    assert run.parent == (HERE/'runs').resolve(), 'Audit may only write the new isolated stress run'
    assert not (run/'audit_manifest.json').exists() and not (run/'verification.json').exists(), 'Audit is append-once; do not overwrite prior evidence'
    report = read_json(run/'report.json')
    paths = dependency_paths()+prior_inputs()+[run/'report.json', SOURCE]
    paths.extend(run/name for name in report['artifact_sha256'])
    paths.extend(Path(path) for path in report['frozen_sha256'])
    frozen = freeze(sorted(set(paths)))
    assert_unchanged(frozen)
    manifest = dict(schema_version=1, frozen_sha256=frozen, versions=versions(),
        independent_feature_dependencies=['verify_delayed_chemistry.py', 'verify_issue_delay.py'],
        main_business_functions_imported=False)
    (run/'audit_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    audit = dict(schema_version=1, status='running', verifier_sha256=sha(__file__),
        report_sha256=sha(run/'report.json'), audit_manifest_sha256=sha(run/'audit_manifest.json'),
        independent_frozen_sha256=frozen, checks=[],
        main_business_functions_imported=False,
        limits=['Passing only establishes exact reproducibility, not scientific accuracy or stability gates',
            'Historical archived stress test; not blind, prospective or live service validation',
            '2014-2019 fixed models with 2020 fixed buffers; no refit, training or parameter selection',
            'AQI is available-station per-pollutant median, not official city AQI',
            'Recovery label is min(T,20) and does not measure full dissipation duration',
            'No path probability or calibration; model disk bytes do not prove service speed'])
    (run/'verification.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    try:
        verify(run, report, audit)
        assert_unchanged(frozen)
        assert sha(run/'report.json') == audit['report_sha256']
        assert sha(run/'audit_manifest.json') == audit['audit_manifest_sha256']
        audit['status'] = 'passed'
    except BaseException as error:
        audit['status'] = 'failed'
        audit['error'] = repr(error)
        raise
    finally:
        (run/'verification.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps(dict(status=audit['status'], report_sha256=audit['report_sha256'],
            checks_completed=len(audit['checks']), error=audit.get('error'))), flush=True)


def assert_close(actual, expected, label='value'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (label, actual.keys(), expected.keys())
        for key in expected:
            assert_close(actual[key], expected[key], label+'/'+key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), label
        for row, (a, b) in enumerate(zip(actual, expected)):
            assert_close(a, b, label+f'/{row}')
    elif expected is None or isinstance(expected, (str, bool)):
        assert actual == expected, (label, actual, expected)
    else:
        assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12), (label, actual, expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)
