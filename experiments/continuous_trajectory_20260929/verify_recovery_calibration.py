"""Independent audit of development-only recovery decision calibration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ('lag1_trend', 'lag1_chemistry')
YEARS = ('2021', '2022', '2023')
FROZEN_REPORT_SHA256 = 'f00346cd05555e20916569c70097f2157c4f89cb879aa7ffaba90dc10b99340e'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def equal(actual, expected, name='root'):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), name
        for key, value in expected.items():
            equal(actual[key], value, f'{name}.{key}')
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), name
        for index, value in enumerate(expected):
            equal(actual[index], value, f'{name}[{index}]')
    elif isinstance(expected, float):
        assert np.isfinite(actual) and abs(actual - expected) <= 1e-12, (name, actual, expected)
    else:
        assert actual == expected, (name, actual, expected)


def starts(curves, delta=0):
    a = np.asarray(curves, dtype=np.float64)
    assert a.ndim == 2 and a.shape[1] == 24 and np.isfinite(a).all()
    assert (a >= 0).all()
    result = np.full(len(a), 20, dtype=np.int64)
    # Evaluate each window directly, independently of the calibration implementation.
    for start in range(19):
        good = (a[:, start:start + 6] + delta <= 100).all(axis=1)
        result[(result == 20) & good] = start + 1
    return result


def metrics(actual, pred):
    yes, hat = actual < 20, pred < 20
    c = dict(tp=int((yes & hat).sum()), fn=int((yes & ~hat).sum()),
             fp=int((~yes & hat).sum()), tn=int((~yes & ~hat).sum()))
    early = int(((pred < actual) & hat).sum())
    ratio = lambda a, b: float(a / b) if b else None
    return dict(origins=len(actual), confusion=c,
                capped_min_T_20_mae_hours=float(np.abs(actual - pred).mean()) if len(actual) else None,
                precision=ratio(c['tp'], c['tp'] + c['fp']),
                recall=ratio(c['tp'], c['tp'] + c['fn']),
                fpr=ratio(c['fp'], c['fp'] + c['tn']),
                premature_start_count=early,
                premature_start_fraction_all=ratio(early, len(actual)),
                premature_start_fraction_predicted_recovery=ratio(early, int(hat.sum())),
                actual_right_censored=int((~yes).sum()), predicted_no_recovery=int((~hat).sum()))


def independent_grid(actual, curves):
    raw = metrics(actual, starts(curves))
    rows = []
    for delta in range(102):
        m = metrics(actual, starts(curves, delta))
        feasible = (m['fpr'] is not None and m['fpr'] <= .05
                    and m['premature_start_fraction_all'] <= raw['premature_start_fraction_all'])
        rows.append(dict(delta=delta, metrics=m, feasible=feasible))
    feasible = [row for row in rows if row['feasible']]
    assert feasible, 'No calibration candidate satisfies the frozen constraints'
    selected = min(feasible, key=lambda row: (
        row['metrics']['capped_min_T_20_mae_hours'],
        -row['metrics']['recall'], row['metrics']['fpr'], -row['delta']))
    return rows, selected['delta']


def audit(run):
    report = json.loads((run / 'report.json').read_text('utf-8'))
    selected = json.loads((run / 'selected.json').read_text('utf-8'))
    grid = json.loads((run / 'grid.json').read_text('utf-8'))
    source = Path(report['source_run'])
    assert digest(source / 'report.json') == FROZEN_REPORT_SHA256
    frozen = json.loads((source / 'report.json').read_text('utf-8'))
    assert set(report['folds']) == set(selected) == set(grid) == set(YEARS)
    required = {str((source / f'{year}_{suffix}').resolve())
                for year in YEARS for suffix in ('origins.parquet', 'targets.npy',
                                                 'lag1_trend.npy', 'lag1_chemistry.npy')}
    hashes = {str(Path(p).resolve()): value for p, value in report['input_sha256'].items()}
    assert required <= set(hashes), 'Missing frozen input hashes'
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    for field in ('protocol_sha256', 'code_sha256'):
        if isinstance(report.get(field), dict):
            for path, expected in report[field].items():
                assert digest(path) == expected, path
        elif field in report:
            name = 'RECOVERY_CALIBRATION_PROTOCOL.md' if field == 'protocol_sha256' else 'recovery_calibration.py'
            assert digest(Path(__file__).parent / name) == report[field], field
    pooled = {arm: {'actual': [], 'raw': [], 'calibrated': []} for arm in ARMS}
    checked = []
    for year in YEARS:
        assert set(report['folds'][year]) == set(selected[year]) == set(grid[year]) == set(ARMS)
        for suffix in ('origins.parquet', 'targets.npy', 'lag1_trend.npy', 'lag1_chemistry.npy'):
            filename = f'{year}_{suffix}'
            assert digest(source / filename) == frozen['artifact_sha256'][filename], filename
        meta = pd.read_parquet(source / f'{year}_origins.parquet')
        assert meta.source_index.is_unique and meta.timestamp.is_unique
        assert set(meta.partition) == {'dev', 'test'}
        dates = pd.DatetimeIndex(meta.timestamp).tz_convert('Asia/Shanghai')
        expected_year = np.where(meta.partition == 'dev', int(year) - 1, int(year))
        np.testing.assert_array_equal(dates.year, expected_year)
        targets = np.load(source / f'{year}_targets.npy', allow_pickle=False)
        assert len(targets) == len(meta)
        all_truth = starts(targets)
        for arm in ARMS:
            prediction = np.load(source / f'{year}_{arm}.npy', allow_pickle=False)
            assert prediction.shape == targets.shape
            polluted = meta.origin_aqi.to_numpy() > 100
            dev = polluted & (meta.partition.to_numpy() == 'dev')
            test = polluted & (meta.partition.to_numpy() == 'test')
            rows, delta = independent_grid(all_truth[dev], prediction[dev])
            equal(grid[year][arm], rows, f'grid.{year}.{arm}')
            equal(selected[year][arm], delta, f'selected.{year}.{arm}')
            fold = dict(delta=delta,
                        dev=metrics(all_truth[dev], starts(prediction[dev], delta)),
                        test=metrics(all_truth[test], starts(prediction[test], delta)),
                        raw_dev=metrics(all_truth[dev], starts(prediction[dev])),
                        raw_test=metrics(all_truth[test], starts(prediction[test])))
            equal(report['folds'][year][arm], fold, f'fold.{year}.{arm}')
            saved = pd.read_parquet(run / f'{year}_{arm}_starts.parquet')
            assert list(saved.columns) == ['source_index', 'timestamp', 'partition', 'origin_aqi',
                                           'true_start', 'raw_start', 'calibrated_start']
            pd.testing.assert_frame_equal(saved[list(meta.columns)], meta, check_dtype=True)
            np.testing.assert_array_equal(saved.true_start, all_truth)
            np.testing.assert_array_equal(saved.raw_start, starts(prediction))
            np.testing.assert_array_equal(saved.calibrated_start, starts(prediction, delta))
            pooled[arm]['actual'].append(all_truth[test])
            pooled[arm]['raw'].append(starts(prediction[test]))
            pooled[arm]['calibrated'].append(starts(prediction[test], delta))
            checked.append(dict(year=year, arm=arm, delta=delta,
                                dev_origins=int(dev.sum()), test_origins=int(test.sum())))
    aggregate = {}
    for arm, values in pooled.items():
        values = {k: np.concatenate(v) for k, v in values.items()}
        aggregate[arm] = {key: metrics(values['actual'], values[key]) for key in ('raw', 'calibrated')}
    equal(report['pooled_test'], aggregate, 'pooled_test')
    result = dict(status='passed', source_report_sha256=FROZEN_REPORT_SHA256,
                  report_sha256=digest(run / 'report.json'),
                  selected_sha256=digest(run / 'selected.json'), grid_sha256=digest(run / 'grid.json'),
                  verifier_sha256=digest(Path(__file__)), checked_folds=checked,
                  independently_checked_development_candidates=612,
                  selection_uses_test=False, target_threshold_unchanged=True,
                  concentration_curves_unchanged=True)
    (run / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run), indent=2))
