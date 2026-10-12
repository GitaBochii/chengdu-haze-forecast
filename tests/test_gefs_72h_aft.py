import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/benchmark_gefs_72h_aft.py'
    assert path.exists(), 'benchmark entry point has not been implemented'
    return importlib.import_module('scripts.benchmark_gefs_72h_aft')


def test_missing_audit_rejects_without_output(module, tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        module.main(tmp_path)
    assert not (tmp_path / 'artifacts/task01_recovery_head/gefs_72h_aft_v1').exists()


def make_audit(module, root):
    folder = root / 'artifacts/task01_recovery_head/gefs_72h_features_v1'
    folder.mkdir(parents=True)
    paths = module.required_audit_paths(root)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture', encoding='utf-8')
    audit = {'passed': True, 'origin_rows': 7890, 'snapshot_rows': 292,
             'input_sha256': {str(p): module.sha(p) for p in paths},
             'checks': list(module.MATCHING_CHECKS)}
    target = folder / 'matching_audit.json'
    target.write_text(json.dumps(audit), encoding='utf-8')
    return target, audit


def test_audit_accepts_exact_contract(module, tmp_path):
    path, _ = make_audit(module, tmp_path)
    inputs = module.require_matching_audit(tmp_path)
    assert inputs[str(path)] == module.sha(path)


@pytest.mark.parametrize('mutation', ['count', 'passed', 'checks', 'missing_hash', 'tamper'])
def test_audit_rejects_incomplete_or_stale(module, tmp_path, mutation):
    path, audit = make_audit(module, tmp_path)
    if mutation == 'count':
        audit['snapshot_rows'] = 291
    elif mutation == 'passed':
        audit['passed'] = False
    elif mutation == 'checks':
        audit['checks'] = []
    elif mutation == 'missing_hash':
        audit['input_sha256'].pop(next(iter(audit['input_sha256'])))
    else:
        Path(next(iter(audit['input_sha256']))).write_text('changed', encoding='utf-8')
    path.write_text(json.dumps(audit), encoding='utf-8')
    with pytest.raises(ValueError):
        module.require_matching_audit(tmp_path)


def test_attach_retains_missing_rows_and_history(module):
    times = pd.date_range('2023-01-01', periods=3, freq='h', tz='UTC')
    base = pd.DataFrame({'timestamp': times, 'aqi': [110., np.nan, 130.]})
    forecast = pd.DataFrame({'timestamp': times[::-1], 'f': [3., 2., np.nan]})
    result = module.attach_forecast(base, forecast, ['f'])
    pd.testing.assert_frame_equal(base, result[base.columns])
    np.testing.assert_allclose(result.f, [np.nan, 2., 3.], equal_nan=True)
    with pytest.raises(ValueError):
        module.attach_forecast(base, pd.concat([forecast, forecast.iloc[:1]]), ['f'])
    with pytest.raises(ValueError):
        module.attach_forecast(base, forecast.iloc[:2], ['f'])


def test_gate_uses_f024_not_easy_history_reference(module):
    frame = pd.DataFrame({'score_allowed': [True, True, False], 'exact_hours': [10., 30., 100.],
                          'continuity_block': [1, 2, 3], 'informative': [True] * 3,
                          'lower_bound_hours': [10., 30., 100.],
                          'forecast_aft': [11., 31., 100.],
                          'history_counts_aft': [20., 40., 100.]})
    result = module.compare_cohort(frame, np.array([12., 32., 0.]))
    assert not result['passed']
    assert result['candidate']['row_mae'] == 2.
    assert result['f024_control']['row_mae'] == 1.
    assert result['history_reference']['row_mae'] == 10.
    assert module.compare_cohort(frame, np.array([10., 30., 0.]))['passed']


def test_tail_regression_or_nonfinite_prediction_rejects(module):
    frame = pd.DataFrame({'score_allowed': [True, True], 'exact_hours': [10., 30.],
                          'continuity_block': [1, 2], 'informative': [True, True],
                          'lower_bound_hours': [10., 30.], 'forecast_aft': [20., 31.],
                          'history_counts_aft': [20., 40.]})
    assert not module.compare_cohort(frame, np.array([10., 32.]))['passed']
    with pytest.raises(ValueError):
        module.compare_cohort(frame, np.array([10., np.nan]))


def test_missing_control_prediction_cannot_pass_on_partial_rows(module):
    frame = pd.DataFrame({'score_allowed': [True, True], 'exact_hours': [10., 30.],
                          'continuity_block': [1, 2], 'informative': [True, True],
                          'lower_bound_hours': [10., 30.], 'forecast_aft': [np.nan, 31.],
                          'history_counts_aft': [20., 40.]})
    with pytest.raises(ValueError, match='control prediction'):
        module.compare_cohort(frame, np.array([10., 30.]))


def test_artifact_rejects_stale_source_and_output(module, tmp_path):
    source = tmp_path / 'source.bin'
    source.write_bytes(b'original')
    folder = tmp_path / 'artifact'
    folder.mkdir()
    for name in ['model.joblib', 'predictions.parquet', 'expanded_predictions.parquet']:
        (folder / name).write_bytes(b'fixture')
    (folder / 'protocol.json').write_text(json.dumps({
        'input_sha256': {str(source): module.sha(source)}}), encoding='utf-8')
    report = {'output_sha256': {p.name: module.sha(p) for p in folder.iterdir()}}
    (folder / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    module.verified_artifact(folder, {})
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='stale or changed input'):
        module.verified_artifact(folder, {})
    source.write_bytes(b'original')
    (folder / 'model.joblib').write_bytes(b'changed')
    with pytest.raises(ValueError, match='stale or changed input'):
        module.verified_artifact(folder, {})
