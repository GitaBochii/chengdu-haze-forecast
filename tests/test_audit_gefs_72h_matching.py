import importlib

import numpy as np
import pandas as pd
import pytest


def auditor():
    module = importlib.util.find_spec('scripts.audit_gefs_72h_matching')
    assert module is not None, 'independent matching auditor missing'
    return importlib.import_module('scripts.audit_gefs_72h_matching').verify_matches


def frames():
    run = pd.Timestamp('2022-01-01T18:00:00Z')
    known = pd.Timestamp('2022-01-01T22:16:06Z')
    valid = pd.Timestamp('2022-01-04T18:00:00Z')
    origins = pd.DatetimeIndex([valid, known, run])
    snapshots = pd.DataFrame([{'run_at': run, 'known_at': known, 'valid_at': valid, 'x': 7.0}])
    features = pd.DataFrame({
        'timestamp': origins, 'x': [np.nan, 7.0, np.nan],
        'run_at': [pd.NaT, run, pd.NaT], 'known_at': [pd.NaT, known, pd.NaT],
        'valid_at': [pd.NaT, valid, pd.NaT], 'forecast_known_age_hours': [np.nan, 0.0, np.nan],
        'forecast_run_age_hours': [np.nan, (known-run).total_seconds()/3600, np.nan],
        'forecast_remaining_lead_hours': [np.nan, (valid-known).total_seconds()/3600, np.nan],
    })
    return snapshots, features, origins


def test_valid_boundary_and_missing_rows():
    snapshots, features, origins = frames()
    assert auditor()(snapshots, features, origins, ['x']) == 3


@pytest.mark.parametrize('column,value', [('x', 8.0), ('forecast_run_age_hours', 1.0)])
def test_wrong_value_or_age_rejected(column, value):
    snapshots, features, origins = frames()
    features.loc[1, column] = value
    with pytest.raises(ValueError, match='mismatch'):
        auditor()(snapshots, features, origins, ['x'])


def test_missing_forecast_cannot_have_values():
    snapshots, features, origins = frames()
    features.loc[0, 'x'] = 7.0
    with pytest.raises(ValueError, match='missing'):
        auditor()(snapshots, features, origins, ['x'])


def test_row_loss_rejected():
    snapshots, features, origins = frames()
    with pytest.raises(ValueError, match='origin'):
        auditor()(snapshots, features.iloc[1:], origins, ['x'])


def test_expired_snapshot_excluded():
    snapshots, features, origins = frames()
    extra = snapshots.iloc[0].copy()
    extra['known_at'] = pd.Timestamp('2022-01-01T21:00:00Z')
    extra['valid_at'] = pd.Timestamp('2022-01-01T22:00:00Z')
    snapshots = pd.concat([snapshots, extra.to_frame().T], ignore_index=True)
    assert auditor()(snapshots, features, origins, ['x']) == 3


def test_future_snapshot_does_not_change_past_match():
    snapshots, features, origins = frames()
    extra = snapshots.iloc[0].copy()
    extra['known_at'] = pd.Timestamp('2022-01-05T00:00:00Z')
    extra['valid_at'] = pd.Timestamp('2022-01-07T00:00:00Z')
    extra['x'] = 900.0
    snapshots = pd.concat([extra.to_frame().T, snapshots], ignore_index=True)
    assert auditor()(snapshots, features, origins, ['x']) == 3


def test_missing_inputs_do_not_produce_audit(tmp_path, monkeypatch):
    auditor()
    module = importlib.import_module('scripts.audit_gefs_72h_matching')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError):
        module.main()
    assert not list(tmp_path.rglob('matching_audit.json'))
