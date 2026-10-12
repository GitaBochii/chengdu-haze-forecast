from datetime import datetime, timedelta, timezone
import hashlib
import sqlite3

import pytest

from haze_forecast.prospective_ledger import freeze_candidate, record_prediction, score_prediction


T = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)


def setup_ledger(tmp_path):
    freeze_candidate(tmp_path, 'baseline', {'model': 'frozen'}, clock=lambda: T)
    p = tmp_path / 'input.json'
    p.write_text('{}')
    inputs = [{'path': str(p), 'sha256': hashlib.sha256(b'{}').hexdigest(),
               'received_at': T.isoformat(), 'verified_at': T.isoformat()}]
    return inputs


def observations(origin=T):
    return [{'timestamp': (origin + timedelta(hours=h)).isoformat(), 'aqi': 90.0}
            for h in range(1, 25)]


def test_full_flow_and_append_only(tmp_path):
    inputs = setup_ledger(tmp_path)
    p = record_prediction(tmp_path, 'baseline', T, inputs,
                          {'origin_aqi': 130, 'recovery_probability': .8}, clock=lambda: T)
    s = score_prediction(tmp_path, p['id'], observations(), clock=lambda: T + timedelta(hours=24))
    assert s['labels'] == {'onset': False, 'recovery': True}
    assert s['scores']['recovery_brier'] == pytest.approx(.04)
    with pytest.raises(ValueError):
        score_prediction(tmp_path, p['id'], observations(), clock=lambda: T + timedelta(hours=25))
    with pytest.raises(ValueError):
        freeze_candidate(tmp_path, 'baseline', {}, clock=lambda: T)
    with pytest.raises(ValueError):
        record_prediction(tmp_path, 'baseline', T, inputs, {'origin_aqi': 130}, clock=lambda: T)
    with sqlite3.connect(tmp_path / 'ledger.sqlite3') as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute('DELETE FROM predictions')


@pytest.mark.parametrize('delta', [-1, 1])
def test_origin_outside_freeze_or_now(tmp_path, delta):
    inputs = setup_ledger(tmp_path)
    with pytest.raises(ValueError):
        record_prediction(tmp_path, 'baseline', T + timedelta(seconds=delta), inputs,
                          {'origin_aqi': 100}, clock=lambda: T)


def test_backfill_boundary(tmp_path):
    inputs = setup_ledger(tmp_path)
    with pytest.raises(ValueError):
        record_prediction(tmp_path, 'baseline', T, inputs, {'origin_aqi': 100},
                          clock=lambda: T + timedelta(minutes=5, microseconds=1))
    record_prediction(tmp_path, 'baseline', T, inputs, {'origin_aqi': 100},
                      clock=lambda: T + timedelta(minutes=5))


def test_non_hour_origin_rejected(tmp_path):
    inputs = setup_ledger(tmp_path)
    origin = T + timedelta(minutes=1)
    with pytest.raises(ValueError, match='exact UTC hour'):
        record_prediction(tmp_path, 'baseline', origin, inputs, {'origin_aqi': 100}, clock=lambda: origin)


def test_manifest_finite_and_timezone_required(tmp_path):
    with pytest.raises(ValueError):
        freeze_candidate(tmp_path, 'bad', {'metric': float('inf')}, clock=lambda: T)
    with pytest.raises(ValueError):
        freeze_candidate(tmp_path, 'bad', {}, clock=lambda: T.replace(tzinfo=None))


def test_five_hours_not_recovery(tmp_path):
    inputs = setup_ledger(tmp_path)
    p = record_prediction(tmp_path, 'baseline', T, inputs, {'origin_aqi': 101}, clock=lambda: T)
    rows = observations()
    for i, row in enumerate(rows): row['aqi'] = 101 if i % 6 == 5 else 100
    result = score_prediction(tmp_path, p['id'], rows, clock=lambda: T + timedelta(hours=24))
    assert result['labels']['recovery'] is False


@pytest.mark.parametrize('kind', ['hash', 'received', 'verified', 'naive', 'nan', 'probability'])
def test_reject_invalid_input(tmp_path, kind):
    inputs = setup_ledger(tmp_path)
    prediction = {'origin_aqi': 100}
    origin = T
    if kind == 'hash':
        (tmp_path / 'input.json').write_text('tampered')
    if kind == 'received':
        inputs[0]['received_at'] = (T + timedelta(seconds=1)).isoformat()
    if kind == 'verified':
        inputs[0]['verified_at'] = (T + timedelta(seconds=1)).isoformat()
    if kind == 'naive':
        origin = T.replace(tzinfo=None)
    if kind == 'nan':
        prediction['extra'] = float('nan')
    if kind == 'probability':
        prediction['onset_probability'] = 2
    with pytest.raises(ValueError):
        record_prediction(tmp_path, 'baseline', origin, inputs, prediction, clock=lambda: T)


@pytest.mark.parametrize('kind', ['early', 'missing', 'duplicate', 'nan', 'negative', 'naive', 'offset'])
def test_bad_labels(tmp_path, kind):
    inputs = setup_ledger(tmp_path)
    p = record_prediction(tmp_path, 'baseline', T, inputs, {'origin_aqi': 100}, clock=lambda: T)
    rows = observations()
    now = T + timedelta(hours=24)
    if kind == 'early': now -= timedelta(microseconds=1)
    if kind == 'missing': rows.pop()
    if kind == 'duplicate': rows[-1] = rows[0]
    if kind == 'nan': rows[0]['aqi'] = float('nan')
    if kind == 'negative': rows[0]['aqi'] = -1
    if kind == 'naive': rows[0]['timestamp'] = (T + timedelta(hours=1)).replace(tzinfo=None).isoformat()
    if kind == 'offset': rows[0]['timestamp'] = (T + timedelta(minutes=61)).isoformat()
    with pytest.raises(ValueError):
        score_prediction(tmp_path, p['id'], rows, clock=lambda: now)


def test_timezone_equivalence_and_last_six_hours(tmp_path):
    inputs = setup_ledger(tmp_path)
    origin = T.astimezone(timezone(timedelta(hours=8)))
    p = record_prediction(tmp_path, 'baseline', origin, inputs,
                          {'origin_aqi': 100, 'onset_probability': .2}, clock=lambda: T)
    rows = observations(origin)
    for row in rows[:18]: row['aqi'] = 101
    s = score_prediction(tmp_path, p['id'], rows, clock=lambda: T + timedelta(hours=24))
    assert s['labels'] == {'onset': True, 'recovery': True}
    assert s['scores']['onset_brier'] == pytest.approx(.64)
    assert 'recovery_brier' not in s['scores']


@pytest.mark.parametrize('threshold', [float('nan'), float('inf'), -0.01, 1.02])
def test_invalid_threshold_rejected(tmp_path, threshold):
    inputs = setup_ledger(tmp_path)
    with pytest.raises(ValueError):
        record_prediction(tmp_path, 'baseline', T, inputs,
                          {'origin_aqi': 100, 'onset_probability': .6, 'onset_threshold': threshold},
                          clock=lambda: T)


@pytest.mark.parametrize('origin_aqi,probability,threshold,high,expected', [
    (100, .6, .6, True, 'tp'),
    (100, .6, .5, False, 'fp'),
    (100, .6, 1.01, False, 'tn'),
    (100, .6, .8, True, 'fn'),
    (101, .6, .6, False, 'tp'),
    (101, .6, .5, True, 'fp'),
    (101, .6, 1.01, True, 'tn'),
    (101, .6, .8, False, 'fn'),
])
def test_only_applicable_task_confusion(tmp_path, origin_aqi, probability, threshold, high, expected):
    inputs = setup_ledger(tmp_path)
    task = 'onset' if origin_aqi <= 100 else 'recovery'
    prediction = {'origin_aqi': origin_aqi, 'onset_probability': probability,
                  'recovery_probability': probability, 'onset_threshold': threshold,
                  'recovery_threshold': threshold}
    p = record_prediction(tmp_path, 'baseline', T, inputs, prediction, clock=lambda: T)
    rows = observations()
    if high:
        for row in rows: row['aqi'] = 101
    s = score_prediction(tmp_path, p['id'], rows, clock=lambda: T + timedelta(hours=24))
    assert s['scores']['task'] == task
    assert s['scores']['decision'] == (probability >= threshold)
    assert {k: s['scores'][k] for k in ('tp', 'fp', 'tn', 'fn')} == {
        k: int(k == expected) for k in ('tp', 'fp', 'tn', 'fn')}
    assert task + '_brier' in s['scores']
    assert ('recovery' if task == 'onset' else 'onset') + '_brier' not in s['scores']
