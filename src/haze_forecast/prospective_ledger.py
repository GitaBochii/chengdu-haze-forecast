"""Local append-only prospective evidence; not proof of historical publication."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def _time(value):
    result = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('Timezone-aware timestamp required')
    return result.astimezone(timezone.utc)


def _now(clock):
    return _time(clock() if clock else datetime.now(timezone.utc))


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _aqi(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('AQI must be finite and nonnegative')
    return value


def _db(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / 'ledger.sqlite3')
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS candidates (
            name TEXT PRIMARY KEY, frozen_at TEXT NOT NULL,
            manifest TEXT NOT NULL, sha256 TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY, candidate TEXT NOT NULL REFERENCES candidates(name),
            origin TEXT NOT NULL, recorded_at TEXT NOT NULL,
            inputs TEXT NOT NULL, prediction TEXT NOT NULL, sha256 TEXT NOT NULL,
            UNIQUE(candidate, origin));
        CREATE TABLE IF NOT EXISTS scores (
            prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id), scored_at TEXT NOT NULL,
            observations TEXT NOT NULL, sha256 TEXT NOT NULL,
            labels TEXT NOT NULL, scores TEXT NOT NULL);
    ''')
    for table in ('candidates', 'predictions', 'scores'):
        for action in ('UPDATE', 'DELETE'):
            db.execute(f'''CREATE TRIGGER IF NOT EXISTS no_{table}_{action}
                BEFORE {action} ON {table} BEGIN
                SELECT RAISE(ABORT, 'append-only ledger'); END''')
    db.commit()
    return db


def freeze_candidate(root, name, manifest, *, clock=None):
    """Freeze a unique candidate manifest with local receipt-time provenance."""
    if not isinstance(name, str) or not name.strip() or not isinstance(manifest, dict):
        raise ValueError('Candidate name and manifest object required')
    payload = _json(manifest)
    result = dict(name=name, frozen_at=_now(clock).isoformat(), manifest=manifest, sha256=_hash(payload))
    db = _db(root)
    try:
        with db:
            db.execute('INSERT INTO candidates VALUES (?, ?, ?, ?)',
                       (name, result['frozen_at'], payload, result['sha256']))
    except sqlite3.IntegrityError as exc:
        raise ValueError('Candidate already frozen') from exc
    finally:
        db.close()
    return result


def record_prediction(root, candidate, origin, inputs, prediction, *, clock=None):
    """Record only current-hour forecasts with input verification preceding origin."""
    origin = _time(origin)
    now = _now(clock)
    if origin.minute or origin.second or origin.microsecond:
        raise ValueError('Origin must be an exact UTC hour')
    if origin > now or now - origin > timedelta(minutes=5):
        raise ValueError('Origin is future or more than five minutes old')
    if not isinstance(prediction, dict):
        raise ValueError('Prediction object required')
    _aqi(prediction.get('origin_aqi'))
    for key in ('onset_probability', 'recovery_probability'):
        if key in prediction and not 0 <= _aqi(prediction[key]) <= 1:
            raise ValueError('Probability must be in [0, 1]')
    for key in ('onset_threshold', 'recovery_threshold'):
        if key in prediction and not 0 <= _aqi(prediction[key]) <= 1.01:
            raise ValueError('Threshold must be in [0, 1.01]')
    prediction_json = _json(prediction)
    if not isinstance(inputs, list) or not inputs:
        raise ValueError('At least one verified input required')
    verified = []
    for item in inputs:
        received_at, verified_at = _time(item['received_at']), _time(item['verified_at'])
        if not received_at <= verified_at <= origin:
            raise ValueError('Require received_at <= verified_at <= origin')
        path = Path(item['path']).resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise ValueError('Input SHA256 mismatch')
        verified.append(dict(path=str(path), sha256=digest,
                             received_at=received_at.isoformat(), verified_at=verified_at.isoformat()))
    inputs_json = _json(verified)
    result = dict(candidate=candidate, origin=origin.isoformat(), recorded_at=now.isoformat(),
                  inputs=verified, prediction=prediction)
    result['sha256'] = _hash(_json(result))
    db = _db(root)
    try:
        with db:
            frozen = db.execute('SELECT frozen_at FROM candidates WHERE name=?', (candidate,)).fetchone()
            if frozen is None or origin < _time(frozen['frozen_at']):
                raise ValueError('Unknown candidate or origin precedes freeze')
            cur = db.execute('''INSERT INTO predictions
                (candidate, origin, recorded_at, inputs, prediction, sha256) VALUES (?, ?, ?, ?, ?, ?)''',
                (candidate, result['origin'], result['recorded_at'], inputs_json, prediction_json, result['sha256']))
            result['id'] = cur.lastrowid
    except sqlite3.IntegrityError as exc:
        raise ValueError('Prediction already recorded') from exc
    finally:
        db.close()
    return result


def score_prediction(root, prediction_id, observations, *, clock=None):
    """Score complete exact-hour outcomes once; unavailable probabilities stay unscored."""
    now = _now(clock)
    db = _db(root)
    try:
        row = db.execute('SELECT * FROM predictions WHERE id=?', (prediction_id,)).fetchone()
        if row is None:
            raise ValueError('Unknown prediction')
        origin = _time(row['origin'])
        if now < origin + timedelta(hours=24):
            raise ValueError('Outcome window not mature')
        if len(observations) != 24:
            raise ValueError('Exactly 24 future hourly observations required')
        normalized = sorted([dict(timestamp=_time(o['timestamp']).isoformat(), aqi=_aqi(o['aqi']))
                             for o in observations], key=lambda o: o['timestamp'])
        expected = [(origin + timedelta(hours=h)).isoformat() for h in range(1, 25)]
        if [o['timestamp'] for o in normalized] != expected:
            raise ValueError('Observations must exactly match origin+1h through +24h')
        if any(_time(o['timestamp']) > now for o in normalized):
            raise ValueError('Observations cannot be from the future')
        values = [o['aqi'] for o in normalized]
        labels = dict(onset=any(v > 100 for v in values),
                      recovery=any(all(v <= 100 for v in values[i:i+6]) for i in range(19)))
        prediction = json.loads(row['prediction'])
        task = 'onset' if prediction['origin_aqi'] <= 100 else 'recovery'
        scores = {}
        if task + '_probability' in prediction:
            scores[task + '_brier'] = (prediction[task + '_probability'] - labels[task]) ** 2
            if task + '_threshold' in prediction:
                decision = prediction[task + '_probability'] >= prediction[task + '_threshold']
                actual = labels[task]
                scores.update(task=task, decision=decision,
                              tp=int(decision and actual), fp=int(decision and not actual),
                              tn=int(not decision and not actual), fn=int(not decision and actual))
        payload = _json(normalized)
        result = dict(prediction_id=prediction_id, scored_at=now.isoformat(),
                      sha256=_hash(payload), labels=labels, scores=scores)
        with db:
            db.execute('INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?)',
                       (prediction_id, result['scored_at'], payload, result['sha256'], _json(labels), _json(scores)))
    except sqlite3.IntegrityError as exc:
        raise ValueError('Prediction already scored') from exc
    finally:
        db.close()
    return result
