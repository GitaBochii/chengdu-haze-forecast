"""Content-version receipts; receipt time never claims historical publication."""

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def _now():
    return datetime.now(UTC).isoformat()


def _fetch(url, limit):
    request = Request(url, headers={'User-Agent': 'Chengdu-weather-evidence/1.0'})
    with urlopen(request, timeout=30) as response:
        return response.read(limit + 1), dict(response.headers.items())


def receive_once(directory, url, *, max_bytes=10_000_000, fetch=_fetch):
    """Record one bounded request, retaining failures and content revisions.

    This records transport success, not valid weather fields or provider first
    publication. Downstream decoding and observation clocks need separate checks.
    """
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or max_bytes <= 0):
        raise ValueError('Require credential-free HTTPS URL without query/fragment and positive size limit')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    blobs = root / 'objects'
    blobs.mkdir(exist_ok=True)
    attempt = uuid.uuid4().hex
    with sqlite3.connect(root / 'receipts.sqlite3', timeout=30) as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, url TEXT NOT NULL, started_at TEXT NOT NULL,
                completed_at TEXT, status TEXT NOT NULL, sha256 TEXT,
                response_headers TEXT, error_type TEXT);
            CREATE TABLE IF NOT EXISTS versions (
                url TEXT NOT NULL, sha256 TEXT NOT NULL, first_received_at TEXT NOT NULL,
                first_attempt TEXT NOT NULL, PRIMARY KEY (url, sha256));
        ''')
        db.execute('INSERT INTO attempts(id,url,started_at,status) VALUES(?,?,?,?)',
                   (attempt, url, _now(), 'started'))
        db.commit()
        try:
            body, headers = fetch(url, max_bytes)
            if not isinstance(body, bytes) or not body or len(body) > max_bytes:
                raise ValueError('Empty, non-byte, or oversized response')
            digest = hashlib.sha256(body).hexdigest()
            path = blobs / digest
            # A failed/interrupted write cannot become a successful receipt.
            try:
                with path.open('xb') as stream:
                    stream.write(body)
            except FileExistsError:
                pass
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Stored object checksum mismatch')
            completed = _now()
            retained = {k.lower(): v for k, v in headers.items()
                        if k.lower() in {'last-modified', 'etag', 'content-type', 'content-length',
                                         'content-range', 'date'}}
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT OR IGNORE INTO versions VALUES(?,?,?,?)',
                       (url, digest, completed, attempt))
            db.execute('UPDATE attempts SET completed_at=?,status=?,sha256=?,response_headers=? WHERE id=?',
                       (completed, 'received_unvalidated', digest, json.dumps(retained), attempt))
            first = db.execute('SELECT first_received_at FROM versions WHERE url=? AND sha256=?',
                               (url, digest)).fetchone()[0]
            db.commit()
            return {'attempt': attempt, 'sha256': digest, 'bytes': len(body),
                    'first_received_at': first, 'completed_at': completed,
                    'status': 'received_unvalidated', 'object_path': str(path.resolve())}
        except Exception as exc:
            db.rollback()
            db.execute('UPDATE attempts SET completed_at=?,status=?,error_type=? WHERE id=?',
                       (_now(), 'failed', type(exc).__name__, attempt))
            db.commit()
            raise
