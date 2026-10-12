import sqlite3
from pathlib import Path

import pytest

from haze_forecast.weather_receipts import receive_once


def test_same_bytes_keep_first_receipt_but_revision_gets_new_version(tmp_path):
    def fetch(_url, _limit):
        return b'weather-v1', {'Last-Modified': 'yesterday'}

    a = receive_once(tmp_path, 'https://example.org/weather', fetch=fetch)
    b = receive_once(tmp_path, 'https://example.org/weather', fetch=fetch)
    c = receive_once(tmp_path, 'https://example.org/weather', fetch=lambda *_: (b'weather-v2', {}))
    assert a['sha256'] == b['sha256'] != c['sha256']
    assert a['first_received_at'] == b['first_received_at']
    assert a['first_received_at'] != 'yesterday'
    with sqlite3.connect(tmp_path / 'receipts.sqlite3') as db:
        assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 3
        assert db.execute('SELECT COUNT(*) FROM versions').fetchone()[0] == 2


def test_failure_persists_without_creating_successful_version(tmp_path):
    def fail(*_):
        raise TimeoutError('timed out')

    with pytest.raises(TimeoutError):
        receive_once(tmp_path, 'https://example.org/weather', fetch=fail)
    with sqlite3.connect(tmp_path / 'receipts.sqlite3') as db:
        assert db.execute('SELECT status FROM attempts').fetchone()[0] == 'failed'
        assert db.execute('SELECT COUNT(*) FROM versions').fetchone()[0] == 0


def test_oversized_response_and_unsafe_url_rejected(tmp_path):
    with pytest.raises(ValueError):
        receive_once(tmp_path, 'https://example.org/weather', max_bytes=2,
                     fetch=lambda *_: (b'123', {}))
    with pytest.raises(ValueError):
        receive_once(tmp_path, 'https://user:secret@example.org/weather')


def test_corrupt_existing_object_cannot_become_new_success(tmp_path):
    fetch = lambda *_: (b'weather-v1', {})
    receipt = receive_once(tmp_path, 'https://example.org/weather', fetch=fetch)
    Path(receipt['object_path']).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        receive_once(tmp_path, 'https://example.org/weather', fetch=fetch)
    with sqlite3.connect(tmp_path / 'receipts.sqlite3') as db:
        assert db.execute("SELECT COUNT(*) FROM attempts WHERE status='failed'").fetchone()[0] == 1
