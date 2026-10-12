import argparse
import json
import subprocess

import pytest

from scripts.resume_gefs_72h_collection import _cli_limit, resume


def state(done, attempted=(), total=10):
    return {'planned_runs': total, 'completed_runs': done, 'remaining_runs': total-done,
            'attempted': list(attempted)}


def complete(run):
    return {'run': str(run), 'status': 'complete'}


def failed(error):
    return {'run': 'failed-run', 'status': 'failed', 'error': error}


def harness(tmp_path, initial, outcomes):
    status = tmp_path / 'status.json'
    status.write_text(json.dumps(initial))
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        payload, code = outcomes[len(calls)-1]
        if payload is not None:
            status.write_text(json.dumps(payload))
        return subprocess.CompletedProcess(command, code, 'out', 'err')

    return status, calls, runner


def test_retry_uses_remaining_and_logs_every_attempt(tmp_path):
    status, calls, runner = harness(tmp_path, state(0), [
        (state(1, [complete(1), failed("SSLError('offline')")]), 0),
        (state(3, [complete(2), complete(3)]), 0),
    ])
    sleeps = []
    assert resume(status, tmp_path / 'logs', target_new_runs=3,
                  runner=runner, sleep=sleeps.append) == 0
    assert [c[-1] for c in calls] == ['3', '2']
    assert len(sleeps) == 1
    attempts = list((tmp_path / 'logs').glob('*/attempt-*'))
    assert len(attempts) == 2
    for attempt in attempts:
        assert (attempt / 'stdout.txt').read_text() == 'out'
        assert (attempt / 'stderr.txt').read_text() == 'err'
        assert json.loads((attempt / 'process.json').read_text())['exitcode'] == 0
        assert (attempt / 'status.json').exists()


@pytest.mark.parametrize('error', ["AssertionError('bad')", "HTTPError('404')",
                                  "ValueError('hash')", "ProtocolError('incomplete body')"])
def test_non_network_failures_stop(tmp_path, error):
    status, calls, runner = harness(tmp_path, state(0), [(state(0, [failed(error)]), 0)])
    assert resume(status, tmp_path / 'logs', runner=runner, sleep=lambda _: None) == 1
    assert len(calls) == 1


@pytest.mark.parametrize('payload,code', [
    (None, 0), (state(1, [complete(1)]), 2),
    (state(2, [complete(1)]), 0),
    ({'planned_runs': 10, 'completed_runs': 1, 'remaining_runs': 10, 'attempted': []}, 0),
    (state(1, [complete(1)]), 0),
])
def test_stale_inconsistent_or_nonzero_stops(tmp_path, payload, code):
    status, calls, runner = harness(tmp_path, state(0), [(payload, code)])
    assert resume(status, tmp_path / 'logs', target_new_runs=3,
                  runner=runner, sleep=lambda _: None) == 1
    assert len(calls) == 1


def test_exhaustion_is_failure(tmp_path):
    status, calls, runner = harness(tmp_path, state(0), [
        (state(0, [failed("ReadTimeout('offline')")]), 0),
        (state(0, [failed("ReadTimeout('offline-again')")]), 0),
    ])
    assert resume(status, tmp_path / 'logs', max_attempts=2,
                  runner=runner, sleep=lambda _: None) == 1
    assert len(calls) == 2


def test_finish_population_early(tmp_path):
    status, calls, runner = harness(tmp_path, state(9), [(state(10, [complete(10)]), 0)])
    assert resume(status, tmp_path / 'logs', runner=runner) == 0
    assert calls[0][-1] == '1'


@pytest.mark.parametrize('kwargs', [{'target_new_runs': -1}, {'max_attempts': -1},
                                   {'target_new_runs': True}, {'max_attempts': 1.5}])
def test_invalid_limits(tmp_path, kwargs):
    with pytest.raises(ValueError):
        resume(tmp_path / 'missing', tmp_path / 'logs', **kwargs)


def test_zero_attempts_does_not_claim_success(tmp_path):
    status, calls, runner = harness(tmp_path, state(0), [])
    assert resume(status, tmp_path / 'logs', max_attempts=0, runner=runner) == 1
    assert calls == []


@pytest.mark.parametrize('kind', ['SSLError', 'ProxyError', 'ConnectionError',
                                  'ReadTimeout', 'ConnectTimeout'])
def test_network_allowlist(tmp_path, kind):
    status, calls, runner = harness(tmp_path, state(0), [
        (state(0, [failed(f"{kind}('offline')")]), 0),
        (state(1, [complete(1)]), 0),
    ])
    assert resume(status, tmp_path / 'logs', target_new_runs=1,
                  runner=runner, sleep=lambda _: None) == 0
    assert len(calls) == 2


def test_chunked_body_interruption_retries(tmp_path):
    error = ("ChunkedEncodingError(ProtocolError('Connection broken: "
             "IncompleteRead(86611 bytes read, 93986 more expected)', "
             "IncompleteRead(86611 bytes read, 93986 more expected)))")
    status, calls, runner = harness(tmp_path, state(0), [
        (state(0, [failed(error)]), 0),
        (state(1, [complete(1)]), 0),
    ])
    sleeps = []
    assert resume(status, tmp_path / 'logs', target_new_runs=1,
                  runner=runner, sleep=sleeps.append) == 0
    assert len(calls) == 2
    assert sleeps == [5]


def test_logs_never_overwritten(tmp_path):
    status, _, runner = harness(tmp_path, state(0), [])
    for _ in range(2):
        assert resume(status, tmp_path / 'logs', target_new_runs=0, runner=runner) == 0
    assert len(list((tmp_path / 'logs').iterdir())) == 2


def test_launch_failure_is_logged_without_retry(tmp_path):
    status, _, _ = harness(tmp_path, state(0), [])

    def runner(*args, **kwargs):
        raise OSError('cannot launch')

    assert resume(status, tmp_path / 'logs', runner=runner) == 1
    logs = list((tmp_path / 'logs').glob('*/attempt-*'))
    assert len(logs) == 1
    assert json.loads((logs[0] / 'process.json').read_text())['launch_error'].startswith(
        'OSError(')
    assert (logs[0] / 'status.json').read_bytes() == status.read_bytes()


@pytest.mark.parametrize('value', ['-1', '1.5', 'abc', '10001'])
def test_cli_rejects_invalid_limits(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _cli_limit(value)


@pytest.mark.parametrize('value', ['0', '20', '10000'])
def test_cli_accepts_bounded_nonnegative_integers(value):
    assert _cli_limit(value) == int(value)
