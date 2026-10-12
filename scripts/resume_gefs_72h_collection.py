"""Synchronously resume the frozen collector with bounded network-only retries.

Collection success is not permission to train or deploy a model.
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NETWORK_ERRORS = ('SSLError(', 'ProxyError(', 'ConnectionError(',
                  'ReadTimeout(', 'ConnectTimeout(', 'ChunkedEncodingError(')


def _limit(value):
    if type(value) is not int or not 0 <= value <= 10000:
        raise ValueError('limits must be integers between 0 and 10000')
    return value


def _snapshot(path):
    data = path.read_bytes()
    return data, path.stat().st_mtime_ns


def _status(data):
    value = json.loads(data)
    if not isinstance(value, dict):
        raise TypeError('status must be an object')
    counts = [value.get(k) for k in ('planned_runs', 'completed_runs', 'remaining_runs')]
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('invalid status counts')
    if counts[0] != counts[1] + counts[2] or not isinstance(value.get('attempted'), list):
        raise ValueError('incoherent status')
    return value


def _validate_attempt(before, after, requested):
    attempted = after['attempted']
    if after['planned_runs'] != before['planned_runs'] or not 0 < len(attempted) <= requested:
        raise ValueError('population changed or invalid attempted count')
    runs = []
    successes = 0
    error = None
    for i, record in enumerate(attempted):
        if not isinstance(record, dict) or not isinstance(record.get('run'), str):
            raise TypeError('invalid attempted record')
        runs.append(record['run'])
        if record.get('status') == 'complete':
            successes += 1
        elif record.get('status') == 'failed' and i == len(attempted)-1:
            error = record.get('error')
            if not isinstance(error, str):
                raise ValueError('missing failure error')
        else:
            raise ValueError('invalid completion sequence')
    if len(set(runs)) != len(runs):
        raise ValueError('duplicate attempted run')
    if after['completed_runs'] - before['completed_runs'] != successes:
        raise ValueError('attempted successes disagree with progress')
    if error is None and successes != requested:
        raise ValueError('collector stopped without a recorded failure')
    return error


def resume(status_path, log_root, *, target_new_runs=20, max_attempts=5,
           runner=subprocess.run, sleep=time.sleep):
    """Return 0 only on requested progress or full population completion."""
    _limit(target_new_runs)
    _limit(max_attempts)
    status_path, log_root = Path(status_path), Path(log_root)
    name = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex
    log = log_root / name
    log.mkdir(parents=True, exist_ok=False)
    outcome = {'collection_only': True, 'model_evaluation_permission': False,
               'target_new_runs': target_new_runs, 'max_attempts': max_attempts}
    result = 1
    try:
        before_bytes, before_mtime = _snapshot(status_path)
        (log / 'initial_status.json').write_bytes(before_bytes)
        before = _status(before_bytes)
        target = min(before['planned_runs'], before['completed_runs'] + target_new_runs)
        for index in range(max_attempts):
            if before['completed_runs'] >= target:
                break
            requested = target - before['completed_runs']
            command = [sys.executable, '-m', 'scripts.extract_gefs_72h_batch',
                       '--max-runs', str(requested)]
            attempt = log / f'attempt-{index+1:03d}'
            attempt.mkdir()
            process = {'command': command, 'exitcode': None}
            stdout, stderr = '', ''
            try:
                child = runner(command, cwd=ROOT, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', check=False)
                stdout, stderr = child.stdout, child.stderr
                process['exitcode'] = child.returncode
            except OSError as exc:
                process['launch_error'] = repr(exc)
            finally:
                (attempt / 'stdout.txt').write_text(stdout or '', encoding='utf-8')
                (attempt / 'stderr.txt').write_text(stderr or '', encoding='utf-8')
                (attempt / 'process.json').write_text(json.dumps(process, indent=2),
                                                       encoding='utf-8')
                if status_path.exists():
                    (attempt / 'status.json').write_bytes(status_path.read_bytes())
                else:
                    (attempt / 'status_missing.txt').write_text('status.json missing')
            if process['exitcode'] != 0:
                raise ValueError('collector exited nonzero or failed to launch')
            after_bytes, after_mtime = _snapshot(status_path)
            if after_bytes == before_bytes and after_mtime == before_mtime:
                raise ValueError('collector did not refresh status')
            after = _status(after_bytes)
            error = _validate_attempt(before, after, requested)
            before, before_bytes, before_mtime = after, after_bytes, after_mtime
            if error is not None:
                if not error.startswith(NETWORK_ERRORS):
                    raise ValueError(f'non-retryable collector failure: {error}')
                if index + 1 < max_attempts:
                    sleep(min(60, 5 * 2 ** index))
        result = int(before['completed_runs'] < target)
        outcome['completed_runs'] = before['completed_runs']
        outcome['required_completed_runs'] = target
    except (OSError, ValueError, TypeError, KeyError) as exc:
        outcome['error'] = repr(exc)
    outcome['exitcode'] = result
    (log / 'result.json').write_text(json.dumps(outcome, indent=2), encoding='utf-8')
    print(json.dumps({'log_directory': str(log), **outcome}, indent=2))
    return result


def _cli_limit(text):
    try:
        return _limit(int(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-new-runs', type=_cli_limit, default=20)
    parser.add_argument('--max-attempts', type=_cli_limit, default=5)
    args = parser.parse_args()
    base = ROOT / 'artifacts/task01_recovery_head'
    return resume(base / 'gefs_72h_collection_v1/status.json',
                  base / 'gefs_72h_collection_retry_logs',
                  target_new_runs=args.target_new_runs, max_attempts=args.max_attempts)


if __name__ == '__main__':
    raise SystemExit(main())
