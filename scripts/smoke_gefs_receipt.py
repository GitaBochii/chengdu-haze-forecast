"""One GEFS index plus one bounded temperature message, linked to receipts."""

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

from haze_forecast.forecast_sources import parse_gfs_index, range_for_gfs_index_entry
from haze_forecast.grib import query_grib_point
from haze_forecast.operational_gefs import gefs_data_url
from haze_forecast.weather_receipts import receive_once

ROOT = Path(__file__).resolve().parents[1]


def checked_range(response, start, end, limit):
    if response.status_code != 206:
        raise ValueError('Server did not honor bounded Range request')
    if not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
        raise ValueError('Unexpected Content-Range')
    data = b''
    for chunk in response.iter_content(65536):
        data += chunk
        if len(data) > limit:
            raise ValueError('Oversized GRIB message')
    if len(data) != end - start + 1 or not data.startswith(b'GRIB') or not data.endswith(b'7777'):
        raise ValueError('Invalid GRIB length or framing')
    return data, dict(response.headers)


def main():
    now = datetime.now(UTC)
    run = now - timedelta(hours=12)
    run = run.replace(hour=(run.hour // 6) * 6, minute=0, second=0, microsecond=0)
    out = ROOT / 'artifacts' / f'gefs_receipt_smoke_{now:%Y%m%dT%H%M%SZ}'
    out.mkdir()
    report = {'status': 'started', 'run_at': run.isoformat(), 'lead_hours': 24,
              'product': 'geavg', 'field': 'TMP:2 m above ground',
              'deployment_authorized': False}
    try:
        url = gefs_data_url(run, product='geavg', lead_hours=24)
        index = receive_once(out, url + '.idx', max_bytes=1_000_000)
        report['index_receipt'] = index
        entries = parse_gfs_index(Path(index['object_path']).read_text(encoding='utf-8'))
        matches = [e for e in entries if e.variable == 'TMP' and e.level == '2 m above ground']
        if len(matches) != 1:
            raise ValueError('Ambiguous temperature field')
        start, end = range_for_gfs_index_entry(entries, matches[0])
        limit = 2_000_000
        if end - start + 1 > limit:
            raise ValueError('Message exceeds smoke-test size budget')
        report['requested_range'] = [start, end]

        def fetch(source, maximum):
            with requests.get(source, headers={'Range': f'bytes={start}-{end}'},
                              stream=True, timeout=30) as response:
                return checked_range(response, start, end, maximum)

        receipt = receive_once(out, url, max_bytes=limit, fetch=fetch)
        report['message_receipt'] = receipt
        message = Path(receipt['object_path'])
        decoder = ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'
        inventory = subprocess.run([str(decoder), str(message), '-s'], check=True,
                                   capture_output=True, text=True, timeout=30).stdout.strip()
        if (len(inventory.splitlines()) != 1 or f':d={run:%Y%m%d%H}:' not in inventory
                or ':TMP:2 m above ground:24 hour fcst:' not in inventory):
            raise ValueError(f'Decoded field identity mismatch: {inventory}')
        point = query_grib_point(message, 104.06, 30.67, decoder)
        if not 180 <= point.value <= 340:
            raise ValueError('Implausible temperature in Kelvin')
        completed = datetime.now(UTC)
        valid = run + timedelta(hours=24)
        report.update(status='decoded', inventory=inventory, valid_at=valid.isoformat(),
                      decoded_at=completed.isoformat(),
                      usable_after=completed.isoformat(),
                      still_future_at_decode=valid > completed,
                      point={'longitude': point.longitude, 'latitude': point.latitude,
                             'temperature_k': point.value, 'temperature_c': point.value - 273.15},
                      limitations=['Single ensemble-mean field, not complete model input',
                                   'Receipt and decode time do not prove provider first publication',
                                   'No model inference, accuracy evaluation or background collection'])
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'directory': str(out), **report}, indent=2), flush=True)


if __name__ == '__main__':
    main()
