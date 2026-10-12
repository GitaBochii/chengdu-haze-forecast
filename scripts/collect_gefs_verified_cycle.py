"""Collect and validate one complete f024 research-input snapshot, no inference."""

import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

from haze_forecast.forecast_sources import parse_gfs_index, range_for_gfs_index_entry
from haze_forecast.grib import query_grib_point
from haze_forecast.operational_gefs import GEFS_FIELD_DEFINITIONS, gefs_data_url
from haze_forecast.operational_gefs_dataset import _FIELD_OUTPUT_NAMES, _converted_value
from haze_forecast.weather_receipts import receive_once
from scripts.smoke_gefs_receipt import checked_range

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('temperature_2m', 'relative_humidity_2m', 'u_wind_10m', 'v_wind_10m')


def validate_inventory(text, run, product, field):
    variable, level = GEFS_FIELD_DEFINITIONS[field]
    if (len(text.splitlines()) != 1 or f':d={run:%Y%m%d%H}:' not in text
            or f':{variable}:{level}:24 hour fcst:' not in text):
        raise ValueError('Run, variable, level or forecast lead mismatch')
    if product == 'geavg' and 'ens mean' not in text:
        raise ValueError('Expected ensemble mean')
    if product == 'gespr' and not any(s in text for s in ('ens spread', 'ens std dev')):
        raise ValueError('Expected ensemble spread')
    if product == 'gec00' and 'ctl' not in text:
        raise ValueError('Expected control member')


def validate_value(product, field, value):
    if not math.isfinite(value):
        raise ValueError('Nonfinite field')
    if product == 'gespr':
        bounds = (0, 100)
    elif field == 'temperature_2m':
        bounds = (180, 340)
    elif field == 'relative_humidity_2m':
        bounds = (0, 100)
    elif field == 'pblh':
        bounds = (0, 15000)
    else:
        bounds = (-150, 150)
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError('Field outside physical sanity limits')


def main():
    now = datetime.now(UTC)
    run = now - timedelta(hours=12)
    run = run.replace(hour=(run.hour // 6) * 6, minute=0, second=0, microsecond=0)
    out = ROOT / 'artifacts' / f'gefs_verified_cycle_{now:%Y%m%dT%H%M%SZ}'
    out.mkdir()
    ledger = ROOT / 'artifacts/weather_receipts'
    decoder = ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    report = {'status': 'collecting', 'run_at': run.isoformat(),
              'valid_at': (run + timedelta(hours=24)).isoformat(),
              'messages': [], 'indexes': [], 'features': {}, 'deployment_authorized': False,
              'ledger': str(ledger), 'decoder_sha256': hashlib.sha256(decoder.read_bytes()).hexdigest()}
    grid = None
    try:
        for product in ('geavg', 'gespr', 'gec00'):
            url = gefs_data_url(run, product=product, lead_hours=24)
            index = receive_once(ledger, url + '.idx', max_bytes=1_000_000)
            report['indexes'].append({'product': product, 'source_url': url + '.idx', **index})
            entries = parse_gfs_index(Path(index['object_path']).read_text(encoding='utf-8'))
            for field in (('pblh',) if product == 'gec00' else FIELDS):
                variable, level = GEFS_FIELD_DEFINITIONS[field]
                matched = [e for e in entries if e.variable == variable and e.level == level]
                if len(matched) != 1:
                    raise ValueError('Missing or ambiguous index field')
                start, end = range_for_gfs_index_entry(entries, matched[0])
                if end - start + 1 > 2_000_000:
                    raise ValueError('Message exceeds download budget')

                def fetch(source, maximum, start=start, end=end):
                    with requests.get(source, headers={'Range': f'bytes={start}-{end}'},
                                      stream=True, timeout=30) as response:
                        return checked_range(response, start, end, maximum)

                receipt = receive_once(ledger, url, max_bytes=2_000_000, fetch=fetch)
                path = Path(receipt['object_path'])
                inventory = subprocess.run([str(decoder), str(path), '-s'], check=True,
                                           capture_output=True, text=True, timeout=30).stdout.strip()
                record = {'product': product, 'field': field, 'source_url': url,
                          'byte_range': [start, end], 'inventory': inventory, 'receipt': receipt}
                report['messages'].append(record)
                validate_inventory(inventory, run, product, field)
                point = query_grib_point(path, 104.06, 30.67, decoder)
                validate_value(product, field, point.value)
                if grid is None:
                    grid = (point.longitude, point.latitude)
                if grid != (point.longitude, point.latitude):
                    raise ValueError('Different nearest grid points across fields')
                if field == 'pblh':
                    name = 'gefs_control_boundary_layer_height_m_f024'
                    value = point.value
                else:
                    stat = 'mean' if product == 'geavg' else 'spread'
                    name = f'gefs_{stat}_{_FIELD_OUTPUT_NAMES[field]}_f024'
                    value = _converted_value(field, stat, point.value)
                report['features'][name] = value
                record.update(raw_value=point.value, feature_name=name, converted_value=value,
                              grid_longitude=point.longitude, grid_latitude=point.latitude)
                print(f'Checked {product} {field}: {value}', flush=True)
        completed = datetime.now(UTC)
        if len(report['features']) != 9 or completed >= run + timedelta(hours=24):
            raise ValueError('Incomplete or expired snapshot')
        report.update(status='validated', usable_after=completed.isoformat(), grid=grid,
                      limitations=['Single cycle; no ongoing availability guarantee',
                                   'Nine weather fields only, not complete 104-feature inference',
                                   'Receipt does not prove provider first publication'])
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Report: {out / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
