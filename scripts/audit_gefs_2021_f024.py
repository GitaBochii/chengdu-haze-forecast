"""Independently decode local 2021 f024 batches, without authorizing training."""

import hashlib
import json
import math
import re
import subprocess
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

MEAN_FIELDS = {'temperature_c': ('TMP', '2 m above ground'),
               'relative_humidity_pct': ('RH', '2 m above ground'),
               'wind_u_ms': ('UGRD', '10 m above ground'),
               'wind_v_ms': ('VGRD', '10 m above ground')}
FIELDS = {'geavg': MEAN_FIELDS, 'gespr': MEAN_FIELDS,
          'gec00': {'boundary_layer_height_m': ('HPBL', 'surface')}}
IDENTITIES = {'geavg': 'ens mean', 'gespr': 'ens std dev', 'gec00': 'ENS=low-res ctl'}
HELPERS = {'src/haze_forecast/forecast_sources.py', 'src/haze_forecast/grib.py',
           'scripts/benchmark_observed_business_aft.py'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aware(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, 'timezone missing')
    return result


def reconstruct_runs(frame):
    runs = set()
    for value in frame['timestamp']:
        timestamp = pd.Timestamp(value)
        require(timestamp.tzinfo is not None, 'naive origin')
        if timestamp.tz_convert('Asia/Shanghai').year != 2021:
            continue
        lagged = timestamp.to_pydatetime().astimezone(UTC) - timedelta(hours=4)
        runs.add(lagged.replace(hour=lagged.hour // 6 * 6, minute=0, second=0, microsecond=0))
    return [run.isoformat() for run in sorted(runs)]


def product_url(run, product):
    directory, suffix = ('pgrb2bp5', 'pgrb2b') if product == 'gec00' else ('pgrb2ap5', 'pgrb2a')
    return (f'https://noaa-gefs-pds.s3.amazonaws.com/gefs.{run:%Y%m%d}/{run:%H}/atmos/'
            f'{directory}/{product}.t{run:%H}z.{suffix}.0p50.f024')


def verify_protocol(root, base, binary, source):
    protocol = json.loads((base / 'protocol.json').read_bytes())
    require(digest(source) == protocol['source_sha256'], 'source hash mismatch')
    require(digest(binary) == protocol['decoder_sha256'], 'decoder hash mismatch')
    require(digest(root / 'scripts/collect_gefs_2021_f024.py') == protocol['collector_sha256'],
            'collector hash mismatch')
    helpers = {Path(name).as_posix(): value for name, value in protocol['helper_sha256'].items()}
    require(set(helpers) == HELPERS, 'helper set mismatch')
    for name, value in helpers.items():
        require(digest(root / name) == value, 'helper hash mismatch')
    runs = reconstruct_runs(pd.read_parquet(source))
    require(len(runs) == 101 and runs == protocol['runs'], 'planned run population mismatch')
    require(protocol['lead_hours'] == 24, 'protocol lead mismatch')
    return runs


def audit_run(marker, binary, planned):
    initial_hash = digest(marker)
    record = json.loads(marker.read_bytes())
    run = aware(record['run_at'])
    valid = run + timedelta(hours=24)
    require(record['run_at'] in planned, 'unplanned run')
    require(run.utcoffset() == timedelta(0), 'run must be UTC')
    require(marker.parent.name == run.strftime('%Y%m%d%H'), 'directory/run mismatch')
    require(aware(record['valid_at']) == valid, 'valid time mismatch')
    require(set(record['products']) == set(FIELDS), 'product set mismatch')
    expected = {f'{product}.idx' for product in FIELDS} | {
        f'{product}_{field}.grb2' for product, fields in FIELDS.items() for field in fields}
    require(set(record['output_sha256']) == expected, 'output manifest set mismatch')
    require({p.name for p in marker.parent.iterdir()} == expected | {'complete.json'}, 'file set mismatch')
    for name in expected:
        require(digest(marker.parent / name) == record['output_sha256'][name], 'output hash mismatch')
    dates = []
    for product, fields in FIELDS.items():
        data = record['products'][product]
        require(data['url'] == product_url(run, product), 'product URL mismatch')
        require(set(data['fields']) == set(fields), 'field set mismatch')
        lines = [(line.split(':')) for line in (marker.parent / f'{product}.idx').read_text().splitlines() if line]
        offsets = [int(line[1]) for line in lines]
        require(offsets == sorted(set(offsets)), 'index offsets not strictly ordered')
        etags, product_dates = set(), set()
        for field, (variable, level) in fields.items():
            item = data['fields'][field]
            matches = [i for i, line in enumerate(lines) if len(line) > 4 and line[3:5] == [variable, level]]
            require(len(matches) == 1 and matches[0] + 1 < len(lines), 'index variable/range mismatch')
            index = matches[0]
            require(item['byte_range'] == [offsets[index], offsets[index + 1] - 1], 'byte range mismatch')
            path = marker.parent / f'{product}_{field}.grb2'
            content = path.read_bytes()
            require(digest(path) == item['sha256'], 'field hash mismatch')
            require(len(content) >= 20 and len(content) == offsets[index + 1] - offsets[index],
                    'range length mismatch')
            require(content[:4] == b'GRIB' and content[-4:] == b'7777' and content[7] == 2
                    and int.from_bytes(content[8:16], 'big') == len(content), 'invalid GRIB framing')
            output = subprocess.check_output(
                [str(binary), str(path), '-s', '-vt', '-lon', '104.0665', '30.5723'], text=True, timeout=30)
            require(len(output.strip().splitlines()) == 1, 'expected exactly one decoded message')
            require(f'd={run:%Y%m%d%H}:' in output and f'vt={valid:%Y%m%d%H}:' in output,
                    'decoded initialization/valid mismatch')
            require(f':{variable}:{level}:24 hour fcst:' in output, 'decoded field/lead mismatch')
            require(IDENTITIES[product] in output, 'decoded product identity mismatch')
            matches = re.findall(r'lon=([^,]+),lat=([^,]+),val=([^\s]+)', output)
            require(len(matches) == 1, 'decoded point missing or duplicated')
            longitude, latitude, value = map(float, matches[0])
            if product == 'geavg' and field == 'temperature_c':
                value -= 273.15
            require(math.isfinite(value) and math.isclose(value, item['value'], rel_tol=0, abs_tol=1e-8),
                    'decoded value mismatch')
            require((longitude, latitude) == (item['longitude'], item['latitude']) == (104.0, 30.5),
                    'decoded grid mismatch')
            if product == 'gespr' or field == 'boundary_layer_height_m':
                require(value >= 0, 'negative spread or PBL')
            if product == 'geavg' and field == 'relative_humidity_pct':
                require(0 <= value <= 100, 'RH outside physical range')
            require(bool(item['etag']), 'missing ETag')
            etags.add(item['etag'])
            modified = aware(item['last_modified'])
            product_dates.add(modified)
            dates.append(modified)
        require(len(etags) == len(product_dates) == 1, 'product headers inconsistent')
    require(max(dates) == aware(record['archive_available_at_proxy']), 'availability proxy mismatch')
    require(digest(marker) == initial_hash, 'completion marker changed during audit')
    return {'run_at': record['run_at'], 'complete_sha256': initial_hash}


def audit_completed(base, binary, runs):
    verified = [audit_run(marker, binary, runs) for marker in sorted(base.glob('*/complete.json'))]
    require(len({item['run_at'] for item in verified}) == len(verified), 'duplicate completed runs')
    return {'verified_runs': len(verified), 'planned_runs': len(runs),
            'verified_grib_messages': 9 * len(verified),
            'all_planned_runs_verified': len(verified) == len(runs),
            'formal_model_evaluation_allowed': False, 'model_permission': False,
            'protocol_sha256': digest(base / 'protocol.json'),
            'audit_script_sha256': digest(Path(__file__)), 'verified': verified,
            'limitations': ['Local recorded HTTP header consistency only; headers were not re-requested.',
                            'Archive Last-Modified is not first-publication certification.',
                            'Causal origin matching and model eligibility require separate validation.']}


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / 'artifacts/task01_recovery_head/gefs_2021_f024_collection_v1'
    source = root / 'artifacts/task01_recovery_head/interval_business_labels_v1/origins.parquet'
    binary = root / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    runs = verify_protocol(root, base, binary, source)
    report = audit_completed(base, binary, runs)
    require(verify_protocol(root, base, binary, source) == runs, 'protocol changed during audit')
    path = base / 'independent_audit.json'
    temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(path)
    print(json.dumps({key: value for key, value in report.items() if key != 'verified'}, indent=2))


if __name__ == '__main__':
    main()
