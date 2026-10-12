"""Bounded historical forecast probe; no training or coverage claim."""

import json
import subprocess
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from haze_forecast.forecast_sources import parse_gfs_index, range_for_gfs_index_entry
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    out = ROOT / 'artifacts/task01_recovery_head/gefs_2021_probe_v1'
    if out.exists():
        raise FileExistsError(out)
    binary = ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    # First candidate initialization for the earliest 2021 origin, selected without labels.
    run = '20210102'
    definitions = {'geavg': ('pgrb2ap5', 'pgrb2a', 'TMP', '2 m above ground', 'ens mean'),
                   'gespr': ('pgrb2ap5', 'pgrb2a', 'TMP', '2 m above ground', 'ens std dev'),
                   'gec00': ('pgrb2bp5', 'pgrb2b', 'HPBL', 'surface', 'ENS=low-res ctl')}
    out.mkdir()
    results = []
    for product, (directory, suffix, variable, level, identity) in definitions.items():
        url = f'https://noaa-gefs-pds.s3.amazonaws.com/gefs.{run}/00/atmos/{directory}/{product}.t00z.{suffix}.0p50.f024'
        response = requests.get(url + '.idx', timeout=45)
        response.raise_for_status()
        (out / f'{product}.idx').write_bytes(response.content)
        entries = parse_gfs_index(response.text)
        matches = [e for e in entries if e.variable == variable and e.level == level]
        assert len(matches) == 1
        start, end = range_for_gfs_index_entry(entries, matches[0])
        response = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=45)
        assert response.status_code == 206
        assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        content = response.content
        assert len(content) == end-start+1 and content[:4] == b'GRIB' and content[-4:] == b'7777'
        assert content[7] == 2 and int.from_bytes(content[8:16], 'big') == len(content)
        path = out / f'{product}.grb2'
        path.write_bytes(content)
        inventory = subprocess.check_output([str(binary), str(path), '-s', '-vt', '-lon', '104.0665', '30.5723'],
                                            text=True, timeout=30).strip()
        assert 'd=2021010200:' in inventory and ':24 hour fcst:' in inventory
        assert f':{variable}:{level}:' in inventory and 'vt=2021010300:' in inventory and identity in inventory
        modified = parsedate_to_datetime(response.headers['Last-Modified']).astimezone(UTC)
        results.append({'product': product, 'url': url, 'inventory': inventory,
                        'last_modified': modified.isoformat(),
                        'before_first_origin_20210102T0800Z': modified <= datetime(2021, 1, 2, 8, tzinfo=UTC),
                        'byte_range': [start, end], 'etag': response.headers.get('ETag')})
    report = {'run_at': '2021-01-02T00:00:00+00:00', 'lead_hours': 24, 'products': results,
              'script_sha256': sha(Path(__file__)), 'decoder_sha256': sha(binary),
              'limitations': ['Three selected messages only; not nine-field or yearly coverage.',
                              'Archive Last-Modified proxy is not first-publication certification.'],
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
