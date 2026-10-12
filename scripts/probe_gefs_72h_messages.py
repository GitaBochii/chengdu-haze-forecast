"""Bounded single-run f072 extraction with independent product-header evidence."""

import json
import subprocess
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from haze_forecast.forecast_sources import parse_gfs_index, range_for_gfs_index_entry
from haze_forecast.grib import query_grib_point
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'gefs_72h_single_run_v1'
    if out.exists():
        raise FileExistsError(out)
    source = base / 'reusable_gefs_snapshots_v1/snapshots.parquet'
    binary = ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    paths = [source, binary, Path(__file__), ROOT / 'src/haze_forecast/app.py']
    before = {str(p): sha(p) for p in paths}
    snapshots = pd.read_parquet(source)
    run = pd.Timestamp(snapshots.run_at.min())
    valid = run + pd.Timedelta(hours=72)
    first_old_known = snapshots.loc[snapshots.run_at.eq(run), 'known_at'].min()
    definitions = {'temperature_c': ('TMP', '2 m above ground'),
                   'relative_humidity_pct': ('RH', '2 m above ground'),
                   'wind_u_ms': ('UGRD', '10 m above ground'),
                   'wind_v_ms': ('VGRD', '10 m above ground')}
    out.mkdir(parents=True)
    products = {}
    for product, directory, suffix in [('geavg', 'pgrb2ap5', 'pgrb2a'), ('gespr', 'pgrb2ap5', 'pgrb2a'), ('gec00', 'pgrb2bp5', 'pgrb2b')]:
        url = f'https://noaa-gefs-pds.s3.amazonaws.com/gefs.{run:%Y%m%d}/{run:%H}/atmos/{directory}/{product}.t{run:%H}z.{suffix}.0p50.f072'
        index = requests.get(url + '.idx', timeout=30)
        index.raise_for_status()
        (out / f'{product}.idx').write_bytes(index.content)
        entries = parse_gfs_index(index.text)
        wanted = definitions if product != 'gec00' else {'boundary_layer_height_m': ('HPBL', 'surface')}
        records = {}
        for field, (variable, level) in wanted.items():
            matches = [e for e in entries if e.variable == variable and e.level == level]
            assert len(matches) == 1
            start, end = range_for_gfs_index_entry(entries, matches[0])
            response = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=45)
            assert response.status_code == 206
            assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
            content = response.content
            assert len(content) == end - start + 1
            assert content[:4] == b'GRIB' and content[7] == 2 and content[-4:] == b'7777'
            assert int.from_bytes(content[8:16], 'big') == len(content)
            path = out / f'{product}_{field}.grb2'
            path.write_bytes(content)
            inventory = subprocess.run([str(binary), str(path), '-s'], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
            assert f'd={run:%Y%m%d%H}' in inventory and ':72 hour fcst:' in inventory
            assert f':{variable}:{level}:' in inventory
            point = query_grib_point(path, 104.0665, 30.5723, binary)
            value = point.value - 273.15 if field == 'temperature_c' and product == 'geavg' else point.value
            assert np.isfinite(value)
            if product == 'gespr' or field == 'boundary_layer_height_m':
                assert value >= 0
            if field == 'relative_humidity_pct' and product == 'geavg':
                assert 0 <= value <= 100
            modified = pd.Timestamp(parsedate_to_datetime(response.headers['Last-Modified']))
            records[field] = {'value': value, 'longitude': point.longitude, 'latitude': point.latitude,
                'byte_range': [start, end], 'sha256': sha(path), 'inventory': inventory,
                'last_modified': modified.isoformat(), 'etag': response.headers.get('ETag'),
                'archive_proxy_available_by_old_known': bool(modified <= first_old_known)}
        assert len({r['etag'] for r in records.values()}) == 1
        products[product] = {'url': url, 'fields': records}
        (out / f'{product}_manifest.json').write_text(json.dumps(products[product], indent=2), encoding='utf-8')
    points = {(r['longitude'], r['latitude']) for p in products.values() for r in p['fields'].values()}
    assert len(points) == 1
    proxy = max(pd.Timestamp(r['last_modified']) for p in products.values() for r in p['fields'].values())
    assert before == {str(p): sha(p) for p in paths}
    report = {'input_sha256': before, 'run_at': run.isoformat(), 'valid_at': valid.isoformat(),
        'old_f024_known_at_reference_only': first_old_known.isoformat(), 'f072_archive_available_at_proxy': proxy.isoformat(),
        'all_products_available_by_old_known_proxy': bool(proxy <= first_old_known), 'products': products,
        'inputs_unchanged': True, 'limitations': ['single run only; not complete historical coverage',
            'Last-Modified is an archive proxy, not historical first-publication proof',
            'no training or deployment; no f024 availability inherited'],
        'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
