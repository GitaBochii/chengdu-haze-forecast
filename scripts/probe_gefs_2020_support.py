"""One post-upgrade nine-field probe, plus bounded old-layout listings; no training."""

import hashlib
import json
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from scripts.collect_gefs_2021_f024 import extract, verify_complete

ROOT = Path(__file__).resolve().parents[1]


def main():
    b = ROOT / 'artifacts/task01_recovery_head'
    out = b / 'gefs_2020_support_probe_20260918'
    if out.exists():
        raise FileExistsError(out)
    source = b / 'gefs_2021_extension_v1/f024_training_rows.parquet'
    binary = ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    paths = [source, binary, Path(__file__).resolve(), ROOT / 'scripts/collect_gefs_2021_f024.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    data = pd.read_parquet(source)
    year = data.loc[data.timestamp.dt.year.eq(2020)]
    recent = year.loc[year.timestamp.ge(pd.Timestamp('2020-09-23', tz='Asia/Shanghai'))]
    first = recent.timestamp.min().tz_convert('UTC')
    run = (first - pd.Timedelta('4h')).floor('6h')
    out.mkdir()
    protocol = {'input_sha256': hashes, 'year_rows': len(year), 'post_upgrade_rows': len(recent),
                'post_upgrade_blocks': int(recent.continuity_block.nunique()),
                'first_origin': first.isoformat(), 'candidate_run': run.isoformat(),
                'budget': 'Three old-layout listing requests and one nine-field post-upgrade batch; no retries or training.'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    listings = []
    for product, folder in [('geavg', 'pgrb2a'), ('gespr', 'pgrb2a'), ('gec00', 'pgrb2b')]:
        prefix = f'gefs.20191231/12/{folder}/{product}'
        response = requests.get('https://noaa-gefs-pds.s3.amazonaws.com/',
                                params={'list-type': '2', 'prefix': prefix, 'max-keys': 3}, timeout=25)
        response.raise_for_status()
        (out / f'old_{product}.xml').write_bytes(response.content)
        root = ET.fromstring(response.content)
        ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        listings.append({'prefix': prefix, 'keys': [n.text for n in root.findall('s:Contents/s:Key', ns)],
                         'truncated': root.findtext('s:IsTruncated', namespaces=ns)})
    folder = out / run.strftime('%Y%m%d%H')
    extract(run, folder, binary)
    record = verify_complete(folder, run)
    decoded = []
    for path in sorted(folder.glob('*.grb2')):
        text = subprocess.check_output([str(binary), str(path), '-s', '-vt', '-lon', '104.0665', '30.5723'],
                                       text=True, timeout=30).strip()
        assert f'd={run:%Y%m%d%H}:' in text and ':24 hour fcst:' in text
        assert f'vt={run + pd.Timedelta("24h"):%Y%m%d%H}' in text
        decoded.append({'file': path.name, 'inventory': text})
    assert len(decoded) == 9
    for p,h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    report = {'old_layout_listings': listings, 'verified_messages': decoded,
              'archive_available_at_proxy': record['archive_available_at_proxy'],
              'proxy_before_first_origin': pd.Timestamp(record['archive_available_at_proxy']) <= first,
              'scope': 'One run only. Last-Modified is not first-publication certification. No annual coverage or accuracy claim.',
              'output_sha256': {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in out.rglob('*') if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'protocol': protocol, 'proxy_before_first_origin': report['proxy_before_first_origin'],
                      'proxy': record['archive_available_at_proxy'], 'verified_messages': len(decoded)}, indent=2))


if __name__ == '__main__':
    main()
