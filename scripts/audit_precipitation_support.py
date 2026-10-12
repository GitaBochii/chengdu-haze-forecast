"""Inventory raw precipitation periods and conservative quality coverage."""

import json
from pathlib import Path

import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    out = ROOT / 'artifacts/task01_recovery_head/precipitation_support_v1'
    if out.exists():
        raise FileExistsError(out)
    records, hashes = [], {}
    for path in sorted((ROOT / 'data/raw/noaa_isd').glob('*/*.csv')):
        if not path.parent.name.isdigit():
            continue
        hashes[str(path)] = sha(path)
        raw = pd.read_csv(path, dtype=str, usecols=lambda c: c in ['DATE', 'AA1', 'AA2', 'AA3', 'AA4'])
        fields = [c for c in raw if c.startswith('AA')]
        if not fields:
            records.append({'year': int(path.parent.name), 'station': path.stem,
                            'field': None, 'rows': len(raw), 'present': 0,
                            'conservative_usable': 0, 'periods': {}, 'conditions': {}, 'quality': {}})
        for field in fields:
            values = raw[field].dropna().str.split(',', expand=True).reindex(columns=range(4))
            hours = pd.to_numeric(values[0], errors='coerce')
            depth = pd.to_numeric(values[1], errors='coerce')
            # Unknown/special accumulation conditions are excluded, not silently normalized.
            accepted = (hours.between(1, 98) & depth.between(0, 9998)
                        & values[2].eq('9') & values[3].isin(['0', '1', '4', '5']))
            records.append({'year': int(path.parent.name), 'station': path.stem,
                            'field': field, 'rows': len(raw), 'present': len(values),
                            'conservative_usable': int(accepted.sum()),
                            'periods': values[0].value_counts().to_dict(),
                            'conditions': values[2].value_counts().to_dict(),
                            'quality': values[3].value_counts().to_dict()})
    assert hashes == {path: sha(Path(path)) for path in hashes}
    out.mkdir()
    report = {'input_sha256': hashes, 'records': records,
              'policy': 'Count only known-duration/depth, condition 9, quality 0/1/4/5; no fill',
              'limits': ['Conservative support inventory, not a validated full precipitation parser',
                         'Counts across AA fields can overlap; never sum them as independent observations',
                         'Archive observation time does not prove first publication time',
                         'No model training or accuracy claim'],
              'standard': 'https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf'}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(pd.DataFrame(records)[['year', 'station', 'field', 'rows', 'present', 'conservative_usable']].to_string(index=False))


if __name__ == '__main__':
    main()
