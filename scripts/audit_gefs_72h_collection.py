"""Independently decode completed f072 batches; never authorize model training."""

import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / 'artifacts/task01_recovery_head/gefs_72h_collection_v1'
    binary = root / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    protocol = json.loads((base / 'protocol.json').read_bytes())
    assert digest(binary) == protocol['decoder_sha256']
    definitions = {
        'temperature_c': ('TMP', '2 m above ground'),
        'relative_humidity_pct': ('RH', '2 m above ground'),
        'wind_u_ms': ('UGRD', '10 m above ground'),
        'wind_v_ms': ('VGRD', '10 m above ground'),
    }
    verified = []
    for marker in sorted(base.glob('*/complete.json')):
        record = json.loads(marker.read_bytes())
        run = datetime.fromisoformat(record['run_at'])
        valid = run + timedelta(hours=72)
        assert record['run_at'] in protocol['runs']
        assert marker.parent.name == run.strftime('%Y%m%d%H')
        assert datetime.fromisoformat(record['valid_at']) == valid
        assert set(record['products']) == {'geavg', 'gespr', 'gec00'}
        expected = {f'{p}.idx' for p in record['products']}
        dates = []
        for product, data in record['products'].items():
            fields = definitions if product != 'gec00' else {'boundary_layer_height_m': ('HPBL', 'surface')}
            assert set(data['fields']) == set(fields)
            for field, (variable, level) in fields.items():
                item = data['fields'][field]
                name = f'{product}_{field}.grb2'
                expected.add(name)
                path = marker.parent / name
                assert digest(path) == item['sha256']
                output = subprocess.check_output(
                    [str(binary), str(path), '-s', '-vt', '-lon', '104.0665', '30.5723'],
                    text=True, timeout=30,
                )
                assert f'd={run:%Y%m%d%H}:' in output
                assert f'vt={valid:%Y%m%d%H}:' in output
                assert f':{variable}:{level}:72 hour fcst:' in output
                identity = {'geavg': 'ens mean', 'gespr': 'ens std dev', 'gec00': 'ENS=low-res ctl'}[product]
                assert identity in output
                match = re.search(r'lon=([^,]+),lat=([^,]+),val=([^\s]+)', output)
                assert match is not None
                longitude, latitude, value = map(float, match.groups())
                if product == 'geavg' and field == 'temperature_c':
                    value -= 273.15
                assert math.isfinite(value) and math.isclose(value, item['value'], abs_tol=1e-8)
                assert (longitude, latitude) == (item['longitude'], item['latitude']) == (104.0, 30.5)
                dates.append(datetime.fromisoformat(item['last_modified']))
        assert set(record['output_sha256']) == expected
        for name in expected:
            assert digest(marker.parent / name) == record['output_sha256'][name]
        assert max(dates) == datetime.fromisoformat(record['archive_available_at_proxy'])
        verified.append({'run_at': record['run_at'], 'complete_sha256': digest(marker)})
    assert len({r['run_at'] for r in verified}) == len(verified)
    report = {
        'verified_runs': len(verified), 'verified_grib_messages': 9 * len(verified),
        'planned_runs': len(protocol['runs']), 'formal_model_evaluation_allowed': False,
        'limitations': ['Local metadata consistency only; HTTP headers not independently re-fetched.',
                        'Origin coverage and causal matching require a separate audit.'],
        'audit_script_sha256': digest(Path(__file__)),
        'protocol_sha256': digest(base / 'protocol.json'),
        'helper_sha256': {name: digest(root / 'src/haze_forecast' / name)
                          for name in ['grib.py', 'forecast_sources.py']},
        'verified': verified,
    }
    (base / 'independent_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in {'verified', 'helper_sha256'}}, indent=2))


if __name__ == '__main__':
    main()
