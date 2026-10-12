import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from scripts import audit_gefs_2021_f024 as audit


def fixture_run(tmp_path, monkeypatch):
    folder = tmp_path / '2021010200'
    folder.mkdir()
    products = {}
    content = b'GRIB' + b'\0' * 3 + b'\x02' + (20).to_bytes(8, 'big') + b'7777'
    for product, fields in audit.FIELDS.items():
        records = {}
        lines = []
        for index, (field, (variable, level)) in enumerate(fields.items()):
            name = f'{product}_{field}.grb2'
            (folder / name).write_bytes(content)
            lines.append(f'{index + 1}:{index * 20}:d=2021010200:{variable}:{level}:24 hour fcst:')
            records[field] = {'value': 6.85 if product == 'geavg' and field == 'temperature_c' else 10,
                              'longitude': 104.0, 'latitude': 30.5,
                              'last_modified': '2021-01-02T04:10:00+00:00', 'etag': 'same',
                              'sha256': audit.digest(folder / name),
                              'byte_range': [index * 20, index * 20 + 19]}
        lines.append(f'{len(fields) + 1}:{len(fields) * 20}:d=2021010200:OTHER:surface:24 hour fcst:')
        (folder / f'{product}.idx').write_text('\n'.join(lines))
        products[product] = {'url': audit.product_url(datetime(2021, 1, 2, tzinfo=UTC), product),
                             'fields': records}
    record = {'run_at': '2021-01-02T00:00:00+00:00', 'valid_at': '2021-01-03T00:00:00+00:00',
              'archive_available_at_proxy': '2021-01-02T04:10:00+00:00', 'products': products,
              'output_sha256': {p.name: audit.digest(p) for p in folder.iterdir()}}
    marker = folder / 'complete.json'
    marker.write_text(json.dumps(record))

    def decode(command, **_kwargs):
        product, field = Path(command[1]).stem.split('_', 1)
        variable, level = audit.FIELDS[product][field]
        value = 280 if product == 'geavg' and field == 'temperature_c' else 10
        return (f'1:0:d=2021010200:{variable}:{level}:24 hour fcst:{audit.IDENTITIES[product]}:'
                f'vt=2021010300:lon=104,lat=30.5,val={value}\n')

    monkeypatch.setattr(audit.subprocess, 'check_output', decode)
    return marker, record


def test_independent_decode_accepts_nine_fields(tmp_path, monkeypatch):
    marker, record = fixture_run(tmp_path, monkeypatch)
    result = audit.audit_run(marker, tmp_path / 'decoder', [record['run_at']])
    assert result == {'run_at': record['run_at'], 'complete_sha256': audit.digest(marker)}


@pytest.mark.parametrize('defect', ['value', 'proxy', 'etag', 'fields', 'hash', 'extra_file', 'range'])
def test_independent_audit_rejects_tampering(tmp_path, monkeypatch, defect):
    marker, record = fixture_run(tmp_path, monkeypatch)
    item = record['products']['geavg']['fields']['temperature_c']
    if defect == 'value':
        item['value'] = 50
    elif defect == 'proxy':
        record['archive_available_at_proxy'] = '2021-01-02T04:00:00+00:00'
    elif defect == 'etag':
        item['etag'] = 'changed'
    elif defect == 'fields':
        record['products']['geavg']['fields'].pop('wind_u_ms')
    elif defect == 'hash':
        record['output_sha256']['geavg.idx'] = 'bad'
    elif defect == 'range':
        item['byte_range'] = [1, 20]
    else:
        (marker.parent / 'unexpected').write_text('extra')
    marker.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        audit.audit_run(marker, tmp_path / 'decoder', [record['run_at']])


def test_plan_reconstructed_without_collector():
    frame = pd.DataFrame({'timestamp': pd.to_datetime([
        '2021-01-01 00:00+08:00', '2021-01-02 08:00+08:00', '2022-01-01 00:00+08:00'])})
    assert audit.reconstruct_runs(frame) == ['2020-12-31T12:00:00+00:00', '2021-01-01T18:00:00+00:00']


def test_partial_report_never_authorizes_model(tmp_path, monkeypatch):
    marker, record = fixture_run(tmp_path, monkeypatch)
    runs = [record['run_at'], '2021-01-03T00:00:00+00:00']
    (tmp_path / 'protocol.json').write_text('{}')
    report = audit.audit_completed(tmp_path, tmp_path / 'decoder', runs)
    assert report['verified_runs'] == 1
    assert report['planned_runs'] == 2
    assert report['verified_grib_messages'] == 9
    assert report['all_planned_runs_verified'] is False
    assert report['formal_model_evaluation_allowed'] is False
    assert report['verified'][0]['complete_sha256'] == audit.digest(marker)


@pytest.mark.parametrize('defect', [None, 'source', 'collector', 'decoder', 'helper', 'runs'])
def test_protocol_checks_frozen_inputs_and_plan(tmp_path, defect):
    source = tmp_path / 'origins.parquet'
    frame = pd.DataFrame({'timestamp': pd.date_range('2021-01-02', periods=101, freq='D',
                                                    tz='Asia/Shanghai')})
    frame.to_parquet(source)
    binary = tmp_path / 'decoder'
    binary.write_bytes(b'decoder')
    collector = tmp_path / 'scripts/collect_gefs_2021_f024.py'
    collector.parent.mkdir()
    collector.write_bytes(b'collector')
    for name in audit.HELPERS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'helper')
    runs = audit.reconstruct_runs(frame)
    protocol = {'source_sha256': audit.digest(source), 'decoder_sha256': audit.digest(binary),
                'collector_sha256': audit.digest(collector), 'runs': runs, 'lead_hours': 24,
                'helper_sha256': {name: audit.digest(tmp_path / name) for name in audit.HELPERS}}
    if defect in {'source', 'decoder', 'collector'}:
        protocol[f'{defect}_sha256'] = 'changed'
    elif defect == 'helper':
        protocol['helper_sha256']['src/haze_forecast/grib.py'] = 'changed'
    elif defect == 'runs':
        protocol['runs'] = runs[:-1]
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    if defect is None:
        assert audit.verify_protocol(tmp_path, tmp_path, binary, source) == runs
    else:
        with pytest.raises(ValueError):
            audit.verify_protocol(tmp_path, tmp_path, binary, source)
