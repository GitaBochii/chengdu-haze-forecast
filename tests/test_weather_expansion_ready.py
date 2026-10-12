import hashlib
import json

import pandas as pd
import pytest

from scripts import verify_weather_expansion_ready as gate


def fixture_evidence(tmp_path):
    base = tmp_path / 'artifacts/task01_recovery_head'
    source = base / 'future_weather_expansion_resolved_v1'
    source.mkdir(parents=True)
    row = {'issue_utc': '2021-01-02T03:00:00+00:00',
           'run_utc': '2021-01-01T18:00:00+00:00', 'ahead_hours': 3,
           'url': 'https://example/object'}
    head = {'url': row['url'], 'http_status': 200, 'etag': '"' + 'a' * 32 + '"',
            'last_modified': 'Fri, 01 Jan 2021 22:00:00 GMT', 'content_length': '100'}
    cache = source / 'head.json'
    cache.write_text(json.dumps(head))
    record = {**row, 'state': 'available', 'evidence': {**head,
              'cache_path': str(cache), 'cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest()}}
    evidence = source / 'accepted_evidence.json'
    evidence.write_text(json.dumps([record]))
    return pd.DataFrame([row]), record, evidence, cache


def test_evidence_hash_and_each_relationship(tmp_path):
    plan, record, _evidence, cache = fixture_evidence(tmp_path)
    tracked = {}
    heads = gate.evidence_for_plan(tmp_path, plan, tracked)
    assert heads[record['url']]['etag'] == record['evidence']['etag']
    assert len(tracked) == 2
    cache.write_text('{}')
    with pytest.raises(ValueError, match='cache hash'):
        gate.evidence_for_plan(tmp_path, plan, {})


@pytest.mark.parametrize('mutation', ['duplicate', 'identity', 'late', 'cached_value', 'state'])
def test_reject_bad_relation_evidence(tmp_path, mutation):
    plan, record, evidence, _cache = fixture_evidence(tmp_path)
    rows = [record]
    if mutation == 'duplicate':
        rows.append(record)
    elif mutation == 'identity':
        record['run_utc'] = '2021-01-01T12:00:00+00:00'
    elif mutation == 'late':
        plan.loc[0, 'issue_utc'] = record['issue_utc'] = '2021-01-01T21:00:00+00:00'
    elif mutation == 'cached_value':
        record['evidence']['etag'] = '"' + 'b' * 32 + '"'
    else:
        record['state'] = 'unknown'
    evidence.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        gate.evidence_for_plan(tmp_path, plan, {})


def test_failed_check_removes_old_ready_gate(tmp_path):
    collection = tmp_path / 'artifacts/task01_recovery_head/future_weather_expansion_collection_v1'
    collection.mkdir(parents=True)
    old = collection / 'training_readiness_gate.json'
    old.write_text('{"status":"ready_for_formal_training"}')
    with pytest.raises(FileNotFoundError):
        gate.verify(tmp_path)
    assert not old.exists()
    assert len(list(collection.glob('training_readiness_gate.previous_*.json'))) == 1


def fixture_result(tmp_path):
    plan, original, _, _ = fixture_evidence(tmp_path)
    row = plan.iloc[0].to_dict()
    head = {k: v for k, v in original['evidence'].items()
            if k not in ('cache_path', 'cache_sha256')}
    record = {'probe': {**row, 'etag': head['etag']}, 'head': head,
              'manifest_sha256': gate.MANIFEST_SHA256, 'status': 'pass',
              'fields': {name: {'byte_range': [i * 10, i * 10 + 9]}
                         for i, name in enumerate(gate.FIELDS)}}
    collection = tmp_path / 'collection'
    results = collection / 'results'
    results.mkdir(parents=True)
    path = results / (hashlib.sha256(row['url'].encode()).hexdigest() + '.json')
    path.write_text(json.dumps(record))
    return plan, record, collection, path, {row['url']: head}


def test_complete_results_and_jobs(tmp_path):
    plan, record, collection, path, evidence = fixture_result(tmp_path)
    hashes, records = gate.load_results(collection, plan, evidence)
    assert hashes[path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert gate.object_jobs(plan, records, evidence)[0][0] == record


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'failed', 'manifest', 'filename', 'reuse'])
def test_results_reject_incomplete_or_untrusted(tmp_path, mutation):
    plan, record, collection, path, evidence = fixture_result(tmp_path)
    if mutation == 'failed':
        record['status'] = 'failed'
    elif mutation == 'manifest':
        record['manifest_sha256'] = 'wrong'
    elif mutation == 'reuse':
        record['reused_from'] = 'report.json'
    path.write_text(json.dumps(record))
    if mutation == 'missing':
        path.unlink()
    elif mutation == 'extra':
        (path.parent / 'extra.json').write_text(path.read_text())
    elif mutation == 'filename':
        path.rename(path.parent / 'wrong.json')
    with pytest.raises(ValueError):
        gate.load_results(collection, plan, evidence)


@pytest.mark.parametrize('mutation', ['probe', 'etag', 'size', 'fields', 'range', 'overlap'])
def test_jobs_reject_conflicting_fields(tmp_path, mutation):
    plan, record, _, _, evidence = fixture_result(tmp_path)
    if mutation == 'probe':
        record['probe']['ahead_hours'] = 6
    elif mutation == 'etag':
        record['probe']['etag'] = 'wrong'
    elif mutation == 'size':
        record['head'] = {**record['head'], 'content_length': '99'}
    elif mutation == 'fields':
        record['fields'].pop('wind_u_ms')
    elif mutation == 'range':
        record['fields']['wind_u_ms']['byte_range'] = [90, 100]
    else:
        record['fields']['wind_u_ms']['byte_range'] = [0, 9]
    with pytest.raises(ValueError):
        gate.object_jobs(plan, {record['probe']['url']: record}, evidence)


@pytest.mark.parametrize('mutation', [None, 'result', 'input', 'grib', 'duplicate_path'])
def test_gate_end_to_end_snapshot_integrity(tmp_path, monkeypatch, mutation):
    plan, record, old_collection, result_path, _ = fixture_result(tmp_path)
    base = tmp_path / 'artifacts/task01_recovery_head'
    collection = base / 'future_weather_expansion_collection_v1'
    old_collection.rename(collection)
    result_path = collection / 'results' / result_path.name
    plan['year'] = 2021
    monkeypatch.setattr(gate, 'COUNTS', (1, 1, 1))
    monkeypatch.setattr(gate, 'check_plan', lambda *args: None)
    plan.loc[1] = {**plan.iloc[0].to_dict(), 'year': 2020,
                   'issue_utc': '2020-01-02T03:00:00+00:00'}
    manifest = base / 'future_weather_expansion_resolved_v1/accepted_manifest.csv'
    plan.to_csv(manifest, index=False)
    monkeypatch.setattr(gate, 'MANIFEST_SHA256', hashlib.sha256(manifest.read_bytes()).hexdigest())
    record['probe']['year'] = 2021
    record['manifest_sha256'] = gate.MANIFEST_SHA256
    result_path.write_text(json.dumps(record))
    monkeypatch.setattr(gate, 'evidence_for_plan', lambda *args: {record['probe']['url']: record['head']})
    monkeypatch.setattr(gate, 'object_jobs', lambda *args: [(record, plan.iloc[0].to_dict())])
    inputs = tmp_path / 'data/processed/regional_air_models_full_history'
    inputs.mkdir(parents=True)
    for name in gate.INPUTS:
        (inputs / name).write_bytes(name.encode())
    formal = base / 'future_weather_formal_v1'
    formal.mkdir()
    (formal / 'report.json').write_text(json.dumps({'status': 'completed_train_validation_only',
        'inputs_before': {name: gate.sha(inputs / name) for name in gate.INPUTS}}))
    binary = tmp_path / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b'binary')
    gribs = {}
    for name in gate.FIELDS:
        path = base / (name + '.grib2')
        path.write_bytes(name.encode())
        gribs[name] = (str(path.relative_to(tmp_path)), gate.sha(path), path.stat().st_size)
    def check_field(field, name, *args):
        result = gribs['temperature_c' if mutation == 'duplicate_path' else name]
        if mutation == 'result':
            result_path.write_text('{}')
        elif mutation == 'input':
            (inputs / gate.INPUTS[0]).write_bytes(b'changed')
        elif mutation == 'grib':
            (tmp_path / result[0]).write_bytes(b'changed')
        return result
    monkeypatch.setattr(gate, 'check_field', check_field)
    if mutation:
        with pytest.raises(ValueError):
            gate.verify(tmp_path, workers=1)
        assert not (collection / 'training_readiness_gate.json').exists()
    else:
        output = gate.verify(tmp_path, workers=1)
        assert output['verified_fields'] == 4
        assert output['status'] == 'ready_for_formal_training'
        assert json.loads((collection / 'training_readiness_gate.json').read_text()) == output
