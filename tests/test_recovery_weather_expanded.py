import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import train_recovery_weather_expanded as expanded


def test_weather_training_selector_excludes_validation_and_other_years():
    origins = pd.DatetimeIndex(['2019-12-31T16:00Z', '2021-01-01T00:00Z',
                               '2022-01-01T00:00Z', '2023-01-01T00:00Z',
                               '2019-01-01T00:00Z'])
    np.testing.assert_array_equal(expanded.weather_training_rows(origins), [True, True, True, False, False])


def test_scaler_uses_training_years_only(monkeypatch):
    origins = pd.DatetimeIndex(['2020-01-01T00:00Z', '2021-01-01T00:00Z',
                               '2022-01-01T00:00Z', '2023-01-01T00:00Z'])
    raw = np.broadcast_to(np.array([1., 3., 5., 999.])[:, None, None], (4, 24, 4)).copy()
    mask = np.ones_like(raw, dtype=bool)
    monkeypatch.setattr(expanded, 'build_weather_inputs', lambda *args: (raw, mask))
    values, observed, scaler = expanded.fit_expanded_weather(None, origins)
    np.testing.assert_array_equal(scaler.mean, np.full(4, 3.))
    np.testing.assert_allclose(scaler.scale, np.full(4, np.std([1., 3., 5.])))
    assert values.dtype == np.float32 and observed.all()
    raw[-1] = -999.
    np.testing.assert_array_equal(expanded.fit_expanded_weather(None, origins)[2].mean, scaler.mean)


def test_expansion_loader_rejects_not_ready(tmp_path):
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps({'status': 'not_ready'}))
    with pytest.raises(ValueError, match='ready'):
        expanded.load_expansion_manifest(gate, tmp_path / 'manifest.csv', tmp_path / 'results', {})


def test_snapshot_detects_mutated_source(tmp_path):
    source = tmp_path / 'source.py'
    source.write_bytes(b'original')
    expected = {str(source): expanded.sha(source)}
    assert expanded.hashes_unchanged(expected)
    source.write_bytes(b'changed')
    assert not expanded.hashes_unchanged(expected)
    source.unlink()
    assert not expanded.hashes_unchanged(expected)


def expansion_fixture(tmp_path, monkeypatch):
    rows = []
    for year in (2020, 2021):
        issue = pd.Timestamp(f'{year}-01-02T03:00Z')
        run = issue.floor('6h') - pd.Timedelta(hours=6)
        for ahead in range(3, 73, 3):
            lead = ahead + 9
            url = (f'https://noaa-gefs-pds.s3.amazonaws.com/gefs.{run:%Y%m%d}/{run:%H}/atmos/'
                   f'pgrb2ap5/geavg.t{run:%H}z.pgrb2a.0p50.f{lead:03d}')
            rows.append({'issue_utc': issue.isoformat(), 'run_utc': run.isoformat(),
                         'valid_utc': (issue + pd.Timedelta(hours=ahead)).isoformat(),
                         'ahead_hours': ahead, 'lead_hours': lead, 'cycles_back': 1, 'url': url, 'year': year})
    plan = pd.DataFrame(rows)
    manifest = tmp_path / 'manifest.csv'
    plan.to_csv(manifest, index=False)
    digest = expanded.sha(manifest)
    monkeypatch.setattr(expanded, 'EXPANSION_COUNTS', (48, 2, 48))
    monkeypatch.setattr(expanded, 'EXPANSION_MANIFEST_SHA256', digest)
    monkeypatch.setattr(expanded, 'ORIGIN_COUNTS', {2020: 1, 2021: 1})
    results = tmp_path / 'results'
    results.mkdir()
    paths = []
    for row in rows:
        path = results / (hashlib.sha256(row['url'].encode()).hexdigest() + '.json')
        path.write_text(json.dumps({'probe': row, 'status': 'pass', 'manifest_sha256': digest}))
        paths.append(path)
    payload = {'status': 'ready_for_formal_training', 'passed_objects': 48,
               'failed_objects': 0, 'pending_objects': 0, 'complete_origins': 2,
               'manifest_relationships': 48, 'verified_fields': 192,
               'manifest_sha256': digest, 'input_data_sha256': {'input': 'fixed'},
               'results_sha256': {path.name: expanded.sha(path) for path in paths}}
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps(payload))
    return gate, manifest, results, payload, paths


def test_expansion_loader_reads_exact_frozen_relationships(tmp_path, monkeypatch):
    gate, manifest, results, payload, _ = expansion_fixture(tmp_path, monkeypatch)
    actual_gate, plan, records = expanded.load_expansion_manifest(gate, manifest, results, {'input': 'fixed'})
    assert actual_gate == payload and len(plan) == len(records) == 48


@pytest.mark.parametrize('mutation', ['count', 'missing', 'extra', 'hash', 'input', 'failed', 'stale', 'url', 'filename'])
def test_expansion_loader_rejects_bad_provenance(tmp_path, monkeypatch, mutation):
    gate, manifest, results, payload, paths = expansion_fixture(tmp_path, monkeypatch)
    if mutation == 'count':
        payload['complete_origins'] = 1
    elif mutation == 'missing':
        paths[0].unlink()
    elif mutation == 'extra':
        (results / 'extra.json').write_text('{}')
    elif mutation == 'hash':
        paths[0].write_text('{}')
    elif mutation == 'input':
        payload['input_data_sha256'] = {'input': 'other'}
    elif mutation == 'filename':
        old = paths[0]
        target = results / 'wrong.json'
        old.rename(target)
        payload['results_sha256'][target.name] = payload['results_sha256'].pop(old.name)
    else:
        record = json.loads(paths[0].read_text())
        if mutation == 'failed':
            record['status'] = 'failed'
        elif mutation == 'stale':
            record['manifest_sha256'] = 'stale'
        else:
            record['probe']['url'] = 'unexpected'
        paths[0].write_text(json.dumps(record))
        payload['results_sha256'][paths[0].name] = expanded.sha(paths[0])
    gate.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        expanded.load_expansion_manifest(gate, manifest, results, {'input': 'fixed'})


@pytest.mark.parametrize('mutation', [None, 'extra', 'missing', 'content', 'gate'])
def test_sources_unchanged_requires_exact_result_set(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(expanded, 'ROOT', tmp_path)
    results = tmp_path / 'collection/results'
    results.mkdir(parents=True)
    for path in (tmp_path / 'gate.json', tmp_path / 'manifest.csv', results / 'one.json'):
        path.write_bytes(b'original')
    source = {'gate_path': 'gate.json', 'gate_sha256': expanded.sha(tmp_path / 'gate.json'),
              'manifest_path': 'manifest.csv', 'manifest_sha256': expanded.sha(tmp_path / 'manifest.csv'),
              'collection': 'collection', 'results_sha256': {'one.json': expanded.sha(results / 'one.json')},
              'evidence_sha256': {}, 'grib_sha256': {}}
    if mutation == 'extra':
        (results / 'two.json').write_bytes(b'extra')
    elif mutation == 'missing':
        (results / 'one.json').unlink()
    elif mutation == 'content':
        (results / 'one.json').write_bytes(b'changed')
    elif mutation == 'gate':
        (tmp_path / 'gate.json').write_bytes(b'changed')
    assert expanded.sources_unchanged({'original': source}) is (mutation is None)


def test_weather_supervision_audit_distinguishes_missing_from_censored():
    dataset = SimpleNamespace(origins=np.arange(5), weather_by_origin={0: (), 1: (), 2: (), 3: ()},
                              eligible=np.ones(5, dtype=bool),
                              risk_mask=np.array([[1, 1], [1, 1], [1, 0], [0, 0], [1, 1]], dtype=bool),
                              hazard=np.array([[1, 0], [0, 0], [0, 0], [0, 0], [1, 0]], dtype=bool))
    assert expanded.weather_label_audit(dataset) == {
        'covered_origins': 4, 'informative': 3, 'exact': 1, 'censored': 2,
        'full_horizon_no_recovery': 1, 'partial_prefix_censored': 1, 'risk_bins': 5}


@pytest.mark.parametrize('tamper', [False, True])
def test_models_start_from_exact_reference_initialization(tmp_path, monkeypatch, tamper):
    class Tiny(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)
    arms = dict.fromkeys(['base', 'masked_values', 'weather'], Tiny)
    monkeypatch.setattr(expanded, 'ARMS', arms)
    monkeypatch.setattr(expanded, 'REFERENCE', tmp_path)
    for arm in arms:
        torch.manual_seed(42)
        model = Tiny()
        if tamper and arm == 'weather':
            with torch.no_grad():
                model.layer.weight.add_(1)
        (tmp_path / arm).mkdir()
        torch.save(model.state_dict(), tmp_path / arm / 'initial_state.pt')
    panel = {'feature_names': ['x'], 'coords': [],
             'scaler': SimpleNamespace(mean=np.ones(2), scale=np.ones(2))}
    if tamper:
        with pytest.raises(ValueError, match='reference initial weight'):
            expanded.initialize_models(panel, torch.device('cpu'))
    else:
        models, full, shared = expanded.initialize_models(panel, torch.device('cpu'))
        assert set(models) == set(arms)
        assert len(set(full.values())) == len(set(shared.values())) == 1
