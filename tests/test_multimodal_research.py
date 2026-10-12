import builtins
import importlib.util
import shutil
from pathlib import Path

import pytest

from haze_forecast.multimodal_research import MultimodalInputError, MultimodalResearch

ROOT = Path(__file__).resolve().parents[1]


def test_real_archive_all_combinations_without_torch(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            raise AssertionError('archive imported torch')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    service = MultimodalResearch(ROOT)
    assert service.status()['origins'] == 25
    cases = service.cases()['records']
    assert len(cases) == 25
    for seed in (24, 42, 2026):
        for arm in ('baseline', 'capacity_control', 'spatial_fusion'):
            for case in cases:
                result = service.replay(case['timestamp'], seed, arm)
                assert result['mode'] == 'multimodal_archive'
                assert result['partition'] == case['partition']
                assert len(result['trend']) == 24
                assert result['trend'][-1]['timestamp'] == result['timestamp']
                assert 'actual_hours' not in result
                assert result['capped_tail'] == (result['prediction_hours'] == 68)
    interval = service.replay('2020-02-27T14:00:00+08:00')
    assert (interval['lower_bound_hours'], interval['upper_bound_hours']) == (9, 20)


@pytest.mark.parametrize('kwargs', [{'seed':True}, {'seed':25}, {'seed':'24'},
    {'arm':'../../other'}, {'timestamp':'2026-01-01'}, {'timestamp':None}])
def test_invalid_parameters(kwargs):
    request = dict(timestamp='2020-01-01T14:00:00+08:00')
    request.update(kwargs)
    with pytest.raises(MultimodalInputError):
        MultimodalResearch(ROOT).replay(**request)


@pytest.mark.parametrize('filename', ['audit.json', 'report.json', 'preprocessing.pt', '24_spatial_fusion.pt'])
def test_tampering_is_closed(tmp_path, filename):
    service = MultimodalResearch(ROOT)
    relative = service.run_dir.relative_to(ROOT)
    shutil.copytree(service.run_dir, tmp_path / relative)
    target = tmp_path / relative / filename
    with target.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises((ValueError, RuntimeError)):
        MultimodalResearch(tmp_path).status()


def test_real_reinfer_matches_archive():
    pytest.importorskip('torch')
    service = MultimodalResearch(ROOT)
    timestamp = '2020-01-01T14:00:00+08:00'
    result = service.reinfer(timestamp)
    archive = service.replay(timestamp)
    assert result['mode'] == 'multimodal_reinference'
    assert {k:v for k,v in result.items() if k != 'mode'} == {
        k:v for k,v in archive.items() if k != 'mode'}


def test_no_torch_status_and_tail_semantics(monkeypatch):
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda name: None if name == 'torch' else original(name))
    service = MultimodalResearch(ROOT)
    assert service.status()['reinfer_available'] is False
    report, rows = service._verified()
    row = next(r for r in rows if r['lower_bound_hours'] == 108)
    result = service._response(report,row,24,'spatial_fusion',68,'multimodal_archive')
    assert result['capped_tail'] is True
    assert result['lower_bound_hours'] == result['upper_bound_hours'] == 108
    assert result['error_lower_hours'] == result['error_upper_hours'] == 0
    assert 'actual_hours' not in result
