import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/build_runtime_package.py'
spec = importlib.util.spec_from_file_location('runtime_builder', SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    files = list(builder.REQUIRED_FILES) + ['src/haze_forecast/__init__.py']
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    (root / 'src/haze_forecast/showcase_server.py').write_text(
        "STATIC_FILES = frozenset({'index.html', 'rebound-cases.json'})", encoding='utf-8')
    for name in ('index.html', 'rebound-cases.json'):
        (root / 'project-showcase').mkdir(exist_ok=True)
        (root / 'project-showcase' / name).write_text('fixture')
    for relative in ('data/raw/secret.csv', '.env', 'experiments/model.pt',
                     'src/haze_forecast/__pycache__/module.pyc'):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'not allowed')
    return root


def test_allowlist_hash_crc_and_source_preserved(source, tmp_path):
    before = {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result = builder.build_runtime_package(source, tmp_path / 'output')
    manifest = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
    assert hashlib.sha256(Path(result['archive']).read_bytes()).hexdigest() == manifest['archive_sha256']
    with zipfile.ZipFile(result['archive']) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {f['path'] for f in manifest['files']}
        for entry in manifest['files']:
            assert hashlib.sha256(archive.read(entry['path'])).hexdigest() == entry['sha256']
        assert not any('secret' in n or '.env' in n or 'experiments/' in n
                       or '__pycache__' in n for n in archive.namelist())
        assert 'project-showcase/rebound-cases.json' in archive.namelist()
        assert b'pyarrow' in archive.read('requirements-runtime.txt')
    assert before == {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}


def test_missing_input_rejected(source, tmp_path):
    (source / builder.REQUIRED_FILES[0]).unlink()
    with pytest.raises(FileNotFoundError):
        builder.build_runtime_package(source, tmp_path / 'output')


def test_output_never_overwritten(source, tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    with pytest.raises(FileExistsError):
        builder.build_runtime_package(source, output)


@pytest.mark.parametrize('unsafe', ['../secret', '/secret', 'C:/secret', '.env', 'assets/../secret'])
def test_unsafe_static_names_rejected(source, tmp_path, unsafe):
    (source / 'src/haze_forecast/showcase_server.py').write_text(
        f'STATIC_FILES = frozenset({{ {unsafe!r} }})')
    with pytest.raises(ValueError):
        builder.build_runtime_package(source, tmp_path / 'output')


def test_symlink_rejected(source, tmp_path):
    link = source / 'src/haze_forecast/linked.py'
    try:
        link.symlink_to(source / 'pyproject.toml')
    except OSError:
        pytest.skip('Windows symlink privilege unavailable')
    with pytest.raises(ValueError):
        builder.build_runtime_package(source, tmp_path / 'output')


def test_source_change_rejected(source, tmp_path, monkeypatch):
    original = builder.digest_file
    changed = False
    def mutate(path):
        nonlocal changed
        result = original(path)
        if not changed:
            path.write_bytes(path.read_bytes() + b'changed')
            changed = True
        return result
    monkeypatch.setattr(builder, 'digest_file', mutate)
    with pytest.raises(RuntimeError, match='Source changed'):
        builder.build_runtime_package(source, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_crc_failure_rejected(source, tmp_path, monkeypatch):
    monkeypatch.setattr(zipfile.ZipFile, 'testzip', lambda self: 'corrupt.txt')
    with pytest.raises(RuntimeError, match='CRC'):
        builder.build_runtime_package(source, tmp_path / 'output')
    assert not list((tmp_path / 'output').glob('*.manifest.json'))


def test_archive_hash_failure_rejected(source, tmp_path, monkeypatch):
    original = zipfile.ZipFile.open
    def corrupted_read(self, name, mode='r', *args, **kwargs):
        if mode == 'r':
            return io.BytesIO(b'corrupted bytes')
        return original(self, name, mode, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, 'testzip', lambda self: None)
    monkeypatch.setattr(zipfile.ZipFile, 'open', corrupted_read)
    with pytest.raises(RuntimeError, match='SHA256'):
        builder.build_runtime_package(source, tmp_path / 'output')
    assert not list((tmp_path / 'output').glob('*.manifest.json'))


def test_size_budget(source, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, 'MAX_BYTES', 1)
    with pytest.raises(ValueError, match='budget'):
        builder.build_runtime_package(source, tmp_path / 'output')


def test_only_workbench_tab_is_adapted(source, tmp_path):
    path = source / 'project-showcase/workbench.html'
    original = b'<button id="tab-multimodal" data-mode="multimodal">Research</button>'
    path.write_bytes(original)
    server = source / 'src/haze_forecast/showcase_server.py'
    server.write_text("STATIC_FILES = frozenset({'index.html', 'rebound-cases.json', 'workbench.html'})")
    result = builder.build_runtime_package(source, tmp_path / 'output')
    manifest = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
    adapted = [f for f in manifest['files'] if f.get('adaptation')]
    assert len(adapted) == 1
    assert adapted[0]['source_sha256'] == hashlib.sha256(original).hexdigest()
    assert adapted[0]['source_sha256'] != adapted[0]['staged_sha256']
    with zipfile.ZipFile(result['archive']) as archive:
        assert archive.read('project-showcase/workbench.html') == original.replace(
            b'id="tab-multimodal"', b'id="tab-multimodal" hidden')
        assert b'.venv_runtime' in archive.read('scripts/install_runtime.ps1')
        assert b'.venv_runtime' in archive.read('scripts/start_runtime.ps1')
        assert b'$env:PYTHONPATH' in archive.read('scripts/start_runtime.ps1')
    assert path.read_bytes() == original


def test_keyboard_tabs_exclude_hidden_research(source, tmp_path):
    original = b"const tabs = [...document.querySelectorAll('[role=tab]')];"
    path = source / 'project-showcase/workbench.js'
    path.write_bytes(original)
    server = source / 'src/haze_forecast/showcase_server.py'
    server.write_text("STATIC_FILES = frozenset({'index.html', 'rebound-cases.json', 'workbench.js'})")
    result = builder.build_runtime_package(source, tmp_path / 'output')
    expected = original.replace(b"document.querySelectorAll('[role=tab]')",
                                b"document.querySelectorAll('[role=tab]:not([hidden])')")
    with zipfile.ZipFile(result['archive']) as archive:
        assert archive.read('project-showcase/workbench.js') == expected
    manifest = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
    entry = next(f for f in manifest['files'] if f['path'] == 'project-showcase/workbench.js')
    assert entry['adaptation'] == 'exclude_hidden_tabs_from_keyboard_navigation'
    assert entry['source_sha256'] == hashlib.sha256(original).hexdigest()
    assert entry['staged_sha256'] == hashlib.sha256(expected).hexdigest()
    assert path.read_bytes() == original
