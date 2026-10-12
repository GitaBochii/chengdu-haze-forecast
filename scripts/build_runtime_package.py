"""Build a verified, explicit-allowlist runtime without modifying research inputs."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import uuid
import zipfile


REQUIRED_FILES = (
    'artifacts/cnemc_2025_deployment_candidate/aft_model.joblib',
    'artifacts/cnemc_2025_deployment_candidate/static_aft_model.joblib',
    'artifacts/cnemc_2025_deployment_candidate/model_metadata.json',
    'artifacts/strict_history_backtest/predictions.parquet',
    'data/processed/regional_air_models_full_history/city_hourly.parquet',
    'pyproject.toml',
    'src/haze_forecast/showcase_server.py',
)
MAX_BYTES = 100 * 1024 * 1024
SOURCE_SUFFIXES = {'.py', '.css'}
STATIC_SUFFIXES = {'.html', '.css', '.js', '.json', '.png', '.md'}


def safe_relative(name):
    if not isinstance(name, str) or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe relative path: {name!r}')
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in ('.', '..') for p in name.split('/')):
        raise ValueError(f'Unsafe relative path: {name!r}')
    for part in path.parts:
        lower = part.lower()
        if (lower.startswith('.') or lower in {'credentials', 'secrets', 'token', 'tokens'}
                or re.search(r'(^|[_.-])(secret|credential|password|private[_-]?key)([_.-]|$)', lower)):
            raise ValueError(f'Sensitive path rejected: {name}')
    return path


def checked_path(root, name):
    relative = safe_relative(name)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError(f'Link or junction rejected: {current}')
    if not current.resolve().is_relative_to(root):
        raise ValueError(f'Path escaped source root: {current}')
    if not current.is_file():
        raise FileNotFoundError(current)
    return current


def source_files(root):
    selected = {name: checked_path(root, name) for name in REQUIRED_FILES}
    package = root / 'src/haze_forecast'
    checked_path(root, 'src/haze_forecast/showcase_server.py')
    # Only this bounded source package is traversed, never the research workspace.
    for directory, folders, files in os.walk(package, followlinks=False):
        parent = Path(directory)
        for folder in list(folders):
            child = parent / folder
            if child.is_symlink() or (hasattr(child, 'is_junction') and child.is_junction()):
                raise ValueError(f'Link or junction rejected: {child}')
            if folder == '__pycache__':
                folders.remove(folder)
            else:
                safe_relative(child.relative_to(root).as_posix())
        for filename in files:
            path = parent / filename
            if path.is_symlink():
                raise ValueError(f'Link rejected: {path}')
            if path.suffix in {'.pyc', '.pyo'}:
                continue
            name = path.relative_to(root).as_posix()
            if path.suffix not in SOURCE_SUFFIXES:
                raise ValueError(f'Unexpected source-package asset: {name}')
            selected[name] = checked_path(root, name)
    tree = ast.parse(selected['src/haze_forecast/showcase_server.py'].read_text(encoding='utf-8'))
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'STATIC_FILES' for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('Expected exactly one static asset allowlist')
    value = assignments[0].value
    if (not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name)
            or value.func.id != 'frozenset' or len(value.args) != 1 or value.keywords):
        raise ValueError('Unexpected static asset allowlist expression')
    names = ast.literal_eval(value.args[0])
    if not isinstance(names, (set, list, tuple)) or not names:
        raise ValueError('Empty or invalid static asset allowlist')
    for name in names:
        relative = safe_relative(name)
        if relative.suffix not in STATIC_SUFFIXES:
            raise ValueError(f'Unexpected static asset: {name}')
        full_name = 'project-showcase/' + name
        selected[full_name] = checked_path(root, full_name)
    return dict(sorted(selected.items()))


README = '''# 成都雾霾消散时间预测运行版

## 安装与启动

解压到独立目录。需要 Windows、Python 3.11 至 3.13 和可联网的依赖安装环境。
首次运行“安装运行环境.cmd”，完成后运行“启动运行版.cmd”。安装与启动统一使用本目录的
`.venv_runtime`。环境不随包分发；离线时不会自动安装依赖。启动后打开本机网页，
仅监听 127.0.0.1，自动选择 8510 至 8530 的空闲端口。启动失败会显示错误和日志位置。
关闭网页不会停止服务；启动窗口给出进程 ID，需要停止时可在任务管理器结束对应 Python 进程。

## 可用内容与边界

保留普通 HTML 展示、严格历史回测、手动输入、连续序列预测、在线估算和反弹案例。
在线估算依赖外部服务，不是监测站实测，外部服务可用性与独立预测精度不作保证。
多模态研究数据和大模型不包含在本包内，研究标签页已隐藏。历史样本不是全新盲测，
本包没有调整模型、特征、标签或预测结果，不用于公共预警。

模型来自 `artifacts/cnemc_2025_deployment_candidate/` 的两份 joblib 与模型元数据；
历史预测来自 `artifacts/strict_history_backtest/predictions.parquet`；
连续观测来自 `data/processed/regional_air_models_full_history/city_hourly.parquet`。
仅加载可信来源的 joblib 文件，不要替换成不明来源文件。

## 完整性

ZIP 旁的 manifest JSON 记录每个包内文件的 SHA256、大小、源文件时间与模型来源，
ZIP SHA256 文件用于检查下载或复制后的归档是否变化。原研究数据、实验、旧包与 Python
环境不包含在内。依赖安装占用空间另计。
'''

INSTALL_PS = r'''$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$python = Join-Path $root '.venv_runtime\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    & py -3 -c "import sys; assert (3,11) <= sys.version_info[:2] < (3,14), 'Python 3.11-3.13 required'"
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11-3.13 with the Windows Python launcher first.' }
    & py -3 -m venv (Join-Path $root '.venv_runtime')
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
}
& $python -m pip install -r (Join-Path $root 'requirements-runtime.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check network access and retry.' }
Write-Output 'Runtime environment ready.'
'''

START_PS = r'''param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv_runtime\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Runtime Python missing. Run the environment installer first.' }
$env:PYTHONPATH = Join-Path $root 'src'
$port = $null
foreach ($candidate in 8510..8530) {
    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $candidate)
    try { $probe.Start(); $port = $candidate; break } catch [System.Net.Sockets.SocketException] { continue } finally { $probe.Stop() }
}
if ($null -eq $port) { throw 'No available local port between 8510 and 8530.' }
$logDir = Join-Path $root 'artifacts\showcase_runtime'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$process = Start-Process -FilePath $python -ArgumentList @('-m', 'haze_forecast.showcase_server', '--port', "$port") -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "$port.stdout.log") -RedirectStandardError (Join-Path $logDir "$port.stderr.log") -PassThru
$url = "http://127.0.0.1:$port/"
$ready = $false
for ($attempt=0; $attempt -lt 60; $attempt++) {
    if ($process.HasExited) { throw "Website service exited. See $logDir\$port.stderr.log" }
    try { $response = Invoke-WebRequest -Uri ($url + 'api/status') -UseBasicParsing -TimeoutSec 2; if ($response.StatusCode -eq 200) { $ready=$true; break } } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) {
    if (-not $process.HasExited) { $process.Kill() }
    throw "Website did not become ready. The process started by this launcher was stopped. See $logDir\$port.stderr.log."
}
Write-Output "Website: $url"
Write-Output "Process ID: $($process.Id)"
Write-Output "Logs: $logDir"
if (-not $NoBrowser) { Start-Process $url }
'''


def generated_files():
    def cmd(script):
        return ('@echo off\r\ncd /d "%~dp0"\r\n'
                f'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\\{script}"\r\n'
                'if errorlevel 1 (echo Operation failed. See the error above. & pause & exit /b 1)\r\n'
                'pause\r\n').encode('ascii')
    return {
        'README.md': README.encode('utf-8'),
        'requirements-runtime.txt': b'-e .\npyarrow>=16,<24\n',
        'scripts/install_runtime.ps1': INSTALL_PS.encode('ascii'),
        'scripts/start_runtime.ps1': START_PS.encode('ascii'),
        '安装运行环境.cmd': cmd('install_runtime.ps1'),
        '启动运行版.cmd': cmd('start_runtime.ps1'),
    }


def digest_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def build_runtime_package(root, output_dir):
    root = Path(root)
    for parent in (root, *root.parents):
        if parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
            raise ValueError(f'Source root link rejected: {parent}')
    root = root.resolve(strict=True)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output must be new: {output}')
    # Output-parent links must not redirect artifacts elsewhere.
    for parent in (output, *output.parents):
        if parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
            raise ValueError(f'Output link rejected: {parent}')
    selected = source_files(root)
    generated = generated_files()
    snapshots = {name: file_signature(path) for name, path in selected.items()}
    hashes = {name: digest_file(path) for name, path in selected.items()}
    staged = {}
    adaptations = {}
    workbench = 'project-showcase/workbench.html'
    if workbench in selected:
        original = selected[workbench].read_bytes()
        marker = b'id="tab-multimodal"'
        if original.count(marker) != 1:
            raise ValueError('Expected exactly one multimodal workbench tab')
        staged[workbench] = original.replace(marker, marker + b' hidden', 1)
        adaptations[workbench] = 'hide_unavailable_multimodal_tab'
    workbench_js = 'project-showcase/workbench.js'
    if workbench_js in selected:
        original = selected[workbench_js].read_bytes()
        marker = b"document.querySelectorAll('[role=tab]')"
        if original.count(marker) != 1:
            raise ValueError('Expected exactly one workbench keyboard tab selector')
        staged[workbench_js] = original.replace(
            marker, b"document.querySelectorAll('[role=tab]:not([hidden])')", 1)
        adaptations[workbench_js] = 'exclude_hidden_tabs_from_keyboard_navigation'
    total = sum(s[0] for s in snapshots.values()) + sum(map(len, generated.values()))
    total += sum(len(content) - snapshots[name][0] for name, content in staged.items())
    if total >= MAX_BYTES:
        raise ValueError(f'Runtime exceeds 100 MiB budget: {total}')
    for name, path in selected.items():
        if file_signature(path) != snapshots[name]:
            raise RuntimeError(f'Source changed during preflight: {name}')
    output.mkdir(parents=True, exist_ok=False)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
    archive_path = output / f'chengdu_haze_runtime_{stamp}.zip'
    records = []
    with archive_path.open('xb') as handle:
        with zipfile.ZipFile(handle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in selected.items():
                if file_signature(path) != snapshots[name]:
                    raise RuntimeError(f'Source changed before copy: {name}')
                sha = hashlib.sha256()
                copied = 0
                if name in staged:
                    content = staged[name]
                    archive.writestr(name, content)
                    sha.update(content)
                    copied = len(content)
                    source_unchanged = digest_file(path) == hashes[name]
                else:
                    with path.open('rb') as source, archive.open(name, 'w') as target:
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                            sha.update(chunk)
                            copied += len(chunk)
                    source_unchanged = copied == snapshots[name][0] and sha.hexdigest() == hashes[name]
                if file_signature(path) != snapshots[name] or not source_unchanged:
                    raise RuntimeError(f'Source changed during copy: {name}')
                records.append(dict(path=name, bytes=copied, sha256=sha.hexdigest(),
                                    staged_sha256=sha.hexdigest(), source_sha256=hashes[name],
                                    source_bytes=snapshots[name][0], source_path=str(path),
                                    source_mtime_ns=snapshots[name][1],
                                    adaptation=adaptations.get(name)))
            for name, content in sorted(generated.items()):
                safe_relative(name)
                archive.writestr(name, content)
                records.append(dict(path=name, bytes=len(content),
                                    sha256=hashlib.sha256(content).hexdigest(), generated=True))
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'Archive CRC verification failed: {bad}')
        if set(archive.namelist()) != {r['path'] for r in records}:
            raise RuntimeError('Archive file list mismatch')
        for record in records:
            with archive.open(record['path']) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != record['sha256'] or archive.getinfo(record['path']).file_size != record['bytes']:
                raise RuntimeError(f'Archive SHA256 verification failed: {record["path"]}')
    for name, path in selected.items():
        if file_signature(path) != snapshots[name] or digest_file(path) != hashes[name]:
            raise RuntimeError(f'Source changed after copy: {name}')
    archive_sha = digest_file(archive_path)
    manifest = dict(schema_version=1, created_utc=stamp, source_root=str(root),
                    archive=archive_path.name, archive_sha256=archive_sha,
                    uncompressed_bytes=total, files=records,
                    source_model_sha256={n: hashes[n] for n in REQUIRED_FILES[:3]},
                    multimodal_available=False)
    manifest_path = output / f'{archive_path.stem}.manifest.json'
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    sha_path = output / f'{archive_path.name}.sha256'
    with sha_path.open('x', encoding='ascii') as stream:
        stream.write(f'{archive_sha}  {archive_path.name}\n')
    return dict(archive=str(archive_path.resolve()), manifest=str(manifest_path.resolve()),
                sha256_file=str(sha_path.resolve()), archive_sha256=archive_sha,
                archive_bytes=archive_path.stat().st_size, uncompressed_bytes=total,
                file_count=len(records), source_model_sha256=manifest['source_model_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.root / 'releases/runtime_20261002'
    print(json.dumps(build_runtime_package(args.root, output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
