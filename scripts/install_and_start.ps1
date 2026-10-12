param(
    [switch]$NoBrowser,
    [switch]$InstallOnly,
    [string]$PythonExecutable
)

$ErrorActionPreference = 'Stop'
$script:HazeInstallerRoot = Split-Path -Parent $PSScriptRoot
$script:HazePythonDownload = 'https://www.python.org/downloads/windows/'

function Get-HazePythonInfo {
    param([string]$Executable)
    if (-not $Executable -or $Executable -match '[\\/]Microsoft[\\/]WindowsApps[\\/]') {
        return $null
    }
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $null }
    $probe = "import json,struct,sys; print(json.dumps(dict(version=list(sys.version_info[:3]),bits=struct.calcsize('P')*8,executable=sys.executable)))"
    try {
        $raw = & $Executable -I -X utf8 -c $probe 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        return ($raw -join "`n" | ConvertFrom-Json)
    } catch { return $null }
}

function Test-HazeSupportedPython {
    param($Info)
    return ($null -ne $Info -and $Info.bits -eq 64 -and $Info.version[0] -eq 3 -and
        $Info.version[1] -in @(11, 12, 13))
}

function Get-HazePythonCandidates {
    $candidates = @()
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $previous = @{}
        foreach ($name in @('PYTHON_MANAGER_AUTOMATIC_INSTALL', 'PYLAUNCHER_ALLOW_INSTALL', 'PYLAUNCHER_ALWAYS_INSTALL')) {
            $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        }
        try {
            [Environment]::SetEnvironmentVariable('PYTHON_MANAGER_AUTOMATIC_INSTALL', 'false', 'Process')
            [Environment]::SetEnvironmentVariable('PYLAUNCHER_ALLOW_INSTALL', $null, 'Process')
            [Environment]::SetEnvironmentVariable('PYLAUNCHER_ALWAYS_INSTALL', $null, 'Process')
            foreach ($version in @('-3.12', '-3.11', '-3.13')) {
                try {
                    $path = & $launcher.Source $version -I -c 'import sys; print(sys.executable)' 2>$null
                    if ($LASTEXITCODE -eq 0 -and $path) { $candidates += ($path -join '').Trim() }
                } catch { }
            }
        } finally {
            foreach ($name in $previous.Keys) {
                [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process')
            }
        }
    }
    foreach ($command in @('python.exe', 'python3.exe')) {
        foreach ($found in @(Get-Command $command -All -ErrorAction SilentlyContinue)) {
            if ($found.Source) { $candidates += $found.Source }
        }
    }
    foreach ($registryRoot in @('HKCU:\Software\Python\PythonCore',
        'HKLM:\Software\Python\PythonCore', 'HKLM:\Software\WOW6432Node\Python\PythonCore')) {
        if (-not (Test-Path -LiteralPath $registryRoot)) { continue }
        foreach ($key in @(Get-ChildItem -LiteralPath $registryRoot -ErrorAction SilentlyContinue)) {
            $install = Join-Path $key.PSPath 'InstallPath'
            if (-not (Test-Path -LiteralPath $install)) { continue }
            $item = Get-Item -LiteralPath $install
            $path = $item.GetValue('ExecutablePath')
            if (-not $path -and $item.GetValue('')) { $path = Join-Path $item.GetValue('') 'python.exe' }
            if ($path) { $candidates += $path }
        }
    }
    return @($candidates | Select-Object -Unique)
}

function Resolve-HazePython {
    param([string]$Root, [string]$ExplicitPython, [switch]$NoBrowser)
    $venvPython = Join-Path $Root '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $info = Get-HazePythonInfo -Executable $venvPython
        if (Test-HazeSupportedPython $info) { return $venvPython }
        throw "The existing .venv interpreter is invalid or unsupported. It was not changed. Rename .venv yourself before retrying."
    }
    $candidates = if ($ExplicitPython) { @($ExplicitPython) } else { @(Get-HazePythonCandidates) }
    foreach ($candidate in $candidates) {
        $info = Get-HazePythonInfo -Executable $candidate
        if (Test-HazeSupportedPython $info) { return $info.executable }
    }
    if (-not $NoBrowser) { Start-Process $script:HazePythonDownload }
    throw "No supported 64-bit Python found. Install Python 3.12 (or 3.11/3.13) from $script:HazePythonDownload and retry. No system Python was installed or changed."
}

function Get-HazeProjectHash {
    param([string]$Root)
    $stream = [IO.File]::OpenRead((Join-Path $Root 'pyproject.toml'))
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '') }
    finally { $sha.Dispose(); $stream.Dispose() }
}

function Test-HazeEnvironmentHealth {
    param([string]$PythonExecutable, [string]$Root)
    $info = Get-HazePythonInfo -Executable $PythonExecutable
    if (-not (Test-HazeSupportedPython $info)) { return $false }
    $health = @'
import importlib, importlib.metadata as md, json, os, pathlib, sys, tomllib
from urllib.parse import urlparse
from urllib.request import url2pathname
from packaging.requirements import Requirement
root = pathlib.Path(sys.argv[1]).resolve()
project = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']
for dependency in project['dependencies']:
    requirement = Requirement(dependency)
    assert md.version(requirement.name) in requirement.specifier, dependency
for module in ['pandas', 'numpy', 'sklearn', 'lightgbm', 'catboost', 'xgboost', 'optuna', 'requests', 'joblib', 'streamlit', 'plotly', 'pyarrow', 'haze_forecast']:
    importlib.import_module(module)
import torch
assert torch.__version__.split('+')[0] == '2.7.1'
assert torch.version.cuda is None
assert torch.tensor([1.0], device='cpu').sum().item() == 1.0
distribution = md.distribution('chengdu-haze-forecast')
direct = json.loads(distribution.read_text('direct_url.json') or '{}')
assert direct.get('dir_info', {}).get('editable') is True
url = urlparse(direct['url'])
assert url.scheme == 'file'
installed = pathlib.Path(url2pathname(url.path)).resolve()
assert os.path.normcase(str(installed)) == os.path.normcase(str(root))
module_root = pathlib.Path(importlib.import_module('haze_forecast').__file__).resolve().parents[2]
assert os.path.normcase(str(module_root)) == os.path.normcase(str(root))
print('HAZE_ENVIRONMENT_OK')
'@
    try {
        $output = & $PythonExecutable -I -X utf8 -c $health $Root 2>$null
        return ($LASTEXITCODE -eq 0 -and ($output -join "`n").Trim() -eq 'HAZE_ENVIRONMENT_OK')
    } catch { return $false }
}

function Invoke-HazePython {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE. Fix the error shown above and retry." }
}

function Ensure-HazeEnvironment {
    param([string]$Root, [string]$SourcePython)
    $venv = Join-Path $Root '.venv'
    $python = Join-Path $venv 'Scripts\python.exe'
    $runtime = Join-Path $Root 'artifacts\showcase_runtime'
    New-Item -ItemType Directory -Path $runtime -Force | Out-Null
    $marker = Join-Path $runtime 'install-state.json'
    $hash = Get-HazeProjectHash -Root $Root
    $state = $null
    if (Test-Path -LiteralPath $marker -PathType Leaf) {
        try { $state = Get-Content -LiteralPath $marker -Raw -Encoding UTF8 | ConvertFrom-Json } catch { }
    }
    if ($state -and $state.schema_version -eq 1 -and $state.project_sha256 -eq $hash -and
        (Test-HazeEnvironmentHealth -PythonExecutable $python -Root $Root)) {
        Write-Host 'Dependencies are ready. Reusing this project environment.'
        return $python
    }
    if ((Test-Path -LiteralPath $venv) -and -not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "The existing .venv folder is incomplete. It was not deleted or overwritten. Rename it yourself before retrying."
    }
    if (Test-Path -LiteralPath $python -PathType Leaf) {
        if (-not (Test-HazeSupportedPython (Get-HazePythonInfo -Executable $python))) {
            throw "The existing .venv interpreter is invalid. It was not changed. Rename .venv yourself before retrying."
        }
    } else {
        Write-Host 'Creating the project-only .venv environment...'
        Invoke-HazePython -Executable $SourcePython -Arguments @('-m', 'venv', $venv)
    }
    if (Test-Path -LiteralPath $marker -PathType Leaf) { Remove-Item -LiteralPath $marker }
    Write-Host 'Installing CPU PyTorch and webpage dependencies. The first installation needs Internet access...'
    Invoke-HazePython -Executable $python -Arguments @('-m', 'pip', 'install', '--upgrade', '--retries', '3',
        'torch==2.7.1+cpu', '--index-url', 'https://download.pytorch.org/whl/cpu')
    Invoke-HazePython -Executable $python -Arguments @('-m', 'pip', 'install', '--retries', '3', '-e', "$Root[multimodal]")
    if (-not (Test-HazeEnvironmentHealth -PythonExecutable $python -Root $Root)) {
        throw 'Installation finished but dependency validation failed. No ready marker was written. Check the network/packages and retry.'
    }
    $state = [ordered]@{schema_version=1; project_sha256=$hash; python=$python; torch='2.7.1+cpu'}
    $state | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding UTF8
    Write-Host 'Installation and dependency validation completed.'
    return $python
}

function Invoke-HazeInstaller {
    param([string]$Root, [switch]$NoBrowser, [switch]$InstallOnly, [string]$PythonExecutable)
    $runtime = Join-Path $Root 'artifacts\showcase_runtime'
    New-Item -ItemType Directory -Path $runtime -Force | Out-Null
    $lockPath = Join-Path $runtime 'install.lock'
    $lock = $null
    try {
        try { $lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None) }
        catch { throw 'Another installation/start is in progress for this project. Wait for that window to finish, then retry.' }
        $source = Resolve-HazePython -Root $Root -ExplicitPython $PythonExecutable -NoBrowser:$NoBrowser
        $python = Ensure-HazeEnvironment -Root $Root -SourcePython $source
        if (-not $InstallOnly) {
            & (Join-Path $Root 'scripts\start_showcase.ps1') -PythonExecutable $python -NoBrowser:$NoBrowser
        }
    } finally { if ($null -ne $lock) { $lock.Dispose() } }
}

if ($MyInvocation.InvocationName -ne '.') {
    [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
    $OutputEncoding = [Console]::OutputEncoding
    try {
        Invoke-HazeInstaller -Root $script:HazeInstallerRoot -NoBrowser:$NoBrowser -InstallOnly:$InstallOnly -PythonExecutable $PythonExecutable
        exit 0
    } catch {
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
}
