"""Exercise the Windows installer without downloading packages."""

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_and_start.ps1"
pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows launcher")


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def run_ps(code, *, load=True):
    if load:
        code = f". {ps_quote(INSTALLER)}; " + code
    code = "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); " + code
    encoded = base64.b64encode(code.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def make_root(tmp_path):
    root = tmp_path / "Chinese \u9879\u76ee with spaces"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='example'\n", encoding="utf-8")
    return root


def test_dot_source_has_no_side_effects():
    assert run_ps("'loaded' | ConvertTo-Json -Compress") == "loaded"


def test_probes_real_existing_python():
    info = run_ps(f"Get-HazePythonInfo -Executable {ps_quote(sys.executable)} | ConvertTo-Json -Compress")
    assert info["bits"] == 64
    assert info["version"][0] == 3
    assert Path(info["executable"]).resolve() == Path(sys.executable).resolve()


def test_rejects_windows_apps_store_stub():
    assert run_ps(
        "[bool](Get-HazePythonInfo -Executable 'C:\\Users\\example\\AppData\\Local\\Microsoft\\WindowsApps\\python.exe') | ConvertTo-Json"
    ) is False


@pytest.mark.parametrize("version,bits", [("3,10,9", 64), ("3,14,0", 64), ("3,12,1", 32)])
def test_rejects_unsupported_interpreter(tmp_path, version, bits):
    root = make_root(tmp_path)
    code = (
        f"function Get-HazePythonInfo {{ [pscustomobject]@{{version=@({version}); bits={bits}; executable='fake'}} }}; "
        "function Get-HazePythonCandidates { @('fake') }; "
        f"$message = try {{ Resolve-HazePython -Root {ps_quote(root)} -NoBrowser | Out-Null; 'accepted' }} "
        "catch { $_.Exception.Message }; $message | ConvertTo-Json -Compress"
    )
    assert "https://www.python.org/downloads/windows/" in run_ps(code)


def test_missing_python_has_actionable_message_without_browser(tmp_path):
    root = make_root(tmp_path)
    code = (
        "function Get-HazePythonCandidates { @() }; "
        "function Start-Process { throw 'browser should not open' }; "
        f"$message = try {{ Resolve-HazePython -Root {ps_quote(root)} -NoBrowser | Out-Null }} "
        "catch { $_.Exception.Message }; $message | ConvertTo-Json -Compress"
    )
    assert "https://www.python.org/downloads/windows/" in run_ps(code)


def test_existing_valid_venv_precedes_candidates(tmp_path):
    root = make_root(tmp_path)
    target = root / ".venv" / "Scripts" / "python.exe"
    target.parent.mkdir(parents=True)
    target.touch()
    code = (
        "function Get-HazePythonCandidates { throw 'must prefer project venv' }; "
        "function Get-HazePythonInfo { param($Executable) [pscustomobject]@{version=@(3,12,1);bits=64;executable=$Executable} }; "
        f"Resolve-HazePython -Root {ps_quote(root)} -NoBrowser | ConvertTo-Json -Compress"
    )
    assert Path(run_ps(code)).resolve() == target.resolve()


def test_broken_existing_venv_is_preserved(tmp_path):
    root = make_root(tmp_path)
    venv = root / ".venv"
    venv.mkdir()
    sentinel = venv / "user-file.txt"
    sentinel.write_text("keep", encoding="utf-8")
    code = (
        "function Invoke-HazePython { throw 'must not execute' }; "
        f"$message = try {{ Ensure-HazeEnvironment -Root {ps_quote(root)} -SourcePython 'fake' | Out-Null }} "
        "catch { $_.Exception.Message }; $message | ConvertTo-Json -Compress"
    )
    assert ".venv" in run_ps(code)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def setup_with_mock_commands(
    root, *, fail=False, repeated=False, initially_healthy=False, health_after_install=True
):
    target = root / ".venv" / "Scripts" / "python.exe"
    code = (
        f"$script:commands = @(); $script:healthy = ${str(initially_healthy).lower()}; "
        "function Get-HazePythonInfo { param($Executable) [pscustomobject]@{version=@(3,12,1);bits=64;executable=$Executable} }; "
        "function Test-HazeEnvironmentHealth { param($PythonExecutable,$Root) $script:healthy }; "
        "function Invoke-HazePython { param($Executable,$Arguments) "
        "$script:commands += ,@($Arguments); "
        "if ($Arguments[0] -eq '-m' -and $Arguments[1] -eq 'venv') { "
        "$scripts = Join-Path $Arguments[2] 'Scripts'; New-Item -ItemType Directory -Path $scripts -Force | Out-Null; "
        "New-Item -ItemType File -Path (Join-Path $scripts 'python.exe') -Force | Out-Null }; "
        + ("if ($Arguments -contains 'install') { throw 'pip failed' }; " if fail else "")
        + f"if ($Arguments -contains '-e') {{ $script:healthy=${str(health_after_install).lower()} }} }}; "
        f"$message = try {{ Ensure-HazeEnvironment -Root {ps_quote(root)} -SourcePython 'fake' | Out-Null; "
        + (f"Ensure-HazeEnvironment -Root {ps_quote(root)} -SourcePython 'fake' | Out-Null; " if repeated else "")
        + "'ok' } catch { $_.Exception.Message }; "
        "$result = @{message=$message;commands=@($script:commands)}; $result | ConvertTo-Json -Depth 6 -Compress"
    )
    return run_ps(code), target


def test_first_install_is_local_and_uses_cpu_index(tmp_path):
    root = make_root(tmp_path)
    result, target = setup_with_mock_commands(root)
    assert result["message"] == "ok"
    assert target.exists()
    commands = result["commands"]
    assert commands[0][:2] == ["-m", "venv"]
    assert Path(commands[0][2]) == root / ".venv"
    cpu = next(args for args in commands if "torch==2.7.1+cpu" in args)
    assert cpu[cpu.index("--index-url") + 1] == "https://download.pytorch.org/whl/cpu"
    assert "--extra-index-url" not in cpu
    assert any("-e" in args and str(root) + "[multimodal]" in args for args in commands)
    assert (root / "artifacts" / "showcase_runtime" / "install-state.json").is_file()


def test_healthy_repeat_does_not_install_again(tmp_path):
    root = make_root(tmp_path)
    result, _ = setup_with_mock_commands(root, repeated=True)
    assert result["message"] == "ok"
    assert len([args for args in result["commands"] if "install" in args]) == 2


def test_failed_install_never_writes_ready_marker(tmp_path):
    root = make_root(tmp_path)
    result, _ = setup_with_mock_commands(root, fail=True)
    assert result["message"] == "pip failed"
    assert not (root / "artifacts" / "showcase_runtime" / "install-state.json").exists()
    assert not any("-e" in args for args in result["commands"])


def test_changed_project_hash_requires_refresh(tmp_path):
    root = make_root(tmp_path)
    result, _ = setup_with_mock_commands(root)
    assert result["message"] == "ok"
    (root / "pyproject.toml").write_text("[project]\nname='changed'\n", encoding="utf-8")
    result, _ = setup_with_mock_commands(root, initially_healthy=True)
    assert len([args for args in result["commands"] if "install" in args]) == 2


def test_healthy_state_is_reused_between_processes(tmp_path):
    root = make_root(tmp_path)
    result, _ = setup_with_mock_commands(root)
    assert result["message"] == "ok"
    result, _ = setup_with_mock_commands(root, initially_healthy=True)
    assert result["message"] == "ok"
    assert not result["commands"]


def test_post_install_health_failure_writes_no_ready_marker(tmp_path):
    root = make_root(tmp_path)
    result, _ = setup_with_mock_commands(root, health_after_install=False)
    assert "validation failed" in result["message"]
    assert not (root / "artifacts" / "showcase_runtime" / "install-state.json").exists()


def test_install_lock_rejects_second_launcher(tmp_path):
    root = make_root(tmp_path)
    runtime = root / "artifacts" / "showcase_runtime"
    runtime.mkdir(parents=True)
    code = (
        f"$lock = [IO.File]::Open({ps_quote(runtime / 'install.lock')}, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None); "
        "function Resolve-HazePython { throw 'must not start another install' }; "
        f"$message = try {{ Invoke-HazeInstaller -Root {ps_quote(root)} -NoBrowser -InstallOnly }} "
        "catch { $_.Exception.Message } finally { $lock.Dispose() }; $message | ConvertTo-Json -Compress"
    )
    assert "Another installation/start is in progress" in run_ps(code)


def test_install_only_does_not_launch_server(tmp_path):
    root = make_root(tmp_path)
    code = (
        "function Resolve-HazePython { 'fake-source' }; "
        "function Ensure-HazeEnvironment { 'fake-env' }; "
        f"Invoke-HazeInstaller -Root {ps_quote(root)} -NoBrowser -InstallOnly; 'ok' | ConvertTo-Json -Compress"
    )
    assert run_ps(code) == "ok"


def test_cmd_files_are_valid_utf8_without_bom():
    for filename in ("\u5b89\u88c5\u5e76\u542f\u52a8\u7f51\u9875.cmd", "\u542f\u52a8\u7f51\u9875.cmd"):
        content = (ROOT / filename).read_bytes()
        assert content.startswith(b"@echo off")
        assert "\u7f51\u9875" in content.decode("utf-8")


def test_py_discovery_disables_automatic_runtime_install_and_restores_settings(tmp_path):
    probe = tmp_path / "fake launcher.ps1"
    observations = tmp_path / "observed.txt"
    probe.write_text(
        f"[IO.File]::AppendAllText({ps_quote(observations)}, "
        "($env:PYTHON_MANAGER_AUTOMATIC_INSTALL + '|' + "
        "[bool](Test-Path Env:PYLAUNCHER_ALLOW_INSTALL) + '|' + "
        "[bool](Test-Path Env:PYLAUNCHER_ALWAYS_INSTALL) + [Environment]::NewLine)); "
        f"Write-Output {ps_quote(sys.executable)}; exit 0",
        encoding="utf-8-sig",
    )
    code = (
        "$env:PYTHON_MANAGER_AUTOMATIC_INSTALL='true'; "
        "$env:PYLAUNCHER_ALLOW_INSTALL='1'; $env:PYLAUNCHER_ALWAYS_INSTALL='1'; "
        f"function Get-Command {{ param($Name,[switch]$All,$ErrorAction) if ($Name -eq 'py.exe') {{ [pscustomobject]@{{Source={ps_quote(probe)}}} }} }}; "
        "Get-HazePythonCandidates | Out-Null; "
        "@{manager=$env:PYTHON_MANAGER_AUTOMATIC_INSTALL;allow=$env:PYLAUNCHER_ALLOW_INSTALL;always=$env:PYLAUNCHER_ALWAYS_INSTALL} | ConvertTo-Json -Compress"
    )
    assert run_ps(code) == {"manager": "true", "allow": "1", "always": "1"}
    assert observations.read_text(encoding="utf-8").splitlines() == ["false|False|False"] * 3


def test_external_python_error_stops_execution():
    message = run_ps(
        f"$message = try {{ Invoke-HazePython -Executable {ps_quote(sys.executable)} -Arguments @('-c','import sys;sys.exit(23)') }} "
        "catch { $_.Exception.Message }; $message | ConvertTo-Json -Compress"
    )
    assert "23" in message


def test_health_check_rejects_python_bound_to_a_different_project(tmp_path):
    other = tmp_path / "other project"
    other.mkdir()
    (other / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    assert run_ps(
        f"Test-HazeEnvironmentHealth -PythonExecutable {ps_quote(sys.executable)} -Root {ps_quote(other)} | ConvertTo-Json"
    ) is False


def test_cmd_launchers_preserve_exit_and_pause_failures():
    for filename in ("\u5b89\u88c5\u5e76\u542f\u52a8\u7f51\u9875.cmd", "\u542f\u52a8\u7f51\u9875.cmd"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert "chcp 65001" in text
        assert 'cd /d "%~dp0"' in text
        assert "install_and_start.ps1" in text
        assert "%errorlevel%" in text
        assert "pause" in text
        assert "exit /b" in text
