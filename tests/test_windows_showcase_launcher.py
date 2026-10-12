import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
HELPER = SCRIPTS / "showcase_process.ps1"


def powershell(code):
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", code],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def test_launcher_files_exist():
    for name in ("showcase_process.ps1", "start_showcase.ps1", "stop_showcase.ps1"):
        assert (SCRIPTS / name).is_file(), name
    assert (ROOT / "停止网页.cmd").is_file()


def test_cmd_entries_preserve_windows_line_endings():
    for path in ROOT.glob("*.cmd"):
        data = path.read_bytes()
        assert data.startswith(b"@echo off\r\n"), path.name
        assert b"\n" not in data.replace(b"\r\n", b""), path.name


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher")
def test_powershell_51_parser_accepts_launcher():
    assert HELPER.is_file()
    result = powershell(
        "$ErrorActionPreference='Stop'; "
        + f"Get-ChildItem -LiteralPath {quote(SCRIPTS)} -Filter '*showcase*.ps1' | ForEach-Object {{ "
        "$tokens=$null; $errors=$null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile($_.FullName,[ref]$tokens,[ref]$errors); "
        "if($errors.Count) {throw ($errors | Out-String)} }; 'parsed'"
    )
    assert result == "parsed"


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher")
@pytest.mark.parametrize("contents", ["not-json", "{}", '{"schema":1,"pid":0}', '{"schema":2}'])
def test_invalid_state_never_identifies_a_process(tmp_path, contents):
    assert HELPER.is_file()
    runtime = tmp_path / "artifacts/showcase_runtime"
    runtime.mkdir(parents=True)
    (runtime / "server.json").write_text(contents, encoding="utf-8")
    result = powershell(
        f". {quote(HELPER)}; $p=Get-ShowcaseOwnedProcess -Root {quote(tmp_path)}; "
        "if($null -ne $p){throw 'Accepted unowned process'}; 'safe'"
    )
    assert result == "safe"


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher")
def test_foreign_root_and_reused_pid_are_rejected(tmp_path):
    assert HELPER.is_file()
    runtime = tmp_path / "artifacts/showcase_runtime"
    runtime.mkdir(parents=True)
    code = (
        f". {quote(HELPER)}; $r={quote(tmp_path)}; $p=Get-Process -Id $PID; "
        "$state=@{schema=1;root='C:\\foreign-project';pid=$PID;port=8510;"
        "start_time_ticks=[string]$p.StartTime.ToUniversalTime().Ticks;image_path=$p.Path}; "
        "$state | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $r 'artifacts/showcase_runtime/server.json'); "
        "if($null -ne (Get-ShowcaseOwnedProcess -Root $r)){throw 'Accepted foreign root'}; "
        "$state.root=$r; $state.start_time_ticks='1'; "
        "$state | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $r 'artifacts/showcase_runtime/server.json'); "
        "if($null -ne (Get-ShowcaseOwnedProcess -Root $r)){throw 'Accepted reused PID'}; 'safe'"
    )
    assert powershell(code) == "safe"


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher")
def test_out_of_range_port_is_rejected(tmp_path):
    assert HELPER.is_file()
    runtime = tmp_path / "artifacts/showcase_runtime"
    runtime.mkdir(parents=True)
    code = (
        f". {quote(HELPER)}; $r={quote(tmp_path)}; $p=Get-Process -Id $PID; "
        "$state=@{schema=1;root=$r;pid=$PID;port=443;"
        "start_time_ticks=[string]$p.StartTime.ToUniversalTime().Ticks;image_path=$p.Path}; "
        "$state | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $r 'artifacts/showcase_runtime/server.json'); "
        "if($null -ne (Get-ShowcaseOwnedProcess -Root $r)){throw 'Accepted unexpected port'}; 'safe'"
    )
    assert powershell(code) == "safe"


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher")
def test_matching_marker_cannot_claim_a_non_python_process(tmp_path):
    assert HELPER.is_file()
    runtime = tmp_path / "artifacts/showcase_runtime"
    runtime.mkdir(parents=True)
    code = (
        f". {quote(HELPER)}; $r={quote(tmp_path)}; $p=Get-Process -Id $PID; "
        "$state=@{schema=1;root=$r;pid=$PID;port=8510;"
        "start_time_ticks=[string]$p.StartTime.ToUniversalTime().Ticks;image_path=$p.Path}; "
        "$state | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $r 'artifacts/showcase_runtime/server.json'); "
        "if($null -ne (Get-ShowcaseOwnedProcess -Root $r)){throw 'Accepted unrelated process'}; 'safe'"
    )
    assert powershell(code) == "safe"


def test_startup_and_stop_share_an_ownership_lock():
    start = (SCRIPTS / "start_showcase.ps1").read_text(encoding="utf-8-sig")
    stop = (SCRIPTS / "stop_showcase.ps1").read_text(encoding="utf-8-sig")
    assert "Get-ShowcaseOwnedProcess" in start
    assert "Get-ShowcaseOwnedProcess" in stop
    assert "Enter-ShowcaseLock" in start and "Enter-ShowcaseLock" in stop
    assert "Stop-Process -Name" not in stop
    assert "-WindowStyle Hidden" in start
    assert "PythonExecutable" in start
