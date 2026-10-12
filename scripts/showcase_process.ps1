function Get-ShowcaseRoot([string]$Root) {
    return [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
}

function Enter-ShowcaseLock([string]$Root) {
    $directory = Join-Path (Get-ShowcaseRoot $Root) 'artifacts\showcase_runtime'
    [void](New-Item -ItemType Directory -Path $directory -Force)
    try {
        return [System.IO.File]::Open((Join-Path $directory 'startup.lock'),
            [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None)
    } catch {
        throw 'Another start or stop operation is running. Wait for it to finish and retry.'
    }
}

function Get-ShowcaseOwnedProcess([string]$Root) {
    $rootPath = Get-ShowcaseRoot $Root
    $path = Join-Path $rootPath 'artifacts\showcase_runtime\server.json'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
    try {
        $state = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($state.schema -ne 1 -or [int]$state.pid -le 0 -or
            [int]$state.port -lt 8510 -or [int]$state.port -gt 8530 -or
            -not $state.root -or -not $state.image_path -or -not $state.start_time_ticks) {
            return $null
        }
        if (-not [string]::Equals((Get-ShowcaseRoot $state.root), $rootPath,
                [System.StringComparison]::OrdinalIgnoreCase)) { return $null }
        $process = Get-Process -Id ([int]$state.pid) -ErrorAction Stop
        if ($process.HasExited -or $process.StartTime.ToUniversalTime().Ticks -ne [long]$state.start_time_ticks) {
            return $null
        }
        if (-not [string]::Equals($process.Path, [string]$state.image_path,
                [System.StringComparison]::OrdinalIgnoreCase)) { return $null }
        if ([System.IO.Path]::GetFileName($process.Path) -notin @('python.exe', 'python3.exe')) { return $null }
        $details = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $([int]$state.pid)" -ErrorAction Stop
        $command = '-m\s+haze_forecast\.showcase_server\s+--port\s+' + [int]$state.port + '(?:\s|$)'
        if (-not $details.CommandLine -or $details.CommandLine -notmatch $command) { return $null }
        return [pscustomobject]@{ Process = $process; State = $state }
    } catch {
        return $null
    }
}

function Stop-ShowcaseProcessTree([System.Diagnostics.Process]$Process) {
    if ($Process.HasExited) { return }
    # Windows virtual-environment launchers can have a separate listening child.
    & (Join-Path $env:SystemRoot 'System32\taskkill.exe') /PID $Process.Id /T /F | Out-Host
    if ($LASTEXITCODE -ne 0 -and -not $Process.HasExited) { throw 'Could not stop the project process tree.' }
    if (-not $Process.WaitForExit(10000)) { throw 'The project process has not stopped yet.' }
}

function Test-ShowcaseReady([int]$Port) {
    try {
        $status = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/status" -TimeoutSec 2
        return ($status.ready -eq $true -and [int]$status.history_records -gt 0)
    } catch { return $false }
}
