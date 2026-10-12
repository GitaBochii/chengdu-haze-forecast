param([switch]$NoBrowser, [switch]$Multimodal, [string]$PythonExecutable)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'showcase_process.ps1')
$lock = Enter-ShowcaseLock $root
$process = $null
try {
    $owned = Get-ShowcaseOwnedProcess $root
    if ($null -ne $owned -and (Test-ShowcaseReady ([int]$owned.State.port))) {
        $url = "http://127.0.0.1:$($owned.State.port)/"
        Write-Output "Website: $url"
        Write-Output "Process ID: $($owned.Process.Id)"
        Write-Output 'Reused existing website service.'
        if (-not $NoBrowser) {
            try { Start-Process $url } catch { Write-Warning "Open this website address in your browser: $url" }
        }
        return
    }
    if ($null -ne $owned) {
        throw 'The existing project service is not responding. Run the stop launcher, then retry.'
    }
    $python = if ($PythonExecutable) { $PythonExecutable } else { Join-Path $root '.venv\Scripts\python.exe' }
    if ($Multimodal -and -not $PythonExecutable) { $python = Join-Path $root '.venv_air_models\Scripts\python.exe' }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw 'Project Python environment missing. Run the installation launcher first.'
    }
    $python = (Resolve-Path -LiteralPath $python).ProviderPath
    $port = $null
    foreach ($candidate in 8510..8530) {
        $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $candidate)
        try { $probe.Start(); $port = $candidate; break }
        catch [System.Net.Sockets.SocketException] { continue }
        finally { $probe.Stop() }
    }
    if ($null -eq $port) { throw 'No available local port between 8510 and 8530.' }
    $logDir = Join-Path $root 'artifacts\showcase_runtime'
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff')
    $stderr = Join-Path $logDir "$stamp.stderr.log"
    $oldPath, $oldEncoding = $env:PYTHONPATH, $env:PYTHONIOENCODING
    try {
        $env:PYTHONPATH = Join-Path $root 'src'
        $env:PYTHONIOENCODING = 'utf-8'
        $process = Start-Process -FilePath $python -ArgumentList @('-m', 'haze_forecast.showcase_server', '--port', "$port") -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "$stamp.stdout.log") -RedirectStandardError $stderr -PassThru
    } finally {
        $env:PYTHONPATH = $oldPath
        $env:PYTHONIOENCODING = $oldEncoding
    }
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($process.HasExited) { throw "Website service exited. See $stderr" }
        if (Test-ShowcaseReady $port) { $ready = $true; break }
        Start-Sleep -Seconds 1
    }
    if (-not $ready) { throw "Website startup timed out. See $stderr" }
    $state = @{schema=1; root=(Get-ShowcaseRoot $root); pid=$process.Id; port=$port;
        start_time_ticks=[string]$process.StartTime.ToUniversalTime().Ticks; image_path=$process.Path}
    $state | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir 'server.json') -Encoding UTF8
    Write-Output "Website: http://127.0.0.1:$port/"
    Write-Output "Process ID: $($process.Id)"
    Write-Output "Logs: $logDir"
    if (-not $NoBrowser) {
        try { Start-Process "http://127.0.0.1:$port/" }
        catch { Write-Warning "Open this website address in your browser: http://127.0.0.1:$port/" }
    }
} catch {
    if ($null -ne $process -and -not $process.HasExited) { Stop-ShowcaseProcessTree $process }
    throw
} finally { $lock.Dispose() }
