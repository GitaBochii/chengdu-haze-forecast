param([switch]$NoBrowser, [switch]$Multimodal)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if ($Multimodal) { $python = Join-Path $root '.venv_air_models\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $python)) { throw 'Project Python environment missing: run the environment installer first.' }
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
