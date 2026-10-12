$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'showcase_process.ps1')
$lock = Enter-ShowcaseLock $root
try {
    $owned = Get-ShowcaseOwnedProcess $root
    if ($null -eq $owned) {
        Write-Output 'No verified website process belongs to this project. No process was stopped.'
        return
    }
    Stop-ShowcaseProcessTree $owned.Process
    Remove-Item -LiteralPath (Join-Path $root 'artifacts\showcase_runtime\server.json') -ErrorAction SilentlyContinue
    Write-Output "Website stopped. Process ID: $($owned.Process.Id)"
} finally { $lock.Dispose() }
