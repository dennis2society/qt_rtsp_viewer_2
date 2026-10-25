<#
.SYNOPSIS
    Creates an Inno Setup installer for QtRtspViewer (stages via build_windows_portable.ps1 first).

.DESCRIPTION
    Output: zz_packaging\dist\QtRtspViewer-<ver>-win64-setup.exe
    Requires Inno Setup 6 (ISCC.exe in PATH or in Program Files). Build parameters are
    forwarded to build_windows_portable.ps1; see that script for details.

.PARAMETER ReuseStage
    Skip building/staging and package the existing dist\QtRtspViewer-<ver>-win64 folder.
#>
[CmdletBinding()]
param(
    [string]$QtDir,
    [string]$OpenCvDir,
    [string]$FfmpegDir,
    [string]$FfmpegUrl,
    [switch]$DownloadFfmpeg,
    [switch]$NoFfmpeg,
    [switch]$SkipBuild,
    [switch]$ReuseStage
)

$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
$SrcDir    = Split-Path -Parent $ScriptDir
$DistDir   = Join-Path $ScriptDir 'dist'
$AppName   = 'QtRtspViewer'

$cmakeText = Get-Content (Join-Path $SrcDir 'CMakeLists.txt') -Raw
$Version = if ($cmakeText -match 'project\(\w+\s+VERSION\s+([\d.]+)') { $Matches[1] } else { '0.0.0' }
$Stage = Join-Path $DistDir "$AppName-$Version-win64"

# ── Locate ISCC ──────────────────────────────────────────────────────────────
$iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $iscc = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $iscc) { throw 'ISCC.exe not found. Install Inno Setup 6 (https://jrsoftware.org/isinfo.php) or add it to PATH.' }

# ── Stage ────────────────────────────────────────────────────────────────────
if (-not $ReuseStage) {
    $fwd = @{ NoZip = $true }
    foreach ($k in 'QtDir', 'OpenCvDir', 'FfmpegDir', 'FfmpegUrl') {
        if ($PSBoundParameters.ContainsKey($k)) { $fwd[$k] = $PSBoundParameters[$k] }
    }
    foreach ($k in 'DownloadFfmpeg', 'NoFfmpeg', 'SkipBuild') {
        if ($PSBoundParameters.ContainsKey($k) -and $PSBoundParameters[$k]) { $fwd[$k] = $true }
    }
    & (Join-Path $ScriptDir 'build_windows_portable.ps1') @fwd
}
if (-not (Test-Path (Join-Path $Stage "$AppName.exe"))) { throw "Staged app not found in $Stage" }

# ── Compile installer ────────────────────────────────────────────────────────
Write-Host '==> Running ISCC'
& $iscc "/DAppVersion=$Version" "/DStageDir=$Stage" "/DSourceRoot=$SrcDir" "/DOutDir=$DistDir" (Join-Path $ScriptDir 'QtRtspViewer.iss')
if ($LASTEXITCODE -ne 0) { throw "ISCC failed (exit code $LASTEXITCODE)" }

Write-Host "==> Done: $(Join-Path $DistDir "$AppName-$Version-win64-setup.exe")"
