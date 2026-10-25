<#
.SYNOPSIS
    Builds QtRtspViewer on Windows (MSVC) and creates a portable folder + zip.

.DESCRIPTION
    Output: zz_packaging\dist\QtRtspViewer-<ver>-win64\ and ...-portable.zip
    Run from a "x64 Native Tools Command Prompt for VS 2022" (or any shell where
    cmake and cl are usable).

.PARAMETER QtDir
    Qt kit root, e.g. C:\Qt\6.10.2\msvc2022_64 (default: $env:QT_DIR or that path).
.PARAMETER OpenCvDir
    OpenCV "build" directory containing OpenCVConfig.cmake (default: $env:OPENCV_DIR).
.PARAMETER FfmpegDir
    FFmpeg *shared* build root with bin\, include\, lib\ (default: $env:FFMPEG_DIR).
    Provides recording support (libav* DLLs are bundled; the ffmpeg.exe tool is not needed).
.PARAMETER DownloadFfmpeg
    Download a shared GPL FFmpeg build (see -FfmpegUrl) when -FfmpegDir is not given.
.PARAMETER NoFfmpeg
    Build and package without FFmpeg.
.PARAMETER SkipBuild
    Reuse the existing build in build-windows\.
.PARAMETER NoZip
    Only create the staging folder.
#>
[CmdletBinding()]
param(
    [string]$QtDir = $(if ($env:QT_DIR) { $env:QT_DIR } else { 'C:\Qt\6.10.2\msvc2022_64' }),
    [string]$OpenCvDir = $env:OPENCV_DIR,
    [string]$FfmpegDir = $env:FFMPEG_DIR,
    # FFmpeg 8 DLLs (avcodec-62) coexist with the FFmpeg 7 DLLs (avcodec-61) that Qt multimedia ships.
    [string]$FfmpegUrl = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip',
    [switch]$DownloadFfmpeg,
    [switch]$NoFfmpeg,
    [switch]$SkipBuild,
    [switch]$NoZip
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$ScriptDir = $PSScriptRoot
$SrcDir    = Split-Path -Parent $ScriptDir
$BuildDir  = Join-Path $ScriptDir 'build-windows'
$ToolsDir  = Join-Path $ScriptDir 'tools'
$DistDir   = Join-Path $ScriptDir 'dist'
$AppName   = 'QtRtspViewer'

function Assert-Exit([string]$what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

$cmakeText = Get-Content (Join-Path $SrcDir 'CMakeLists.txt') -Raw
$Version = if ($cmakeText -match 'project\(\w+\s+VERSION\s+([\d.]+)') { $Matches[1] } else { '0.0.0' }

$Stage = Join-Path $DistDir "$AppName-$Version-win64"
$Zip   = Join-Path $DistDir "$AppName-$Version-win64-portable.zip"

# ── Validate inputs ──────────────────────────────────────────────────────────
$windeployqt = Join-Path $QtDir 'bin\windeployqt.exe'
if (-not (Test-Path $windeployqt)) { throw "windeployqt.exe not found. Check -QtDir ($QtDir)." }
if (-not $OpenCvDir -or -not (Test-Path $OpenCvDir)) { throw 'Set -OpenCvDir (or $env:OPENCV_DIR) to the OpenCV build directory.' }
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw 'cmake not found in PATH.' }

New-Item -ItemType Directory -Force $ToolsDir, $DistDir | Out-Null

# ── FFmpeg ───────────────────────────────────────────────────────────────────
if ($NoFfmpeg) {
    $FfmpegDir = $null
} elseif (-not $FfmpegDir -and $DownloadFfmpeg) {
    $ffRoot = Join-Path $ToolsDir 'ffmpeg'
    $existing = Get-ChildItem $ffRoot -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $existing) {
        Write-Host "==> Downloading FFmpeg: $FfmpegUrl"
        $zipPath = Join-Path $ToolsDir 'ffmpeg.zip'
        Invoke-WebRequest -UseBasicParsing -Uri $FfmpegUrl -OutFile $zipPath
        Expand-Archive -Path $zipPath -DestinationPath $ffRoot -Force
        Remove-Item $zipPath
        $existing = Get-ChildItem $ffRoot -Directory | Select-Object -First 1
    }
    $FfmpegDir = $existing.FullName
}
if ($FfmpegDir) {
    foreach ($sub in 'include\libavcodec\avcodec.h', 'lib') {
        if (-not (Test-Path (Join-Path $FfmpegDir $sub))) { throw "FFmpeg dir '$FfmpegDir' is missing '$sub' (a shared dev build is required)." }
    }
    Write-Host "==> Using FFmpeg: $FfmpegDir"
} else {
    Write-Warning 'No FFmpeg: recording support will be disabled in this build.'
}

# ── Build ────────────────────────────────────────────────────────────────────
if (-not $SkipBuild) {
    Write-Host '==> Configuring'
    $cfg = @('-S', $SrcDir, '-B', $BuildDir, "-DCMAKE_PREFIX_PATH=$QtDir", "-DOpenCV_DIR=$OpenCvDir")
    if ($FfmpegDir) { $cfg += "-DFFMPEG_DIR=$FfmpegDir" }
    & cmake @cfg
    Assert-Exit 'cmake configure'

    Write-Host '==> Building (Release)'
    & cmake --build $BuildDir --config Release --parallel
    Assert-Exit 'cmake build'
}

$exe = Get-ChildItem (Join-Path $BuildDir 'bin') -Recurse -Filter "$AppName.exe" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $exe) { throw "$AppName.exe not found under $BuildDir\bin" }

# ── Stage ────────────────────────────────────────────────────────────────────
Write-Host "==> Staging into $Stage"
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory $Stage | Out-Null

Copy-Item $exe.FullName $Stage
# The app loads the cascade from <exe dir>\opencv\
Copy-Item (Join-Path $SrcDir 'opencv') (Join-Path $Stage 'opencv') -Recurse
Copy-Item (Join-Path $SrcDir 'LICENSE') $Stage

Write-Host '==> Running windeployqt'
& $windeployqt --release --no-translations --no-opengl-sw --compiler-runtime (Join-Path $Stage "$AppName.exe")
Assert-Exit 'windeployqt'

# OpenCV runtime (the app does not use cv::VideoCapture, so no videoio plugins)
$world = Get-ChildItem $OpenCvDir -Recurse -Filter 'opencv_world*.dll' |
    Where-Object { $_.Name -match '^opencv_world\d+\.dll$' } | Select-Object -First 1
if (-not $world) { throw "Release opencv_world*.dll not found under $OpenCvDir" }
Write-Host "==> Bundling OpenCV from $($world.DirectoryName)"
Copy-Item $world.FullName $Stage

# FFmpeg runtime. Copied last so the app's link-time DLLs win over Qt's bundled ones.
if ($FfmpegDir) {
    Write-Host '==> Bundling FFmpeg'
    foreach ($f in Get-ChildItem (Join-Path $FfmpegDir 'bin') -Filter '*.dll') {
        $dest = Join-Path $Stage $f.Name
        if ((Test-Path $dest) -and ((Get-FileHash $dest).Hash -ne (Get-FileHash $f.FullName).Hash)) {
            Write-Warning "Overwriting $($f.Name) deployed by windeployqt with the FFmpeg build's copy"
        }
        Copy-Item $f.FullName $dest -Force
    }
}

# ── Zip ──────────────────────────────────────────────────────────────────────
if (-not $NoZip) {
    Write-Host "==> Creating $Zip"
    if (Test-Path $Zip) { Remove-Item $Zip }
    Compress-Archive -Path $Stage -DestinationPath $Zip -CompressionLevel Optimal
    Write-Host "==> Done: $Zip"
} else {
    Write-Host "==> Done: $Stage"
}
