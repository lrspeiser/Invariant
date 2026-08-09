param(
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Destination) {
    $Destination = (Join-Path (Split-Path (Split-Path $ProjectRoot -Parent) -Parent) "work\cadabra2-root")
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
if (-not $Destination.StartsWith((Split-Path (Split-Path $ProjectRoot -Parent) -Parent), [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination must remain inside the Codex task workspace."
}
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$WslDestination = (wsl -d Ubuntu-24.04 -- wslpath -a ($Destination -replace '\\','/')).Trim()
if (-not $WslDestination) {
    throw "Could not translate the destination into a WSL path."
}

$Command = @"
set -eu
cd '$WslDestination'
apt download cadabra2 fonts-cmu libgmpxx4ldbl
mkdir -p root
dpkg-deb -x cadabra2_*_amd64.deb root
dpkg-deb -x fonts-cmu_*_all.deb root
dpkg-deb -x libgmpxx4ldbl_*_amd64.deb root
test -f root/usr/bin/cadabra2
test -f root/usr/lib/python3/dist-packages/cadabra2.cpython-312-x86_64-linux-gnu.so
"@
wsl -d Ubuntu-24.04 -- bash -lc $Command
if ($LASTEXITCODE -ne 0) {
    throw "Cadabra 2 local extraction failed."
}

Write-Output "cadabra_root=$Destination\root"
Write-Output "Set SIGMA_CADABRA_ROOT to that root only if auto-detection cannot find it."
