param(
    [switch]$ReleaseBuild = $true
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$workspaceRoot = (Resolve-Path (Join-Path $projectRoot "..\..")).Path
$toolRoot = Join-Path $workspaceRoot "work\rust-local"
$cargoHome = Join-Path $toolRoot "cargo"
$rustupHome = Join-Path $toolRoot "rustup"
$rustupInit = Join-Path $toolRoot "rustup-init.exe"
$cargo = Join-Path $cargoHome "bin\cargo.exe"

New-Item -ItemType Directory -Force -Path $toolRoot | Out-Null
if (-not (Test-Path -LiteralPath $cargo)) {
    if (-not (Test-Path -LiteralPath $rustupInit)) {
        Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustupInit
    }
    $env:CARGO_HOME = $cargoHome
    $env:RUSTUP_HOME = $rustupHome
    & $rustupInit -y --no-modify-path --profile minimal --default-host x86_64-pc-windows-gnu
}

$env:CARGO_HOME = $cargoHome
$env:RUSTUP_HOME = $rustupHome
$env:PATH = "$(Join-Path $cargoHome 'bin');$env:PATH"
$manifest = Join-Path $projectRoot "generator-v2\Cargo.toml"
& $cargo test --manifest-path $manifest
if ($ReleaseBuild) {
    & $cargo build --release --manifest-path $manifest
}

