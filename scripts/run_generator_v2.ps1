param(
    [long]$Limit = 0,
    [int]$Threads = 8,
    [long]$BlockSize = 65536,
    [string]$Output = "runs\generator-v2\reproduction.json",
    [string]$CheckpointDirectory = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$binary = Join-Path $projectRoot "generator-v2\target\release\sigma-generator-v2.exe"
$config = Join-Path $projectRoot "configs\generator_v2_billion.json"
$outputPath = Join-Path $projectRoot $Output

if (-not (Test-Path -LiteralPath $binary)) {
    throw "Generator v2 is not built. Run scripts\bootstrap_generator_v2.ps1 first."
}

$arguments = @(
    "run",
    "--config", $config,
    "--output", $outputPath,
    "--threads", $Threads,
    "--block-size", $BlockSize
)
if ($Limit -gt 0) {
    $arguments += @("--limit", $Limit)
}
if ($CheckpointDirectory) {
    $arguments += @("--checkpoint-dir", (Join-Path $projectRoot $CheckpointDirectory))
}

& $binary @arguments

