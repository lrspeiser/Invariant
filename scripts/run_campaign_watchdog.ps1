param(
    [string]$Database = "runs\campaigns\campaign-v1-live.sqlite",
    [string]$CampaignId = "",
    [string]$Slice = "6h",
    [string]$StopFile = "runs\campaigns\STOP",
    [string]$Python = "python"
)

$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
$worker = "$env:COMPUTERNAME-watchdog"

while (-not (Test-Path -LiteralPath $StopFile)) {
    $arguments = @(
        "-m", "sigma_theory_compiler.campaign_cli", "run",
        "--database", $Database,
        "--worker-id", $worker,
        "--duration", $Slice,
        "--follow"
    )
    if ($CampaignId) {
        $arguments += @("--campaign-id", $CampaignId)
    }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Campaign worker exited with code $LASTEXITCODE; restarting in 10 seconds."
        Start-Sleep -Seconds 10
    }
    else {
        Start-Sleep -Seconds 2
    }
}

Write-Host "Campaign watchdog stopped because $StopFile exists."
