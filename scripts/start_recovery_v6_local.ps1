param([switch]$Probe)

# Run from Windows without keeping a terminal open. No system settings change.
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$artifactDir = Join-Path $projectDir 'artifacts'
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$mode = if ($Probe) { 'probe' } else { '80m' }
$logBase = Join-Path $artifactDir "recovery_local_${mode}_${stamp}"
$runTag = "recovery_v6_local_${mode}_${stamp}"
$stepCount = if ($Probe) { 102400 } else { 80000000 }
$evalCount = if ($Probe) { 3 } else { 41 }
$metricSteps = if ($Probe) { 5120 } else { 51200 }

$wslArguments = @(
    '-d', 'Ubuntu', '--exec', 'env',
    "RECOVERY_TIMESTEPS=$stepCount",
    "RECOVERY_NUM_EVALS=$evalCount",
    "RECOVERY_METRICS_STEPS=$metricSteps",
    "RECOVERY_RUN_TAG=$runTag"
)
if ($Probe) {
    # A bounded performance measurement, not a meaningful gait experiment.
    $wslArguments += @('timeout', '--signal=INT', '--kill-after=30s', '1800s')
}
$wslArguments += @('nice', '-n', '5', 'bash', 'scripts/train_recovery_v6_tracker.sh')
$runProcess = Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArguments `
    -WorkingDirectory $projectDir -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$logBase.stdout.log" `
    -RedirectStandardError "$logBase.stderr.log"

[pscustomobject]@{
    WindowsPid = $runProcess.Id
    Mode = $mode
    RunTag = $runTag
    Started = (Get-Date).ToString('o')
    Stdout = "$logBase.stdout.log"
    Stderr = "$logBase.stderr.log"
} | ConvertTo-Json | Set-Content -LiteralPath "$logBase.json" -Encoding UTF8

Write-Output "Started WSL launcher PID $($runProcess.Id); mode=$mode (GPU training is not yet confirmed)."
Write-Output "Training log: $logBase.stdout.log"
Write-Output "Errors/compiler warnings: $logBase.stderr.log"
Write-Output 'Only one recovery launcher can run at a time; inspect the log for startup errors.'
Write-Output 'Sleep, shutdown, or wsl --shutdown will interrupt training.'
