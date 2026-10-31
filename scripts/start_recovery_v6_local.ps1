param(
    [switch]$Probe,
    [ValidateSet(64, 128, 256, 512)][int]$NumEnvs = 64,
    [ValidateRange(1, 2000000000)][long]$Timesteps = 80000000,
    [string]$ResumeFrom = '',
    [switch]$PrintCommand
)

# Run from Windows without keeping a terminal open. No system settings change.
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$artifactDir = Join-Path $projectDir 'artifacts'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$mode = if ($Probe) { 'probe' } else { '80m' }
if (-not $Probe -and $Timesteps -ne 80000000) { $mode = 'custom' }
if ($ResumeFrom) { $mode += '_resume' }
$logBase = Join-Path $artifactDir "recovery_local_${mode}_${stamp}"
$runTag = "recovery_v6_local_${mode}_${stamp}"
$stepCount = if ($Probe) { 102400 } else { $Timesteps }
$evalCount = if ($Probe) { 3 } else { 41 }
$metricSteps = if ($Probe) { 5120 } else { 51200 }

$wslArguments = @(
    '-d', 'Ubuntu', '--exec', 'env',
    "RECOVERY_TIMESTEPS=$stepCount",
    "RECOVERY_NUM_EVALS=$evalCount",
    "RECOVERY_METRICS_STEPS=$metricSteps",
    "RECOVERY_NUM_ENVS=$NumEnvs",
    "RECOVERY_RUN_TAG=$runTag"
)
if ($ResumeFrom) {
    $resumePath = (Resolve-Path -LiteralPath $ResumeFrom).Path
    $projectPrefix = $projectDir + [IO.Path]::DirectorySeparatorChar
    if (-not $resumePath.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Resume checkpoint must be inside this project for portable WSL paths.'
    }
    if (-not (Test-Path -LiteralPath $resumePath -PathType Container)) {
        throw 'Resume checkpoint must be a directory.'
    }
    $resumeRelative = $resumePath.Substring($projectPrefix.Length).Replace('\', '/')
    # Start-Process joins ArgumentList into a Windows command line. Quote this
    # whole argument so checkpoint paths with spaces remain a single env value.
    $wslArguments += '"RECOVERY_RESUME_FROM=' + $resumeRelative + '"'
}
if ($Probe) {
    # A bounded performance measurement, not a meaningful gait experiment.
    $wslArguments += @('timeout', '--signal=INT', '--kill-after=30s', '1800s')
}
$wslArguments += @('nice', '-n', '5', 'bash', 'scripts/train_recovery_v6_tracker.sh')
if ($PrintCommand) {
    Write-Output ('wsl.exe ' + ($wslArguments -join ' '))
    return
}
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$runProcess = Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArguments `
    -WorkingDirectory $projectDir -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$logBase.stdout.log" `
    -RedirectStandardError "$logBase.stderr.log"

[pscustomobject]@{
    WindowsPid = $runProcess.Id
    Mode = $mode
    NumEnvs = $NumEnvs
    Timesteps = $stepCount
    ResumeFrom = $ResumeFrom
    RunTag = $runTag
    Started = (Get-Date).ToString('o')
    Stdout = "$logBase.stdout.log"
    Stderr = "$logBase.stderr.log"
} | ConvertTo-Json | Set-Content -LiteralPath "$logBase.json" -Encoding UTF8

Write-Output "Started WSL launcher PID $($runProcess.Id); mode=$mode (GPU training is not yet confirmed)."
Write-Output "Training log: $logBase.stdout.log"
Write-Output "Errors/compiler warnings: $logBase.stderr.log"
if ($ResumeFrom) {
    Write-Output "Restoring weights/normalizer; optimizer and step count restart. Requested new steps: $stepCount."
}
Write-Output 'Only one recovery launcher can run at a time; inspect the log for startup errors.'
Write-Output 'Sleep, shutdown, or wsl --shutdown will interrupt training.'
