param(
    [ValidateSet(32, 64, 128, 256)][int]$NumEnvs = 64,
    [ValidateRange(1, 2000000000)][long]$Timesteps = 30000000,
    [switch]$PrintCommand
)

# Start the WSL training independently from this terminal. No timeout is used.
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$artifactDir = Join-Path $projectDir 'artifacts'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$logBase = Join-Path $artifactDir "arms_joystick_v1_${stamp}"
$runTag = "arms_joystick_v1_30m_${stamp}"
$evalCount = 31
$metricSteps = 50000

$wslArguments = @(
    '-d', 'Ubuntu', '--exec', 'env',
    "ARMS_JOYSTICK_TIMESTEPS=$Timesteps",
    "ARMS_JOYSTICK_NUM_ENVS=$NumEnvs",
    "ARMS_JOYSTICK_NUM_EVALS=$evalCount",
    "ARMS_JOYSTICK_METRICS_STEPS=$metricSteps",
    "ARMS_JOYSTICK_RUN_TAG=$runTag",
    'nice', '-n', '5', 'bash', 'scripts/train_arms_joystick_v1.sh'
)

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
    Mode = 'arms_joystick_v1'
    NumEnvs = $NumEnvs
    Timesteps = $Timesteps
    WarmStart = 'prezentacija/1st_success/final_model_000025681920 -> 26 DOF migration'
    RunTag = $runTag
    Started = (Get-Date).ToString('o')
    Stdout = "$logBase.stdout.log"
    Stderr = "$logBase.stderr.log"
} | ConvertTo-Json | Set-Content -LiteralPath "$logBase.json" -Encoding UTF8

Write-Output "Started WSL launcher PID $($runProcess.Id)."
Write-Output "Training log: $logBase.stdout.log"
Write-Output "Errors/compiler warnings: $logBase.stderr.log"
Write-Output 'This run has no timeout. Sleep, shutdown, or wsl --shutdown will interrupt it.'
