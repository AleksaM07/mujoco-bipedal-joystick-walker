param(
    [ValidateSet(32, 64, 128, 256)][int]$NumEnvs = 64,
    [ValidateRange(1, 2000000000)][long]$Timesteps = 10000000,
    [ValidateRange(0.01, 4.0)][double]$ArmRewardWeight = 1.0,
    [switch]$PrintCommand
)

$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$runTag = "arms_ik_v2_$stamp"
$logBase = Join-Path $projectDir "artifacts/$runTag"
$weight = $ArmRewardWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$arguments = @('-d', 'Ubuntu', '--exec', 'env',
    "ARM_IK_NUM_ENVS=$NumEnvs", "ARM_IK_TIMESTEPS=$Timesteps",
    "ARM_IK_REWARD_WEIGHT=$weight", "ARM_IK_RUN_TAG=$runTag",
    'nice', '-n', '5', 'bash', 'scripts/train_arm_ik_transfer.sh')
if ($PrintCommand) {
    Write-Output ('wsl.exe ' + ($arguments -join ' '))
    return
}
New-Item -ItemType Directory -Force -Path (Join-Path $projectDir 'artifacts') | Out-Null
$process = Start-Process -FilePath 'wsl.exe' -ArgumentList $arguments `
    -WorkingDirectory $projectDir -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$logBase.stdout.log" `
    -RedirectStandardError "$logBase.stderr.log"
[pscustomobject]@{
    WindowsPid=$process.Id; RunTag=$runTag; Timesteps=$Timesteps; NumEnvs=$NumEnvs
    ArmRewardWeight=$ArmRewardWeight; Started=(Get-Date).ToString('o')
    Stdout="$logBase.stdout.log"; Stderr="$logBase.stderr.log"
} | ConvertTo-Json | Set-Content -LiteralPath "$logBase.launch.json" -Encoding UTF8
Write-Output "Started arm IK transfer PID $($process.Id), no timeout."
Write-Output "Log: $logBase.stdout.log"
