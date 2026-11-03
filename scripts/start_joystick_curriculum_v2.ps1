param(
    [ValidateSet(32, 64, 128, 256)][int]$NumEnvs = 64,
    [ValidateRange(3, 2000000000)][long]$Timesteps = 30000000,
    [ValidateRange(0.16, 2.0)][double]$SpeedCenter = 1.1,
    [switch]$PrintCommand
)

$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$campaign = "joystick_v2_$stamp"
$logBase = Join-Path $projectDir "artifacts/$campaign"
$speedText = $SpeedCenter.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$arguments = @('-d', 'Ubuntu', '--exec', 'env',
    "JOYSTICK_NUM_ENVS=$NumEnvs", "JOYSTICK_TIMESTEPS=$Timesteps",
    "JOYSTICK_SPEED_CENTER=$speedText", "JOYSTICK_CAMPAIGN=$campaign",
    'nice', '-n', '5', 'bash', 'scripts/train_joystick_curriculum_v2.sh')
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
    WindowsPid=$process.Id; Campaign=$campaign; Timesteps=$Timesteps; NumEnvs=$NumEnvs
    SpeedCenter=$SpeedCenter; Started=(Get-Date).ToString('o')
    Stdout="$logBase.stdout.log"; Stderr="$logBase.stderr.log"
} | ConvertTo-Json | Set-Content -LiteralPath "$logBase.launch.json" -Encoding UTF8
Write-Output "Started launcher PID $($process.Id), no timeout."
Write-Output "Log: $logBase.stdout.log"
Write-Output "Curriculum state: $logBase/status.json"
