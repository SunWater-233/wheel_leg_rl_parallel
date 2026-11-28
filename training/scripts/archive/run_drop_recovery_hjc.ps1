$ErrorActionPreference = 'Stop'
$root = 'D:\RL_Stuff\wheel_leg_rl_parallel\training'
Set-Location $root
$env:QD_OFFLINE_CACHE_FILE_PATH = Join-Path $root '.cache\quadrants'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

$outDir = Join-Path $root 'outputs\v02_drop_recovery_hjc_20261125'
$logDir = Join-Path $root 'outputs\v02_drop_recovery_hjc_20261125_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (Test-Path $outDir) {
    Remove-Item -Recurse -Force $outDir
}

$worker = Join-Path $logDir 'worker.cmd'
$cmd = @(
    '@echo off'
    'cd /d D:\RL_Stuff\wheel_leg_rl_parallel\training'
    'set QD_OFFLINE_CACHE_FILE_PATH=D:\RL_Stuff\wheel_leg_rl_parallel\training\.cache\quadrants'
    'set PYTHONUNBUFFERED=1'
    'set PYTHONIOENCODING=utf-8'
    'D:\RL_Stuff\wheel_leg_rl_parallel\training\.venv\Scripts\python.exe -u main.py train --robot v0.2 --num-envs 1536 --config configs/v02_drop_recovery_finetune.toml --init-checkpoint outputs/v02_airborne_fix_20261125/model_299.pt --iterations 400 --live-port 8765 --live-stream --live-fps 8 --output outputs/v02_drop_recovery_hjc_20261125 > "D:\RL_Stuff\wheel_leg_rl_parallel\training\outputs\v02_drop_recovery_hjc_20261125_logs\train.log" 2> "D:\RL_Stuff\wheel_leg_rl_parallel\training\outputs\v02_drop_recovery_hjc_20261125_logs\train.err.log"'
) -join "`r`n"
[IO.File]::WriteAllText($worker, $cmd, [Text.Encoding]::ASCII)

$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd.exe /c `"$worker`""
    CurrentDirectory = $root
}
Write-Output ("LAUNCH_RETURN=" + $res.ReturnValue + " CMD_PID=" + $res.ProcessId)
