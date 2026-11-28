$ErrorActionPreference = 'Stop'
$trainDir = 'D:\RL_Stuff\wheel_leg_rl_parallel\training'
Set-Location $trainDir

$runName = 'v02_active_susp_hjc_20261125'
$outDir = Join-Path $trainDir "outputs\$runName"
$logDir = Join-Path $trainDir "outputs\${runName}_logs"

New-Item -ItemType Directory -Force $logDir | Out-Null
if (Test-Path $outDir) {
    Remove-Item -Recurse -Force $outDir
}

$workerCmd = Join-Path $logDir 'worker.cmd'
$trainLog = Join-Path $logDir 'train.log'
$errLog = Join-Path $logDir 'train.err.log'

$cmdBody = @"
@echo off
cd /d $trainDir
set QD_OFFLINE_CACHE_FILE_PATH=$trainDir\.cache\quadrants
"$trainDir\.venv\Scripts\python.exe" -u main.py train --robot v0.2 --num-envs 1536 --iterations 500 --config configs/v02_obstacle_step_lift_finetune.toml --init-checkpoint outputs/v02_obstacle_lift_hjc_20261125/model_450.pt --output outputs/$runName --live-port 8765 --live-stream --live-fps 8 > "$trainLog" 2> "$errLog"
"@
Set-Content -Path $workerCmd -Value $cmdBody -Encoding ASCII

$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd.exe /c `"$workerCmd`""
    CurrentDirectory = $trainDir
}
if ($result.ReturnValue -ne 0) {
    throw "Failed to launch training process: ReturnValue=$($result.ReturnValue)"
}
Write-Host "Launched detached cmd PID=$($result.ProcessId) for $runName"

# Wait up to 90 seconds for live monitor port 8765 to come up.
$ready = $false
for ($i = 0; $i -lt 45; $i++) {
    Start-Sleep -Seconds 2
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8765/metrics.json' -TimeoutSec 2
        if ($resp.StatusCode -eq 200) {
            Write-Host "Live monitor READY (HTTP 200) after $(($i+1)*2)s"
            $ready = $true
            break
        }
    } catch {}
}
if (-not $ready) {
    Write-Host "Live monitor not up yet; tailing logs:"
    if (Test-Path $trainLog) { Get-Content $trainLog -Tail 20 }
    if (Test-Path $errLog) { Get-Content $errLog -Tail 20 }
}
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*$runName*" } | Select-Object ProcessId, CommandLine
