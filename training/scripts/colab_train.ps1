<#
.SYNOPSIS
    Windows PowerShell wrapper for training/scripts/colab_train.sh (WSL google-colab-cli).
    可通过环境变量覆盖默认 WSL 配置: $env:COLAB_WSL_DISTRO (默认 Ubuntu-24.04), $env:COLAB_WSL_USER (默认 sunwater).

.EXAMPLE
    # 一键申请 GPU、同步代码、安装环境、启动训练并转发监控 (8765 + 6006)
    powershell -ExecutionPolicy Bypass -File .\training\scripts\colab_train.ps1 up --init-ckpt training/outputs/v02_scut_stage2_scut_aligned/model_1600.pt

    # 单独开启/重连本机浏览器监控转发 (http://127.0.0.1:8765 & http://127.0.0.1:6006)
    powershell -ExecutionPolicy Bypass -File .\training\scripts\colab_train.ps1 tunnel

    # 查看远端训练进度、GPU 占用与已保存的 checkpoints
    powershell -ExecutionPolicy Bypass -File .\training\scripts\colab_train.ps1 status

    # 拉回最新或指定 checkpoint 及 run_config.json 到本地 training/outputs/
    powershell -ExecutionPolicy Bypass -File .\training\scripts\colab_train.ps1 pull --run-name v02_sim2real_stage2_robust

    # 查看 Colab 算力余额 / 停止实例释放计费
    powershell -ExecutionPolicy Bypass -File .\training\scripts\colab_train.ps1 usage
    powershell -ExecutionPolicy Bypass -File .\training\scripts\colab_train.ps1 stop
#>

$ErrorActionPreference = 'Stop'

$distro = if ($env:COLAB_WSL_DISTRO) { $env:COLAB_WSL_DISTRO } else { 'Ubuntu-24.04' }
$wslUser = if ($env:COLAB_WSL_USER) { $env:COLAB_WSL_USER } else { 'sunwater' }

$shWinPath = Join-Path $PSScriptRoot 'colab_train.sh'
if (-not (Test-Path $shWinPath)) {
    throw "Missing script: $shWinPath"
}

$resolved = (Resolve-Path $shWinPath).Path
$drive = $resolved.Substring(0, 1).ToLowerInvariant()
$rest = $resolved.Substring(2).Replace('\', '/')
$shWslPath = "/mnt/$drive$rest"

# 自动把参数中的 Windows 盘符路径（如 E:\...）转为 WSL 路径（/mnt/e/...）
$convertedArgs = @()
foreach ($arg in $args) {
    $s = [string]$arg
    if ($s -match '^([A-Za-z]):[\\/](.*)$') {
        $d = $Matches[1].ToLowerInvariant()
        $r = $Matches[2].Replace('\', '/')
        $convertedArgs += "/mnt/$d/$r"
    } else {
        $convertedArgs += $s
    }
}

& wsl -d $distro -u $wslUser -- bash $shWslPath @convertedArgs
exit $LASTEXITCODE
