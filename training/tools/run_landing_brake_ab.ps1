param(
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Baseline,
    [int]$Iterations=60,
    [int]$Envs=768,
    [int]$LivePort=6022,
    [switch]$ActorSpeed
)

$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUNBUFFERED='1'
$source=Join-Path $RunDirectory 'source'
$status=Join-Path $RunDirectory 'status.txt'
$control='configs/v02_landing_brake_control.toml'
$variant=if ($ActorSpeed) { 'true_speed' } else { 'persistent' }
$treatment="configs/v02_landing_brake_$variant.toml"

function Phase([string]$name, [string[]]$cli, [int[]]$allowed=@(0)) {
    [IO.File]::WriteAllText($status, $name)
    $argsForPython=($cli | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $p=Start-Process -FilePath $Python -WorkingDirectory $source -ArgumentList $argsForPython `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $RunDirectory ($name+'.log')) `
        -RedirectStandardError (Join-Path $RunDirectory ($name+'.err.log')) -Wait -PassThru
    if ($p.ExitCode -notin $allowed) { throw "$name exited $($p.ExitCode)" }
}

function Assert-Rows([string]$name, [int]$count) {
    $file=Join-Path $RunDirectory ($name+'.csv')
    if (@(Import-Csv -LiteralPath $file).Count -ne $count) { throw "Unexpected row count: $name" }
}

if (Test-Path -LiteralPath $status) { throw 'The run already has a status file. Use a fresh directory.' }
try {
    Set-Location -LiteralPath $source
    $field=if ($ActorSpeed) { 'actor_true_forward_speed' } else { 'preserve_across_airborne' }
    $section=if ($ActorSpeed) { "['task']['locomotion']" } else { "['task']['scut_rough_reward']['braking']" }
    & $Python -c "from configs.experiment import load_config; from pathlib import Path; a=load_config(Path('$control')); b=load_config(Path('$treatment')); assert b$section.pop('$field'); assert not a$section.pop('$field'); assert a==b"
    if ($LASTEXITCODE -ne 0) { throw "A/B configs must differ only in $field" }
    $landing=@('-u','-m','tools.landing_brake_acceptance','--config',$control)
    if (-not $ActorSpeed) {
        Phase 'baseline_trace' ($landing + @('--output',(Join-Path $RunDirectory 'baseline.csv'),
            '--trace',(Join-Path $RunDirectory 'baseline_trace.csv'),$Baseline))
        Assert-Rows 'baseline' 64
    }
    $models=@($Baseline)
    foreach ($arm in 'control',$variant) {
        $dest=Join-Path $RunDirectory $arm
        Phase "train_$arm" @('-u','main.py','train','--config',"configs/v02_landing_brake_$arm.toml",
            '--init-checkpoint',$Baseline,'--num-envs',$Envs,'--iterations',$Iterations,'--seed','42',
            '--output',$dest,'--live-port',$LivePort,'--live-stream','--live-fps','5')
        $checkpoint=Join-Path $dest "model_$($Iterations-1).pt"
        if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "Missing $arm checkpoint" }
        $models+=$checkpoint
    }
    if ($ActorSpeed) { $landing+= '--mask-actor-speed' }
    Phase 'landing_eval' ($landing + @('--output',(Join-Path $RunDirectory 'landing_eval.csv'),
        '--trace',(Join-Path $RunDirectory 'landing_trace.csv')) + $models)
    if ($ActorSpeed) {
        Assert-Rows 'landing_eval' 256
        Phase 'landing_eval_seed_827' ($landing + @('--seed','827','--output',
            (Join-Path $RunDirectory 'landing_eval_seed_827.csv')) + $models)
        Assert-Rows 'landing_eval_seed_827' 256
    } else {
        Assert-Rows 'landing_eval' 192
    }
    Phase 'flat_eval' (@('-u','-m','tools.flat_height_acceptance','--config','configs/v02_sim2real_flat_height.toml',
        '--output',(Join-Path $RunDirectory 'flat_eval.csv')) + $models)
    Assert-Rows 'flat_eval' 27
    Phase 'obstacle_eval' (@('-u','-m','tools.terrain_obstacle_acceptance','--output',
        (Join-Path $RunDirectory 'obstacle_eval.csv')) + $models) @(0,1)
    Assert-Rows 'obstacle_eval' 18
    [IO.File]::WriteAllText($status,'complete')
} catch {
    [IO.File]::WriteAllText($status,('failed: '+$_.Exception.Message))
    throw
}
