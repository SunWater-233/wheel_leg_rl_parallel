param([Parameter(Mandatory = $true)][string]$RunDirectory)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$state = Join-Path $RunDirectory 'status.json'
$stdout = Join-Path $RunDirectory 'stdout.log'
$stderr = Join-Path $RunDirectory 'stderr.log'

function Write-Status($phase, $code) {
    $data = @{phase = $phase; exit_code = $code; pid = $PID}
    $temp = Join-Path $RunDirectory 'status.tmp'
    [IO.File]::WriteAllText($temp, ($data | ConvertTo-Json -Compress), $utf8)
    if ([IO.File]::Exists($state)) {
        [IO.File]::Replace($temp, $state, [NullString]::Value)
    } else {
        [IO.File]::Move($temp, $state)
    }
}

# A repeated SSH launch can race with the first launch's acknowledgement.
$claim = Join-Path $RunDirectory 'launch.claim'
try {
    $lock = [IO.File]::Open($claim, 'CreateNew', 'Write', 'None')
    $lock.Dispose()
} catch [IO.IOException] {
    exit 2
}

try {
    Write-Status 'preparing' $null
    $job = Get-Content -LiteralPath (Join-Path $RunDirectory 'job.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $source = Join-Path $RunDirectory 'source'
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($job.snapshot, $source)
    if ($job.environment) { $env:UV_PROJECT_ENVIRONMENT = $job.environment }
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $argsForUv = @('run', '--locked')
    if ($job.no_sync) { $argsForUv += '--no-sync' }
    $argsForUv += @((Join-Path $source 'main.py')) + @($job.arguments)
    # All argument values are constructed by the manager. Quote paths for the Win32 command line.
    $commandLine = ($argsForUv | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $job.uv
    $start.Arguments = $commandLine
    $start.WorkingDirectory = $source
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $outFile = [IO.FileStream]::new($stdout, 'Create', 'Write', 'Read', 1)
    $errFile = [IO.FileStream]::new($stderr, 'Create', 'Write', 'Read', 1)
    try {
        $proc = [Diagnostics.Process]::Start($start)
        $outCopy = $proc.StandardOutput.BaseStream.CopyToAsync($outFile)
        $errCopy = $proc.StandardError.BaseStream.CopyToAsync($errFile)
        Write-Status 'running' $null
        $proc.WaitForExit()
        $outCopy.GetAwaiter().GetResult()
        $errCopy.GetAwaiter().GetResult()
        $code = $proc.ExitCode
    } finally {
        $outFile.Dispose()
        $errFile.Dispose()
        if ($proc) { $proc.Dispose() }
    }
    Write-Status 'finished' $code
} catch {
    [IO.File]::AppendAllText($stderr, ($_ | Out-String), $utf8)
    Write-Status 'finished' 1
    exit 1
}
