import base64
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile

PACKAGE = Path(__file__).parent
PROJECT = PACKAGE.parent
TRAINING = PROJECT / "training"
POWERSHELL = ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive"]
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
               "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
ARTIFACT = re.compile(r"model_\d+\.pt|run_config\.json|events\.out\.tfevents\.[\w.-]+|replay\.mp4|std(out|err)\.log")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def command(args, timeout=15):
    result = subprocess.run(args, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr.strip() or result.stdout.strip() or f"退出码 {result.returncode}")[-1600:])
    return result.stdout.strip()


def snapshot(directory):
    # Git supplies the allowed working-tree paths, including new configurations.
    names = subprocess.check_output(
        ["git", "-C", str(PROJECT), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "training"]
    ).decode("utf-8").split("\0")
    revision = command(["git", "-C", str(PROJECT), "rev-parse", "HEAD"])
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=directory, suffix=".zip", delete=False) as file:
        temp = Path(file.name)
    try:
        lock_hash = None
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name in sorted(set(names) - {""}):
                path = PROJECT / name
                if not path.is_file() or not path.resolve().is_relative_to(TRAINING.resolve()):
                    continue
                relative = path.relative_to(TRAINING).as_posix()
                data = path.read_bytes()
                info = zipfile.ZipInfo(relative)
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data, compresslevel=1)
                if relative == "uv.lock":
                    lock_hash = hashlib.sha256(data).hexdigest()
        if lock_hash is None:
            raise ValueError("训练目录缺少 uv.lock")
        digest = file_hash(temp)
        final = directory / f"{digest}.zip"
        if not final.exists():
            temp.replace(final)
        return {"sha256": digest, "lock": lock_hash, "git": revision}
    finally:
        temp.unlink(missing_ok=True)


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def encoded(script):
    header = "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; "
    header += "[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); "
    header += """
        function Get-Sha256([string]$Path) {
            $hash = [Security.Cryptography.SHA256]::Create()
            $stream = [IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
            try {
                [BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
            } finally {
                $stream.Dispose()
                $hash.Dispose()
            }
        }
    """
    return base64.b64encode((header + script).encode("utf-16-le")).decode("ascii")


class Node:
    def __init__(self, root, host=None):
        self.root = root
        self.host = host
        self.info = {"online": False, "busy": False}
        self.processes = {}

    def ps(self, script, timeout=15):
        shell = [*POWERSHELL, "-EncodedCommand", encoded(script)]
        if self.host:
            shell = ["ssh", *SSH_OPTIONS, self.host, *shell]
        return command(shell, timeout)

    def probe(self):
        try:
            self.info = json.loads(self.ps(r"""
                $active = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='uv.exe'" |
                    Where-Object { $_.CommandLine -match '(?i)main\.py"?\s+"?(train|demo)\b' -and
                        $_.CommandLine -notmatch '(?i)--backend"?\s+"?cpu\b' })
                $gpu = & nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
                $hasGpu = $LASTEXITCODE -eq 0
                @{online=$true; busy=($active.Count -gt 0); gpu=(@($gpu)[0]); cuda=$hasGpu;
                  uv=(Get-Command uv.exe).Source; computer=$env:COMPUTERNAME} | ConvertTo-Json -Compress
            """))
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
            self.info = {"online": False, "busy": False, "error": str(error)}
        return self.info

    def upload(self, source, target, digest=None):
        if not self.host:
            shutil.copy2(source, target)
            return
        digest = digest or file_hash(source)
        final = literal(target)
        existing = self.ps(
            f"if ((Test-Path -LiteralPath {final}) -and "
            f"((Get-Sha256 {final}) -eq '{digest}')) {{ 'ready' }}"
        )
        if existing == "ready":
            return
        part = target.with_name(target.name + ".partial")
        command(["scp", "-q", *SSH_OPTIONS, str(source), f"{self.host}:{part.as_posix()}"], timeout=300)
        self.ps(
            f"if ((Get-Sha256 {literal(part)}) -ne '{digest}') {{ throw 'Upload hash mismatch' }}; "
            f"Move-Item -LiteralPath {literal(part)} -Destination {final} -Force"
        )

    def prepare(self, job, directory, snapshots):
        run = self.root / "executions" / job["id"]
        source = job["source"]
        archive = snapshots / f"{source['sha256']}.zip"
        remote_archive = self.root / "snapshots" / archive.name if self.host else archive
        environment = self.root / "envs" / source["lock"]
        no_sync = False
        if self.host:
            self.ps(f"@({literal(run)}, {literal(remote_archive.parent)}, {literal(environment.parent)}) | "
                    "ForEach-Object { [void][IO.Directory]::CreateDirectory($_) }")
            self.upload(archive, remote_archive, source["sha256"])
        else:
            run.mkdir(parents=True, exist_ok=True)
            existing = TRAINING / ".venv"
            if (existing / "Scripts" / "python.exe").is_file() and file_hash(TRAINING / "uv.lock") == source["lock"]:
                try:
                    command([self.info["uv"], "sync", "--project", str(TRAINING), "--locked", "--check", "--offline"])
                except (OSError, RuntimeError, subprocess.SubprocessError):
                    pass
                else:
                    environment = existing
                    no_sync = True
        self.upload(PACKAGE / "run_job.ps1", run / "run_job.ps1")
        inputs = directory / "input"
        if inputs.is_dir():
            target = run / "input"
            if self.host:
                self.ps(f"[void][IO.Directory]::CreateDirectory({literal(target)})")
            else:
                target.mkdir(exist_ok=True)
            for path in inputs.iterdir():
                if path.is_file():
                    self.upload(path, target / path.name)
        manifest = directory / "job.json"
        write_json(manifest, {
            "snapshot": str(remote_archive), "environment": str(environment), "uv": self.info["uv"], "no_sync": no_sync,
            "arguments": job["arguments"],
        })
        self.upload(manifest, run / "job.json")

    def start(self, job_id):
        run = self.root / "executions" / job_id
        script = run / "run_job.ps1"
        # Bypass policy only for the shipped, per-run script. Remote work is a one-shot process.
        args = [*POWERSHELL, "-ExecutionPolicy", "Bypass", "-File", str(script), str(run)]
        if not self.host:
            with (run / "launcher.log").open("ab") as log:
                self.processes[job_id] = subprocess.Popen(
                    args, cwd=run, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            return
        line = literal(subprocess.list2cmdline(args))
        result = json.loads(self.ps(
            "Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
            f"-Arguments @{{CommandLine={line}; CurrentDirectory={literal(run)}}} | "
            "Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress"
        ))
        if result["ReturnValue"]:
            raise RuntimeError(f"后台启动失败，WMI 返回 {result['ReturnValue']}")

    def inspect(self, job_id):
        run = self.root / "executions" / job_id
        script = r"""
            $status = Join-Path $run 'status.json'
            $data = if (Test-Path -LiteralPath $status) {
                Get-Content -LiteralPath $status -Raw -Encoding UTF8 | ConvertFrom-Json
            }
            $alive = $false
            if (!$data -or $data.phase -ne 'finished') {
                $alive = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='python.exe' OR Name='uv.exe'" |
                    Where-Object { $_.CommandLine -and $_.CommandLine.Contains($run) }).Count -gt 0
            }
            $out = @(Get-ChildItem -LiteralPath (Join-Path $run 'artifacts') -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^(model_\d+\.pt|run_config\.json|events\.out\.tfevents\.[\w.-]+|replay\.mp4)$' -and
                    ($_.Name -ne 'replay.mp4' -or $data.phase -eq 'finished') } |
                ForEach-Object {
                    try {
                        @{name=$_.Name; sha256=(Get-Sha256 $_.FullName)}
                    } catch {}
                })
            $tail = foreach ($name in ('stdout.log','stderr.log','launcher.log')) {
                $where = Join-Path $run $name
                if (Test-Path -LiteralPath $where) {
                    if ($data.phase -eq 'finished' -and $name -ne 'launcher.log') {
                        $out += @{name=$name; sha256=(Get-Sha256 $where)}
                    }
                    Get-Content -LiteralPath $where -Tail 70 -Encoding UTF8
                }
            }
            @{status=$data; alive=$alive; artifacts=$out; log=($tail -join [Environment]::NewLine)} |
                ConvertTo-Json -Compress -Depth 5
        """
        report = json.loads(self.ps(f"$run = {literal(run)}; " + script))
        proc = self.processes.get(job_id)
        if proc and proc.poll() is not None:
            del self.processes[job_id]
            if not report.get("status") and proc.returncode:
                report["status"] = {"phase": "finished", "exit_code": proc.returncode}
        return report

    def collect(self, job_id, entry, directory):
        name, digest = entry["name"], entry["sha256"]
        if not ARTIFACT.fullmatch(name) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("不支持的产物名称或校验值")
        source = self.root / "executions" / job_id
        source = source / name if name.endswith(".log") else source / "artifacts" / name
        dest = directory / name
        part = dest.with_name(name + ".partial")
        try:
            if self.host:
                command(["scp", "-q", *SSH_OPTIONS, f"{self.host}:{source.as_posix()}", str(part)], timeout=120)
            else:
                shutil.copy2(source, part)
            if file_hash(part) != digest:
                raise ValueError(f"{name} 在传输期间变化")
            if name.endswith(".pt"):
                with zipfile.ZipFile(part) as archive:
                    if archive.testzip() is not None:
                        raise ValueError(f"{name} 尚未写完")
            elif name == "run_config.json":
                json.loads(part.read_text(encoding="utf-8"))
            part.replace(dest)
        finally:
            part.unlink(missing_ok=True)


def nodes(state, host, remote_root):
    result = {"local": Node(state)}
    if host:
        path = PureWindowsPath(remote_root)
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", host) or not path.is_absolute():
            raise ValueError("请使用 SSH 主机别名和 Windows 绝对工作路径")
        result["remote"] = Node(path, host)
    return result
