import json
import math
import re
import shutil
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from .execution import ARTIFACT, TRAINING, file_hash, snapshot

ACTIVE = ("staging", "starting", "running", "disconnected", "syncing")
DONE = ("completed", "failed", "cancelled")
JOB_ID = re.compile(r"[a-f0-9]{12}")
CHECKPOINT = re.compile(r"model_\d+\.pt")
ERRORS = (OSError, RuntimeError, ValueError, subprocess.SubprocessError, BadZipFile)


def validate(request):
    if not isinstance(request, dict):
        raise ValueError("任务必须是一个 JSON 对象")
    mode = request.get("mode", "train")
    target = request.get("target", "auto")
    backend = request.get("backend", "cuda")
    if mode not in ("train", "demo") or target not in ("auto", "local", "remote") or backend not in ("cuda", "cpu"):
        raise ValueError("任务类型、设备或后端无效")

    def number(key, default, low, high, kind):
        value = request.get(key)
        if value in (None, ""):
            return default
        try:
            if isinstance(value, bool) or (kind is int and isinstance(value, float) and not value.is_integer()):
                raise ValueError
            value = kind(value)
            valid = math.isfinite(value) and low <= value <= high
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{key} 必须是数字") from None
        if not valid:
            raise ValueError(f"{key} 必须在 {low}–{high} 之间")
        return value

    config = request.get("config") or ""
    if not isinstance(config, str) or (config and not re.fullmatch(r"configs/[\w.-]+\.toml", config)):
        raise ValueError("请从 configs 下选择 TOML 配置")
    checkpoint = request.get("checkpoint") or ""
    if not isinstance(checkpoint, str):
        raise ValueError("模型路径无效")
    checkpoint_mode = request.get("checkpoint_mode", "init" if mode == "train" else "replay")
    if checkpoint and checkpoint_mode not in (("init", "resume") if mode == "train" else ("replay",)):
        raise ValueError("请选择有效的模型加载方式")
    if checkpoint and checkpoint_mode == "resume" and config:
        raise ValueError("续训使用原训练配置，请清空配置一栏")
    heights = request.get("heights") or ""
    try:
        heights = [float(value) for value in str(heights).replace(",", " ").split()]
    except ValueError:
        raise ValueError("请用空格分隔腿高目标") from None
    if len(heights) > 20 or any(not (0.17 <= h <= 0.33) for h in heights):
        raise ValueError("腿高目标需在 0.17–0.33 m 之间，每次最多 20 个")
    surface = request.get("surface") or ""
    if surface not in ("", "flat", "slope", "rough", "bars", "stairs"):
        raise ValueError("地形无效")
    return {
        "name": str(request.get("name") or ("PPO 训练" if mode == "train" else "策略回放"))[:100],
        "mode": mode, "target": target, "backend": backend, "config": config, "checkpoint": checkpoint,
        "checkpoint_mode": checkpoint_mode, "envs": number("envs", None if checkpoint_mode == "resume" else (
            128 if mode == "train" else 1), 1, 16384, int),
        "iterations": number("iterations", 30, 1, 100000, int),
        "live_port": number("live_port", None, 1, 65535, int) if mode == "train" else None,
        "live_fps": number("live_fps", None, 1, 60, int) if mode == "train" and request.get("live_port") else None,
        "seed": number("seed", None, 0, 4294967295, int),
        "forward_limit": number("forward_limit", None, 0.001, 2.5, float),
        "seconds": number("seconds", None, 0.02, 120, float),
        "speed": number("speed", 0.35, -2.5, 2.5, float), "yaw": number("yaw", 0, -10, 10, float),
        "heights": " ".join(map(str, heights)), "surface": surface,
        "terrain_seed": number("terrain_seed", None, 0, 4294967295, int),
        "video": mode == "demo" and request.get("video") is True, "after": request.get("after") or "",
    }


def arguments(spec, input_name, node=None):
    args = [spec["mode"], "--backend", spec["backend"]]
    for key, flag in (("envs", "--num-envs"), ("seed", "--seed"), ("config", "--config")):
        if spec[key] not in ("", None):
            args.extend([flag, str(spec[key])])
    if input_name:
        flag = {"init": "--init-checkpoint", "resume": "--resume", "replay": "--checkpoint"}[spec["checkpoint_mode"]]
        args.extend([flag, f"../input/{input_name}"])
    if spec["mode"] == "train":
        args.extend(["--iterations", str(spec["iterations"]), "--output", "../artifacts"])
        if spec.get("live_port") is not None:
            args.extend(["--live-port", str(spec["live_port"])])
            if (node or spec["target"]) != "local":
                args.append("--live-stream")
            if spec.get("live_fps") is not None:
                args.extend(["--live-fps", str(spec["live_fps"])])
        if spec["forward_limit"] is not None:
            args.extend(["--forward-limit", str(spec["forward_limit"])])
    else:
        args.extend(["--speed", str(spec["speed"]), "--yaw", str(spec["yaw"])])
        for key, flag in (("seconds", "--seconds"), ("surface", "--surface"), ("terrain_seed", "--terrain-seed")):
            if spec[key] not in ("", None):
                args.extend([flag, str(spec[key])])
        if spec["heights"]:
            args.extend(["--height-sequence", *spec["heights"].split()])
        if spec["video"]:
            args.extend(["--video", "../artifacts/replay.mp4"])
    return args


class Scheduler:
    def __init__(self, state, nodes, diagnostic=None, auto_followup=False):
        self.state = state
        self.nodes = nodes
        self.diagnostic = diagnostic
        self.auto_followup = auto_followup
        self.snapshots = state / "snapshots"
        self.runs = state / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.database = state / "queue.sqlite3"
        self.submit_lock = Lock()
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, created REAL NOT NULL, data TEXT NOT NULL
            )""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def list(self, limit=100):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT data FROM jobs ORDER BY created DESC LIMIT ?", (limit,))]

    def get(self, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise ValueError("任务编号无效")
        with self.connect() as db:
            row = db.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("任务不存在")
        return json.loads(row[0])

    def update(self, job, *, previous=None):
        with self.connect() as db:
            statement = "UPDATE jobs SET status=?, data=? WHERE id=?"
            values = (job["status"], json.dumps(job, ensure_ascii=False), job["id"])
            if previous is not None:
                statement += " AND status=?"
                values += (previous,)
            return bool(db.execute(statement, values).rowcount)

    def checkpoint(self, value):
        if value.startswith(("run:", "input:")):
            location, _, reference = value.partition(":")
            job_id, _, filename = reference.partition("/")
            job = self.get(job_id)
            valid = filename in job["artifacts"] if location == "run" else filename == job.get("input_name")
            if not CHECKPOINT.fullmatch(filename) or not valid:
                raise ValueError("找不到指定的训练模型")
            path = self.runs / job_id / ("artifacts" if location == "run" else "input") / filename
            allowed = self.runs.resolve()
        else:
            path = TRAINING / Path(value)
            allowed = (TRAINING / "outputs").resolve()
        if not path.resolve().is_relative_to(allowed) or not path.is_file() or not CHECKPOINT.fullmatch(path.name):
            raise ValueError("请选择 outputs 下的 model_*.pt 或已同步的模型")
        return path

    def catalog(self):
        configs = sorted(path.relative_to(TRAINING).as_posix() for path in (TRAINING / "configs").glob("*.toml"))
        ordinary = [path for path in (TRAINING / "outputs").glob("*/model_*.pt") if CHECKPOINT.fullmatch(path.name)]
        ordinary.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        models = [{"value": path.relative_to(TRAINING).as_posix(), "label": f"{path.parent.name}/{path.name}"}
                  for path in ordinary[:60]]
        for job in self.list(60):
            for name in sorted((name for name in job["artifacts"] if CHECKPOINT.fullmatch(name)),
                               key=lambda name: int(name[6:-3]), reverse=True):
                models.append({"value": f"run:{job['id']}/{name}", "label": f"{job['spec']['name']}/{name}"})
        return {"configs": configs, "models": models}

    def submit(self, request, *, source=None, parent=None):
        spec = validate(request)
        if spec["target"] == "remote" and "remote" not in self.nodes:
            raise ValueError("目前配置为仅本机")
        if spec["after"]:
            self.get(spec["after"])
        original = self.checkpoint(spec["checkpoint"]) if spec["checkpoint"] else None
        if spec["checkpoint_mode"] == "resume" and (not original or not original.with_name("run_config.json").is_file()):
            raise ValueError("续训需要同时提供模型和同目录的 run_config.json")
        with self.submit_lock:
            source = source or snapshot(self.snapshots)
            if spec["config"]:
                with ZipFile(self.snapshots / f"{source['sha256']}.zip") as archive:
                    if spec["config"] not in archive.namelist():
                        raise ValueError("所选配置不在训练快照中")
            job_id = uuid4().hex[:12]
            directory = self.runs / job_id
            (directory / "artifacts").mkdir(parents=True)
            if original:
                target = directory / "input"
                target.mkdir()
                model = target / original.name
                shutil.copy2(original, model)
                with ZipFile(model) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("所选 checkpoint 尚未写完")
                config = original.with_name("run_config.json")
                if config.is_file():
                    shutil.copy2(config, target / config.name)
            job = {"id": job_id, "created": time.time(), "status": "queued", "node": None, "spec": spec,
                   "source": source, "arguments": arguments(spec, original.name if original else None),
                   "input_name": original.name if original else None,
                   "artifacts": {}, "parent": parent, "log": "", "message": "", "diagnosis": None}
            with self.connect() as db:
                db.execute("INSERT INTO jobs VALUES (?,?,?,?)",
                           (job_id, job["status"], job["created"], json.dumps(job, ensure_ascii=False)))
        return job

    def cancel(self, job_id):
        job = self.get(job_id)
        job.update(status="cancelled", message="任务已取消")
        if not self.update(job, previous="queued"):
            raise ValueError("可以取消仍在等待的任务")
        return job

    def choose(self, spec, reserved):
        if spec["target"] == "auto":
            long_train = spec["mode"] == "train" and spec["iterations"] > 30
            order = ("remote", "local") if long_train else ("local", "remote")
        else:
            order = (spec["target"],)
        for name in order:
            node = self.nodes.get(name)
            if not node or name in reserved or not node.info["online"]:
                continue
            if spec["backend"] == "cpu" or (node.info["cuda"] and not node.info["busy"]):
                return name
        return None

    def launch(self, job):
        node = self.nodes[job["node"]]
        job["arguments"] = arguments(job["spec"], job["input_name"], job["node"])
        try:
            node.prepare(job, self.runs / job["id"], self.snapshots)
        except ERRORS as error:
            node.probe()
            if job["spec"]["target"] == "auto" and not node.info["online"]:
                job.update(status="queued", node=None, message="节点离线，等待可用设备")
            else:
                job.update(status="failed", message=f"准备训练失败。{error}")
            self.update(job)
            return
        job.update(status="starting", launched=time.time(), message="", gpu=node.info.get("gpu"))
        self.update(job)
        try:
            node.start(job["id"])
        except ERRORS as error:
            job.update(status="disconnected", message=f"启动状态待核对。{error}")
            self.update(job)

    def refresh(self, job):
        node = self.nodes.get(job["node"])
        if job["status"] == "staging":
            # The launch transition is persisted before any process can start.
            job.update(status="queued", node=None)
            self.update(job)
            return
        if node is None or not node.info["online"]:
            job.update(status="disconnected", message="节点离线，等待重新核对训练进度")
            self.update(job)
            return
        try:
            result = node.inspect(job["id"])
        except ERRORS as error:
            job.update(status="disconnected", message=str(error))
            self.update(job)
            return
        job["log"] = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.get("log") or "")[-16000:]
        status = result.get("status") or {}
        collection_errors = []
        for entry in sorted(result["artifacts"], key=lambda item: item["name"] != "run_config.json"):
            name = entry["name"]
            if not ARTIFACT.fullmatch(name) or job["artifacts"].get(name) == entry["sha256"]:
                continue
            if CHECKPOINT.fullmatch(name) and "run_config.json" not in job["artifacts"]:
                collection_errors.append("等待与 checkpoint 配套的 run_config.json")
                continue
            try:
                node.collect(job["id"], entry, self.runs / job["id"] / "artifacts")
            except ERRORS as error:
                collection_errors.append(str(error))
            else:
                job["artifacts"][name] = entry["sha256"]
        job["message"] = ""
        if status.get("phase") == "finished":
            job["exit_code"] = status.get("exit_code")
            if collection_errors:
                job.update(status="syncing", message=collection_errors[0])
            elif job["exit_code"] == 0:
                if job["spec"]["mode"] == "train" and not any(CHECKPOINT.fullmatch(name) for name in job["artifacts"]):
                    job.update(status="failed", message="进程已退出，任务目录中缺少可用 checkpoint")
                else:
                    job["status"] = "completed"
            else:
                job.update(status="failed", message=f"训练进程退出，退出码 {job['exit_code']}")
        elif result["alive"]:
            job["status"] = "running" if status.get("phase") == "running" else "starting"
        elif time.time() - job["launched"] > 30:
            job.update(status="failed", message="已确认任务进程停止，可从已同步的 checkpoint 恢复")
        else:
            job["status"] = "starting"
        self.update(job)

    def tick(self):
        for node in self.nodes.values():
            node.probe()
        # Active and queued jobs must be considered even if older than the UI history.
        with self.connect() as db:
            jobs = [json.loads(row[0]) for row in db.execute(
                "SELECT data FROM jobs WHERE status NOT IN (?,?,?) ORDER BY created", DONE)]
        for job in jobs:
            if job["status"] in ACTIVE:
                self.refresh(job)
        reserved = {job["node"] for job in jobs if job["status"] in ACTIVE}
        for job in jobs:
            if job["status"] != "queued":
                continue
            if job["spec"]["after"]:
                upstream = self.get(job["spec"]["after"])
                if upstream["status"] in ("failed", "cancelled"):
                    job.update(status="cancelled", message="前置任务未通过")
                    self.update(job, previous="queued")
                    continue
                if upstream["status"] != "completed":
                    continue
            node = self.choose(job["spec"], reserved)
            if node:
                job.update(status="staging", node=node, message="")
                if self.update(job, previous="queued"):
                    reserved.add(node)
                    self.launch(job)

    def followup(self, job_id, action):
        job = self.get(job_id)
        if job["status"] not in DONE:
            raise ValueError("请等当前任务结束后再派发后续任务")
        spec = dict(job["spec"], after="")
        if job.get("input_name"):
            spec["checkpoint"] = f"input:{job_id}/{job['input_name']}"
        checkpoints = [name for name in job["artifacts"] if CHECKPOINT.fullmatch(name)]
        latest = max(checkpoints, key=lambda name: int(name[6:-3]), default=None)
        if action == "smoke" and spec["mode"] == "train":
            spec.update(name=f"{spec['name']} · 短训复查", envs=min(spec["envs"] or 128, 128),
                        target="auto", iterations=min(spec["iterations"], 30))
        elif action == "resume" and latest:
            spec.update(name=f"{spec['name']} · 续训", checkpoint=f"run:{job_id}/{latest}", checkpoint_mode="resume",
                        config="", seed=None, forward_limit=None, envs=None, target="auto")
        elif action == "replay" and (latest or spec["checkpoint"]):
            spec.update(name=f"{spec['name']} · 回放", mode="demo", checkpoint_mode="replay", target="auto",
                        checkpoint=f"run:{job_id}/{latest}" if latest else spec["checkpoint"],
                        envs=1, config="", video=False)
        else:
            raise ValueError("该任务尚无可用于此操作的模型，或任务类型不支持")
        return self.submit(spec, source=job["source"], parent=job_id)
