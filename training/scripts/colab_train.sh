#!/usr/bin/env bash
# Wheel Leg V0.2 — Google Colab CLI 远程训练、代码同步、本机监控转发与权重拉回一体化脚本
# 依赖：WSL 中已安装并登录 google-colab-cli (~/.local/bin/colab) 及 ~/.ssh/id_ed25519
#
# 用法（在 WSL 中直接运行，或在 Windows PowerShell 通过 training/scripts/colab_train.ps1 调用）：
#   colab_train.sh up      [--session NAME] [--gpus "A100 L4 T4"] [--config ...] [--num-envs 2048] [--iterations 1200] [--init-ckpt PATH] [--run-name NAME]
#   colab_train.sh sync    [--session NAME]                     # 仅同步本地 training/ 改动到远端
#   colab_train.sh train   [--session NAME] [--config ...] ...  # 在已有会话上重跑训练与 TensorBoard
#   colab_train.sh tunnel  [--session NAME] [--live-port 8765] [--tb-port 6006]  # 本机监控端口转发（断线自动清 429 重连）
#   colab_train.sh status  [--session NAME] [--run-name NAME]   # 查看 GPU、tmux、端口、checkpoints 与最新日志
#   colab_train.sh pull    [--session NAME] --run-name NAME [--ckpt model_1200.pt]  # 拉回权重和 run_config.json 到本地
#   colab_train.sh usage                                        # 查看 Colab 算力余额与活跃会话
#   colab_train.sh stop    [--session NAME]                     # 停止 Colab 会话释放计费

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

COLAB="${COLAB_BIN:-$HOME/.local/bin/colab}"
REPO_URL="${COLAB_REPO_URL:-https://github.com/SunWater-233/wheel_leg_rl_parallel.git}"
REMOTE_REPO="/content/wheel_leg_rl_parallel"
REMOTE_TRAIN="$REMOTE_REPO/training"
SSH_KEY="${COLAB_SSH_KEY:-$HOME/.ssh/id_ed25519}"

SESSION="wheel-leg-s2r"
GPUS="L4 A100 T4"
CONFIG="configs/v02_sim2real_stage2_robust.toml"
NUM_ENVS="2048"
ITERATIONS="1200"
RUN_NAME="v02_sim2real_stage2_robust"
INIT_CKPT=""
RESUME_CKPT=""
CKPT_NAME=""
LIVE_PORT="8765"
TB_PORT="6006"
LIVE_FPS="8"
SEED="42"
SKIP_TESTS="0"
SKIP_TUNNEL="0"

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

CMD="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--session)     SESSION="$2"; shift 2 ;;
    --gpus)           GPUS="$2"; shift 2 ;;
    --config)         CONFIG="$2"; shift 2 ;;
    --num-envs)       NUM_ENVS="$2"; shift 2 ;;
    --iterations)     ITERATIONS="$2"; shift 2 ;;
    --run-name)       RUN_NAME="$2"; shift 2 ;;
    --init-ckpt)      INIT_CKPT="$2"; shift 2 ;;
    --resume)         RESUME_CKPT="$2"; shift 2 ;;
    --ckpt)           CKPT_NAME="$2"; shift 2 ;;
    --live-port)      LIVE_PORT="$2"; shift 2 ;;
    --tb-port)        TB_PORT="$2"; shift 2 ;;
    --live-fps)       LIVE_FPS="$2"; shift 2 ;;
    --seed)           SEED="$2"; shift 2 ;;
    --skip-tests)     SKIP_TESTS="1"; shift ;;
    --no-tunnel)      SKIP_TUNNEL="1"; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

require_colab() {
  if [[ ! -x "$COLAB" ]]; then
    echo "ERROR: colab CLI not found at $COLAB" >&2
    exit 1
  fi
}

colab_exec_py() {
  local timeout="$1"
  local tmp_py
  tmp_py="$(mktemp /tmp/colab_exec_XXXXXX.py)"
  cat > "$tmp_py"
  "$COLAB" exec -s "$SESSION" --timeout "$timeout" -f "$tmp_py"
  local rc=$?
  rm -f "$tmp_py"
  return $rc
}

clear_remote_sshd() {
  # 只清理卡死的 sshd 子会话，保留主监听进程 (/usr/sbin/sshd -D)，解决断线重连后的 HTTP 429
  colab_exec_py 25 <<'EOF' >/dev/null 2>&1 || true
import subprocess
subprocess.run("pkill -9 -f '^sshd: root' || true", shell=True)
EOF
}

pack_and_upload_overlay() {
  local overlay="/tmp/colab_overlay_${SESSION}.tar.gz"
  local list_file="/tmp/colab_overlay_${SESSION}.files"

  # 收集本地 training/ 和 deploy/host/export_policy.py 中所有未提交修改及新增代码/配置
  (
    cd "$REPO_ROOT"
    git status --porcelain -- training deploy/host/export_policy.py \
      | awk '{print $NF}' \
      | grep -Ev '(\.venv/|__pycache__/|outputs/|\.cache/|\.log$)' || true
  ) > "$list_file"

  if [[ -s "$list_file" ]]; then
    echo "=== 打包本地未提交改动 ($(wc -l < "$list_file") 个文件) ==="
    cat "$list_file"
    tar -czf "$overlay" -C "$REPO_ROOT" -T "$list_file"
    "$COLAB" upload -s "$SESSION" "$overlay" /content/colab_overlay.tar.gz
    colab_exec_py 60 <<EOF
import subprocess
subprocess.run(["tar", "-xzf", "/content/colab_overlay.tar.gz", "-C", "$REMOTE_REPO"], check=True)
print(subprocess.check_output(["git", "-C", "$REMOTE_REPO", "status", "-s"]).decode())
EOF
  else
    echo "=== 本地无未提交改动，直接使用远端 git 仓库代码 ==="
  fi
  rm -f "$overlay" "$list_file"
}

upload_init_ckpt_if_needed() {
  if [[ -z "$INIT_CKPT" ]]; then
    return 0
  fi
  # 若路径以 /content/ 开头则视为远端已有文件；否则从本地上传到 /content/
  if [[ "$INIT_CKPT" == /content/* ]]; then
    echo "=== 使用远端初始权重: $INIT_CKPT ==="
    return 0
  fi
  local local_path="$INIT_CKPT"
  if [[ ! -f "$local_path" && -f "$REPO_ROOT/$INIT_CKPT" ]]; then
    local_path="$REPO_ROOT/$INIT_CKPT"
  fi
  if [[ ! -f "$local_path" ]]; then
    echo "ERROR: 找不到初始权重文件: $INIT_CKPT" >&2
    exit 1
  fi
  local remote_ckpt="/content/init_$(basename "$local_path")"
  echo "=== 上传初始权重: $local_path -> $remote_ckpt ==="
  "$COLAB" upload -s "$SESSION" "$local_path" "$remote_ckpt"
  INIT_CKPT="$remote_ckpt"
}

cmd_allocate() {
  echo "=== 1. 分配 Colab GPU (会话: $SESSION, 候选: $GPUS) ==="
  if "$COLAB" sessions 2>/dev/null | grep -q "$SESSION"; then
    echo "会话 '$SESSION' 已存在，直接复用。"
    return 0
  fi
  local allocated=""
  for gpu in $GPUS; do
    echo "尝试申请 GPU=$gpu ..."
    if COLAB_CLI_RETRIES=1 COLAB_CLI_RETRY_BACKOFF=5 "$COLAB" new -s "$SESSION" --gpu "$gpu"; then
      allocated="$gpu"
      echo "成功分配 GPU=$gpu!"
      break
    fi
    echo "GPU=$gpu 暂不可用，尝试下一个..."
  done
  if [[ -z "$allocated" ]]; then
    echo "ERROR: 所有候选 GPU ($GPUS) 均申请失败。" >&2
    exit 1
  fi
}

cmd_prepare_env() {
  echo "=== 2. 克隆/更新仓库 ==="
  colab_exec_py 180 <<EOF
import os, subprocess
print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]).decode().strip())
repo = "$REMOTE_REPO"
if not os.path.exists(repo):
    subprocess.run(["git", "clone", "--depth", "1", "$REPO_URL", repo], check=True)
else:
    subprocess.run(["git", "-C", repo, "pull", "--ff-only"], check=False)
EOF

  pack_and_upload_overlay
  upload_init_ckpt_if_needed

  echo "=== 3. 安装依赖 (uv sync) 与运行单元测试 ==="
  colab_exec_py 600 <<EOF
import os, shutil, subprocess, time

def run(cmd, *, cwd=None, check=True, env=None):
    print(f"\$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {cmd}")
    return rc

train_dir = "$REMOTE_TRAIN"
if not shutil.which("uv"):
    run("curl -LsSf https://astral.sh/uv/install.sh | sh")
    os.environ["PATH"] = f"{os.path.expanduser('~/.local/bin')}:{os.environ['PATH']}"

for _ in range(30):
    if not subprocess.run(["pgrep", "-af", "uv sync"], capture_output=True, text=True).stdout.strip():
        break
    time.sleep(3)

if run("uv sync", cwd=train_dir, check=False) != 0:
    run("uv sync --default-index https://pypi.org/simple/", cwd=train_dir, check=True)

run(
    ".venv/bin/python -c 'import torch, genesis, rsl_rl, mujoco; "
    "print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), "
    "torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\")'",
    cwd=train_dir,
)
if "$SKIP_TESTS" != "1":
    run(
        ".venv/bin/python -m unittest tests/test_sim2real_delay.py tests/test_scut_agility.py tests/test_actor_history.py -v",
        cwd=train_dir,
    )
EOF
}

cmd_train() {
  upload_init_ckpt_if_needed
  echo "=== 4. 在 tmux 中启动训练 (live-port $LIVE_PORT) 与 TensorBoard ($TB_PORT) ==="
  colab_exec_py 120 <<EOF
import os, pathlib, subprocess, time

train_dir = "$REMOTE_TRAIN"
out_root = pathlib.Path(train_dir) / "outputs"
out_root.mkdir(parents=True, exist_ok=True)

# 检测无头渲染后端 (优先 egl)
env_egl = {**os.environ, "MUJOCO_GL": "egl"}
rc_egl = subprocess.run(
    [f"{train_dir}/.venv/bin/python", "-c",
     "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type=\"plane\" size=\"1 1 0.1\"/></worldbody></mujoco>'); d=mujoco.MjData(m); r=mujoco.Renderer(m, 64, 64); r.update_scene(d); r.render()"],
    env=env_egl, capture_output=True
).returncode
mujoco_gl = "egl" if rc_egl == 0 else "osmesa"
print("MUJOCO_GL =", mujoco_gl, flush=True)

extra_args = []
if "$RESUME_CKPT":
    extra_args += ["--resume", "$RESUME_CKPT"]
elif "$INIT_CKPT":
    extra_args += ["--init-checkpoint", "$INIT_CKPT"]
extra_str = " ".join(extra_args)

out_dir = f"outputs/$RUN_NAME"
if (pathlib.Path(train_dir) / out_dir).exists() and any((pathlib.Path(train_dir) / out_dir).iterdir()):
    out_dir = f"outputs/{'$RUN_NAME'}_{time.strftime('%H%M%S')}"
log_file = f"outputs/train_{pathlib.Path(out_dir).name}.log"

run_sh = out_root / f"run_{pathlib.Path(out_dir).name}.sh"
run_sh.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
cd {train_dir}
export MUJOCO_GL={mujoco_gl}
export QD_OFFLINE_CACHE_FILE_PATH={train_dir}/.cache/quadrants
export PYTHONUNBUFFERED=1

echo "=== [\$(date)] Starting {out_dir} (envs=$NUM_ENVS, iters=$ITERATIONS, live=$LIVE_PORT) ==="
.venv/bin/python main.py train \\
    --robot v0.2 \\
    --num-envs $NUM_ENVS \\
    --iterations $ITERATIONS \\
    --seed $SEED \\
    --config $CONFIG \\
    {extra_str} \\
    --output "{out_dir}" \\
    --live-port $LIVE_PORT --live-stream --live-fps $LIVE_FPS 2>&1 | tee "{log_file}"
echo "=== [\$(date)] Training finished! ==="
""", encoding="utf-8")
os.chmod(run_sh, 0o755)

subprocess.run("tmux kill-session -t rl_train 2>/dev/null || true", shell=True)
subprocess.run("tmux kill-session -t tb_mon 2>/dev/null || true", shell=True)
subprocess.run(f"tmux new-session -d -s rl_train 'bash {run_sh}'", shell=True, check=True)
subprocess.run(
    f"tmux new-session -d -s tb_mon 'cd {train_dir} && .venv/bin/tensorboard --logdir outputs --port $TB_PORT --host 127.0.0.1 > outputs/tensorboard.log 2>&1'",
    shell=True, check=True,
)
time.sleep(3)
print(subprocess.check_output("tmux ls", shell=True).decode())
print("Output dir:", out_dir, "| Log:", log_file)
EOF
}

cmd_tunnel() {
  if [[ ! -f "$SSH_KEY" ]]; then
    echo "ERROR: SSH 私钥不存在: $SSH_KEY，请先运行 ssh-keygen -t ed25519" >&2
    exit 1
  fi
  echo "=== 开启本机监控端口转发（Ctrl+C 停止）==="
  echo "  实时训练监控: http://127.0.0.1:${LIVE_PORT}/"
  echo "  TensorBoard : http://127.0.0.1:${TB_PORT}/"
  pkill -f "colab ssh --proxy-mode -s $SESSION" 2>/dev/null || true
  pkill -f "root@colab" 2>/dev/null || true

  while true; do
    clear_remote_sshd
    sleep 2
    echo "[$(date '+%H:%M:%S')] Connecting SSH tunnel -> $SESSION (127.0.0.1:$LIVE_PORT & 127.0.0.1:$TB_PORT) ..."
    ssh -N \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o ExitOnForwardFailure=yes \
      -o TCPKeepAlive=yes \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=4 \
      -o ProxyCommand="$COLAB ssh --proxy-mode -s $SESSION" \
      -i "$SSH_KEY" \
      -L "127.0.0.1:${LIVE_PORT}:127.0.0.1:${LIVE_PORT}" \
      -L "127.0.0.1:${TB_PORT}:127.0.0.1:${TB_PORT}" \
      root@colab || true
    echo "[$(date '+%H:%M:%S')] Tunnel disconnected; cleaning remote sshd & retrying in 8s..."
    sleep 8
  done
}

cmd_status() {
  colab_exec_py 45 <<EOF
import glob, os, subprocess
for title, cmd in [
    ("GPU", "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader"),
    ("TMUX", "tmux ls 2>&1 || true"),
    ("PORTS", "ss -tlnp 2>/dev/null | grep -E '$LIVE_PORT|$TB_PORT' || true"),
    ("CHECKPOINTS", "ls -lht $REMOTE_TRAIN/outputs/$RUN_NAME/model_*.pt 2>/dev/null | head -n 10 || ls -lht $REMOTE_TRAIN/outputs/*/model_*.pt 2>/dev/null | head -n 10 || echo 'none yet'"),
]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"=== {title} ===\n{(r.stdout + r.stderr).strip()}\n")

logs = sorted(glob.glob("$REMOTE_TRAIN/outputs/train*.log"), key=os.path.getmtime, reverse=True)
if logs:
    print(f"=== LOG TAIL ({os.path.basename(logs[0])}) ===")
    print(subprocess.run(["tail", "-n", "30", logs[0]], capture_output=True, text=True).stdout)
EOF
}

cmd_pull() {
  local local_dir="$REPO_ROOT/training/outputs/$RUN_NAME"
  mkdir -p "$local_dir"
  local remote_dir="$REMOTE_TRAIN/outputs/$RUN_NAME"

  if [[ -z "$CKPT_NAME" ]]; then
    CKPT_NAME="$(colab_exec_py 30 <<EOF | tail -n 1 | tr -d '\r'
import glob, os
ckpts = sorted(glob.glob("$remote_dir/model_*.pt"), key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
if not ckpts:
    raise SystemExit("No checkpoints found in $remote_dir")
print(os.path.basename(ckpts[-1]))
EOF
)"
  fi

  echo "=== 拉回权重与配置: $remote_dir/$CKPT_NAME -> $local_dir/ ==="
  "$COLAB" download -s "$SESSION" "$remote_dir/run_config.json" "$local_dir/run_config.json"
  "$COLAB" download -s "$SESSION" "$remote_dir/$CKPT_NAME" "$local_dir/$CKPT_NAME"
  ls -lh "$local_dir/run_config.json" "$local_dir/$CKPT_NAME"
  echo "导出实机 bundle 命令："
  echo "  python deploy/host/export_policy.py --checkpoint training/outputs/$RUN_NAME/$CKPT_NAME"
}

require_colab

case "$CMD" in
  up)
    cmd_allocate
    cmd_prepare_env
    cmd_train
    if [[ "$SKIP_TUNNEL" != "1" ]]; then
      cmd_tunnel
    fi
    ;;
  sync)
    pack_and_upload_overlay
    ;;
  train)
    cmd_train
    ;;
  tunnel)
    cmd_tunnel
    ;;
  status)
    cmd_status
    ;;
  pull)
    cmd_pull
    ;;
  usage)
    "$COLAB" usage
    "$COLAB" sessions
    ;;
  stop)
    clear_remote_sshd
    "$COLAB" stop -s "$SESSION"
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    usage
    exit 1
    ;;
esac
