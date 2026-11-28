# 自研轮腿 V0.2 · Genesis 并行强化学习训练

基于 Genesis GPU 并行物理仿真与 RSL-RL PPO 的轮腿机器人训练框架，默认加载自研模型 [wheel_leg_ground.xml](assets/wheel_leg_xml_V0.2/wheel_leg_ground.xml)（500 Hz 物理步进 / 50 Hz 策略控制，6 处闭链由 12 条 `connect` 约束实现）。

动作空间依次控制 `hip_L`、`hip_R`、`hipknee_L`、`hipknee_R` 的目标位置和 `wheel_L`、`wheel_R` 的目标角速度，右侧镜像符号由环境统一处理；策略观测默认采用 28D 本体状态（支持 `actor_history_steps` 历史帧堆叠与非对称 Critic 地形/线速度观测）。

---

## 项目结构

```text
training/
├── main.py                  # CLI 入口（train / demo 参数解析与分发）
├── pyproject.toml           # uv 依赖与镜像配置
├── assets/                  # V0.2 / SCUT MJCF、CAD 网格与授权
├── configs/
│   ├── flat.toml            # 全局默认配置（所有可覆写参数的基线 schema）
│   ├── v02_sim2real_stage1_flat.toml     # Sim2Real 一阶段（平地 + 延迟 + 域随机化）
│   ├── v02_sim2real_stage2_robust.toml   # Sim2Real 二阶段（复杂地形 + 课程 + 外部扰动）
│   ├── v02_scut_stage1_flat_speed.toml   # SCUT 对齐一阶段（平地敏捷速度/小陀螺）
│   ├── v02_scut_stage2_terrain.toml      # SCUT 对齐二阶段（全地形贴地跑场）
│   ├── archive/             # 历史实验与消融配置归档
│   └── schema/              # TOML 加载（支持 extends 继承）、校验与 run_config.json 快照
├── robots/                  # 纯机器人硬件抽象（无任务/奖励耦合）
│   ├── v02/                 # spec.py 单一常量源、Genesis 适配器、闭链姿态求解、固件自起移植
│   └── scut/                # SCUT Wheelbipe V14.2 参考适配器
├── tasks/                   # RL 环境与可组合任务模块
│   ├── base_env.py          # FlatEnv（VecEnv 步进、动作/IMU 延迟缓冲、姿态终止）
│   ├── locomotion_env.py    # WheelLegLocomotionEnv 编排器
│   ├── terrains/            # 混合/SCUT/过渡高度场与视觉材质
│   ├── commands/            # SpecialMode / 地形指令 / 航向锁 / 摩擦圆 / 旧版机动
│   ├── rewards/             # 华南虎 V14 Rough-v1 奖励与制动俯仰代价
│   ├── randomization/       # 动力学域随机化与外部扰动事件（push/wrench/air_drop）
│   ├── observations.py      # 观测拼装、传感器噪声、IMU 延迟与历史帧
│   ├── recovery.py          # 低位断电落地恢复训练子任务
│   └── metrics.py           # 回合指标统计与自适应地形课程
├── rl/                      # 训练与策略推理核心（runner / checkpoint / trainer / evaluator）
├── eval/                    # 验收评测套件（共享 harness + 过坎/平地升降/复杂地形/过渡制动等）
├── viz/                     # 实时网页监控、Genesis 键鼠驾驶、斜坡规划、视频录制与 sim_logger
├── scripts/                 # Colab 一体化远程训练脚本（colab_train.*）、资产转换与旧脚本归档
├── tests/                   # 单元测试（unittest）
├── envs/ & tools/           # 向后兼容 re-export 壳（旧命令与旧 import 保持可用）
└── outputs/                 # 本地训练产物、checkpoint、TensorBoard 日志与回放视频（不入库）
```

---

## 快速开始

以下命令在 `wheel_leg_rl_parallel/training` 下执行：

```powershell
uv sync --locked

# 1. 脚本控制器冒烟 + 录制 CAD 网格视频
uv run --locked main.py demo --num-envs 4 --seconds 5 --video outputs/wheel_leg_v02.mp4

# 2. 运行全部单元测试
uv run --locked python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 主线训练流程（Sim2Real）

### 本机训练与续训

```powershell
# 一阶段：平地敏捷 + 执行器/IMU 延迟 + 域随机化
uv run --locked main.py train --config configs/v02_sim2real_stage1_flat.toml --num-envs 1024 --iterations 800 --output outputs/v02_sim2real_stage1 --live-port 6007

# 二阶段：从一阶段权重热启动（--init-checkpoint 使用新优化器，自动零填充新增维度）
uv run --locked main.py train --config configs/v02_sim2real_stage2_robust.toml --init-checkpoint outputs/v02_sim2real_stage1/model_800.pt --num-envs 1536 --iterations 1200 --output outputs/v02_sim2real_stage2 --live-port 6007 --live-stream --live-fps 8

# 断点续训（保留优化器与迭代编号，自动读取同目录 run_config.json）
uv run --locked main.py train --resume outputs/v02_sim2real_stage2/model_500.pt --iterations 500 --output outputs/v02_sim2real_stage2_resume
```

配置文件支持 `extends = "base_name.toml"` 继承同目录或 `training/` 相对路径的父配置，只需写差异项；每次训练仍会将展平后的完整配置写入 `outputs/<run>/run_config.json`，保证部署导出与复现兼容。

### Colab 远程训练与本机监控（`colab_train.*`）

通过 WSL `google-colab-cli` 封装 GPU 申请、本地未提交改动覆盖、权重上传、`uv sync`、单元测试、`tmux` 后台训练与防 HTTP 429 自动重连端口转发（脚本位于 [scripts/colab_train.cmd](scripts/colab_train.cmd)，`tools/colab_train.cmd` 保留兼容转发）：

```powershell
# 1. 一键分配 GPU、同步代码、跑单测、启动训练并转发监控到本机浏览器
.\training\scripts\colab_train.cmd up --session wheel-leg-s2r --gpus "L4 A100 T4" --config configs/v02_sim2real_stage2_robust.toml --num-envs 2048 --iterations 1200 --init-ckpt training/outputs/v02_scut_stage2_scut_aligned/model_1600.pt

# 2. 单独开启/重连监控转发：http://127.0.0.1:8765/ (实时画面) 与 http://127.0.0.1:6006/ (TensorBoard)
.\training\scripts\colab_train.cmd tunnel --session wheel-leg-s2r

# 3. 查看远端 GPU、tmux、checkpoint 列表与日志尾部
.\training\scripts\colab_train.cmd status --session wheel-leg-s2r

# 4. 拉回最新（或指定）checkpoint 与 run_config.json 到本地 training/outputs/
.\training\scripts\colab_train.cmd pull --session wheel-leg-s2r --run-name v02_sim2real_stage2_robust

# 5. 查看算力余额 / 停止实例释放计费
.\training\scripts\colab_train.cmd usage
.\training\scripts\colab_train.cmd stop --session wheel-leg-s2r
```

多机调度也可直接通过同仓库根目录的 [training_manager/](../training_manager) 进行任务排队与远程拉取。

---

## 回放驾驶、实时日志与验收评测

### 1. Genesis 键盘驾驶与实时遥测

```powershell
# 原生 Genesis 窗口键鼠驾驶（W/S 前后、A/D 转向、Z/X 连续腿高、Space 刹停、R 复位）
# 倒地默认触发固件 RESCUE -> GETUP 自起并交还策略控制（加 --fall-reset 可恢复旧复位行为）
uv run --locked main.py demo --checkpoint outputs/<run>/model_1200.pt --keyboard --speed 2.5

# 全赛道自由驾驶
uv run --locked main.py demo --checkpoint outputs/<run>/model_1200.pt --keyboard --free-drive

# 指定地形与难度回放（支持 flat / slope / inv_slope / rough / bars / stairs / rough_slope / staggered_bars / single_side_bars_left|right / speed_bump）
uv run --locked main.py demo --checkpoint outputs/<run>/model_1200.pt --surface single_side_bars_left --terrain-variant 3 --speed -0.55 --height 0.20 --viewer

# 远程网页键鼠回放（配合 SSH -L 6007:127.0.0.1:6007）
uv run --locked main.py demo --checkpoint outputs/<run>/model_1200.pt --live-port 6007 --keyboard
```

每次 `main.py demo` 默认将 env 0 的 50 Hz 遥测与跌倒事件记录到 `outputs/sim_logs/<timestamp>_<run>/`（`telemetry.csv`、`events.log`、`incidents/fall_XXX.html`、`summary.json`）；可用 `--sim-log DIR` 指定路径或 `--no-sim-log` 关闭。

### 2. 自动化验收对比套件（`eval/`）

```powershell
# 平地升降、启停换向与低位恢复对比
uv run --locked python -m eval.flat_height --config configs/archive/v02_sim2real_flat_height.toml --output outputs/flat_acceptance.csv CKPT_A CKPT_B

# 单凸条过坎 -> 停稳 -> 倒车跟踪验收
uv run --locked python -m eval.bump --checkpoint outputs/<run>/model_1200.pt --output outputs/bump_acceptance.csv

# 曲面减速带（双轮 / 仅左轮 / 仅右轮）对比
uv run --locked python -m eval.terrain_obstacle --output outputs/obstacle.csv --trace outputs/obstacle_trace.csv CKPT_A CKPT_B

# 复杂地形全赛道通过率对比
uv run --locked python -m eval.complex_terrain --output outputs/complex.csv CKPT_A CKPT_B
```

> 兼容提示：旧的 `python -m tools.*_acceptance` 调用路径已通过薄壳保留，行为与 `eval.*` 完全一致。

### 3. 导出实机部署 Bundle

```powershell
python ../deploy/host/export_policy.py --checkpoint outputs/<run>/model_1200.pt
```

### 4. TensorBoard 与 SCUT 参考模型

```powershell
uv run --locked tensorboard --logdir outputs --host 127.0.0.1 --port 6006
uv run --locked main.py demo --robot scut
uv run --locked python -m scripts.convert_wheelbipe
```
