# 自研轮腿 V0.2 · Genesis 训练

默认加载自研模型的 [wheel_leg_ground.xml](assets/wheel_leg_xml_V0.2/wheel_leg_ground.xml)。训练统一使用 [华南虎 V14 Rough-v1 reward 的 Genesis 移植](robots/scut_rough_reward.py)。连续腿高任务使用完整的 reward，固定高度平地任务使用其中的地面项。V0.2 按 CAD 原件导入质量、惯量、关节限位和 6 处闭链，每处闭链由一对 `connect` 约束实现。训练以 500 Hz 步进物理、50 Hz 更新控制。

V0.2 的站立复位姿态让双轮轴位于重心下方。动作依次控制 `hip_L`、`hip_R`、`hipknee_L`、`hipknee_R` 的位置和 `wheel_L`、`wheel_R` 的转速，右侧镜像轴的符号由环境处理。膝关节由连杆驱动，气弹簧沿原模型轴施加 200 N，四个腿部驱动默认沿用原模型的 ±12 N·m 限幅。连续腿高任务可用 `task.locomotion.leg_torque_limit` 调整单关节限矩。轮端限矩 ±5 N·m、PD 参数和被动关节阻尼是仿真假设。

批量仿真用按 CAD 拟合的轮胎和导轮圆柱，以及其余刚体的包围盒，平地场景关闭自碰撞。原始 CAD 文件供回放绘图。

## 项目结构

```text
training/
├── main.py              # train 和 demo 入口
├── assets/              # 机器人 MJCF、CAD 网格和授权
├── robots/              # 自研 V0.2 和 SCUT 模型的姿态、关节映射与驱动参数
├── envs/flat.py         # 并行仿真、指令、观测、奖励、终止和重置
├── configs/flat.toml    # 平地任务和 PPO 默认配置
├── tools/               # 资产转换、回放渲染和训练画面
└── outputs/             # 本地实验配置、TensorBoard 日志、权重与视频
```

## 运行 demo

以下命令在 `wheel_leg_rl_parallel/training` 下运行。

```powershell
uv sync --locked
uv run --locked main.py demo --num-envs 4 --seconds 5 --video outputs/wheel_leg_v02.mp4
```

默认用脚本控制器从静止切换到 0.35 m/s 前进命令。终端输出速度、位移、重置次数和双轮接地步数。视频用原始 CAD 网格绘制 Genesis 记录的第一台机器人状态。加 `--viewer` 可在 Genesis 实时查看碰撞几何。初次运行可能需要编译 GPU 内核。

## V0.2 连续腿高训练

以下命令在 `training` 下运行。新任务从空白权重起步，腿部限矩为 ±40 N·m。闭链姿态用髋关节中心到轮轴中心的垂向距离换算，奖励跟踪车身参考点的离地高度。`v02_low` 先在 33 cm 腿高低速训练，前 80 个 PPO 迭代保持 33 cm。之后把采样下限逐步降到第 160 轮的 30 cm、第 280 轮的 25 cm、第 450 轮的 20 cm，以及第 600 轮的 17 cm。新目标仍可采到较高腿位，回合内按腿高变化率连续过渡。恢复训练时沿用 checkpoint 的迭代编号。

`v02_low` 有 45% 的驻车回合，保持初始腿高和零运动指令。回放验收时测零指令下的轮速和车身漂移。

```powershell
uv sync --locked
uv run --locked main.py train --config configs/v02_low.toml --num-envs 128 --iterations 30 --output outputs/v02_low_check
uv run --locked main.py train --config configs/v02_low.toml --num-envs 1024 --iterations 1000 --output outputs/v02_low --live-port 6007
```

回放检查 `33 → 25 → 20 → 17 cm`。也可以加上 `--yaw 0.25` 在转向时检查。原生 Genesis 键盘回放中，按住 `Z/X` 可连续抬高或降低目标。

```powershell
uv run --locked main.py demo --checkpoint outputs/v02_low/model_999.pt --num-envs 1 --speed 0.35 --height-sequence 0.33 0.25 0.20 0.17 --height-hold 2
```

TensorBoard 记录 `Episode/body_height_mae_cm`、`Episode/forward_mae_m_s`、`Episode/fall_rate` 和 `Episode/leg_saturation_rate`。确认低腿位站稳、启停及高度切换后，再逐步提高平地速度，每次从上一档通过验证的权重初始化。`--init-checkpoint` 加载相同观测结构的权重，并重新初始化优化器。

```powershell
uv run --locked main.py train --config configs/v02_speed.toml --init-checkpoint outputs/v02_low/model_999.pt --forward-limit 1.0 --num-envs 1024 --iterations 600 --output outputs/v02_speed_1
```

`v02_speed` 保留 10% 的驻车回合。其余平地回合按 7% 常规随机、15% 直线启停与反向、7% 原地快转、13% 连续转弯、11% 入弯后制动、22% 松键再起步、25% 专项升降采样。速度工况在回合内保持腿高，初始腿高覆盖 17–33 cm。专项升降在静止或 ±0.50 m/s 匀速直行时交替升高、降低。按住时长从 0.26、0.70、1.70 s 中选择，零速指令保持 0.22、0.50 或 1.10 s 后再起步，同时采样同向连按与反向起步。快速转向请求上限为 ±4 rad/s。

平地默认线速度档位为 0.8、1.2、1.6 m/s，超过当前速度上限的采样会截断。扩展高速档时，增加 `command_speed_levels` 并用 `--forward-limit` 设置对应上限。线加减速度上限按回合以 35%、45%、20% 的概率选 1、2、4 m/s²。偏航角加速度上限为 24 rad/s²，轮速和底盘摩擦圆会进一步限制组合指令。`v02_speed` 的高度参考限速 0.30 m/s、限加速度 0.90 m/s²。

回放分别测启停与升降。TensorBoard 记录回合平均误差及 pitch、roll 峰值。

### 从低腿高扩展到高腿高

用低腿高权重或上一版高腿高权重初始化 17–33 cm 的平地训练。高腿高配置的四个腿部驱动仿真限矩均为 ±40 N·m，轮端限矩为 ±5 N·m。保留 ±0.5 m/s 线速度、±0.5 rad/s 偏航角速度和 0.30 m/s 的腿高参考变化上限。训练回合内继续切换高度。目标中 25% 为 17 cm，35% 为 20 cm，10% 为 33 cm，余下 30% 在整段高度连续采样。

```powershell
uv run --locked main.py train --config configs/v02_height_33.toml --init-checkpoint outputs/v02_low/model_999.pt --num-envs 1024 --iterations 600 --output outputs/v02_height_33_40nm
uv run --locked main.py demo --checkpoint outputs/v02_height_33_40nm/model_599.pt --num-envs 1 --speed 0.35 --height-sequence 0.17 0.25 0.33 0.28 0.20 0.17 --height-hold 2
```

回放同时检查行进升降、低位回归、高位保持和力矩饱和。也可用 `--height-sequence 0.17 0.33 0.17` 单独检查跨全行程的高度切换。

华南虎 V14 Rough-v1 的防劈叉项按车体系轮心前后差计罚，超过 5 cm 时追加固定惩罚。动作平滑项包括一阶和二阶动作差分以及关节速度和加速度，腿和轮分别计权。TensorBoard 的 `leg_fore_aft_gap_cm` 记录轮心相对同侧髋中心的前后投影差。

### 华南虎贴地跑场

用 [v02_scut_ground.toml](configs/v02_scut_ground.toml) 训练一个 policy。地形以华南虎 V14 Rough-v1 的跑场形状为参考，包含平地、凸坡、凹坡、随机粗糙面、交叉凸条、连续凸阶和两类凹阶。非平地以 55%、30%、15% 的概率抽取三档难度，从赛道两端出发直行，速度请求为 0.52–0.80 m/s。当前阶段的地形均为贴地通过。

平地占 40%，赛道约 24 m 长，保留驻车、缓起缓停、重复启停、快转和升降。训练腿高覆盖 17–33 cm，回合内连续调整。当前配置将凸条回合提高到 18%，其中 75% 倒车通过，其余非平地按原比例缩放。平地接地时额外惩罚横滚误差和横滚角速度。

```powershell
uv run --locked main.py train --config configs/v02_scut_ground.toml --resume outputs/v02_scut_ground_065_20261123_031406/artifacts/model_499.pt --backend cuda --num-envs 4096 --iterations 500 --output outputs/v02_scut_ground_065_focus --live-port 6013 --live-stream --live-fps 8
uv run --locked main.py demo --checkpoint outputs/v02_scut_ground_065_focus/model_999.pt --surface bars --terrain-variant 3 --terrain-seed 101 --num-envs 1 --seconds 13 --speed -0.70 --height 0.20 --viewer
```

先验证 `scut_ground_scale = 0.65`，再根据结果分阶段提高到 `0.80` 和 `1.00`。每阶段从上一阶段通过验证的检查点初始化，使用独立输出目录以保留配置快照。回放逐地形按 1–3 档、两方向、17/20/33 cm 验证，确认确实穿过障碍，并记录倒地、pitch、roll、速度恢复和限矩。随机路面用不同种子复测。平地单独回归驻车、快转、升降和 1.6 m/s 启停。地形效果稳定后，再单独加入地面摩擦、载荷、关节摩擦和观测噪声的随机化。

`--surface` 可选 `flat`、`slope`、`inv_slope`、`rough`、`bars`、`stairs`、`inv_stairs_low`、`inv_stairs_high`。通过 `--terrain-variant` 选择难度，正负 `--speed` 检查两个方向。TensorBoard 的 `Episode/terrain_progress_m` 记录车体相对出生点的最大纵向位移。

## 旧平地 PPO 短训与回放

```powershell
uv run --locked main.py train --num-envs 256 --iterations 100 --output outputs/ppo_wheel_leg_v02_run1
uv run --locked main.py demo --checkpoint outputs/ppo_wheel_leg_v02_run1/model_99.pt --video outputs/ppo_v02_replay.mp4
```

平地 PPO 使用独立的策略和 critic 观测。Checkpoint 校验机器人型号和任务观测结构。

## 从 checkpoint 继续训练

```powershell
uv run --locked main.py train --resume outputs/ppo_wheel_leg_v02_run1/model_99.pt --iterations 100 --live-port 6007 --live-fps 60
```

续训会恢复策略、critic 和优化器，使用 checkpoint 同目录的 `run_config.json` 还原任务、PPO 设置和启动参数。`--iterations` 指本次追加的轮数。上面的命令运行第 100 至 199 轮，生成 `model_199.pt`。仿真环境会重新初始化。续训的 checkpoint 和 TensorBoard 日志写入新的输出目录，TensorBoard 的迭代编号接续旧 checkpoint。旧 checkpoint 如果缺少配置快照，请用 `--robot`、`--num-envs` 和 `--config` 等参数指定原来的设置。

## 实时训练监控

在本机启动带监控的训练：

```powershell
uv run --locked main.py train --num-envs 64 --iterations 12 --live-port 6007
```

Genesis 自动打开，显示并行环境 0 的机器人和当前地形。跑道按地形分色，高差会加深台阶边缘。打开 [实时训练页](http://127.0.0.1:6007/) 查看同一台机器人的地形、速度、位移、车身离地高度和姿态，目标值与当前值并列显示。目标位移由本回合的速度和转向指令逐步推算。关闭 Genesis 窗口后，训练和网页监控继续运行。`--live-fps` 可调整窗口刷新率，默认 60 FPS。

连续腿高任务用 `base_link` 原点到局部地面的垂向高度计算高度奖励，并记录 `Episode/body_height_mae_cm`。V0.2 的 0.17–0.33 m 髋到轮心指令对应约 0.0975–0.2575 m 的车身离地高度。回放可用 `--body-height 0.13` 或 `--body-height-sequence 0.13 0.22 0.13` 指定离地高度。`--height` 和 `--height-sequence` 使用髋到轮心口径。

本地和远程监控页都提供训练轮次、采样速度、平均回合奖励、平均回合长度和两类 PPO 损失。曲线每轮更新，保留最近 300 轮，悬停可查看该轮数值。

通过 SSH 使用 Windows 训练机时，在本机终端转发端口：

```powershell
ssh -t -L 6007:127.0.0.1:6007 hjcComputer powershell.exe -NoLogo -NoProfile
```

在远端的 `training` 目录里运行：

```powershell
uv run --locked main.py train --num-envs 64 --iterations 12 --live-port 6007 --live-stream
```

然后在本机浏览器打开 [实时训练页](http://127.0.0.1:6007/)。检测到 SSH 会话时，训练会自动采用网页串流。`--live-stream` 也可用于计划任务等无桌面的启动方式。画面由 MuJoCo 根据训练状态离屏绘制，默认目标为 8 FPS。关闭页面后渲染会暂停。训练进程退出时页面服务随之关闭。

## Genesis 窗口驾驶

```powershell
uv run --locked main.py demo --checkpoint outputs/ppo_wheel_leg_v02_run1/model_99.pt --keyboard --speed 2.5
```

在 Genesis 窗口中按住 `W/S` 前进或后退，按住 `A/D` 左转或右转。松键、按 `Space` 或切出窗口时，对应的运动指令归零。关闭窗口即可退出。原生键盘模式的前后目标速度默认上限为 2.5 m/s。实际速度取决于 checkpoint 的跟踪能力。

搭配 `--live-port 6007 --live-telemetry` 可以在网页上查看数据，键盘输入由 Genesis 窗口接收。训练采样带有 PPO 探索噪声，回放使用策略的确定性动作。

## 网页键鼠控制

用相同的 SSH 端口转发，在远端 `training` 目录加载训练好的权重：

```powershell
uv run --locked main.py demo --robot v0.2 --checkpoint outputs/ppo_wheel_leg_v02_run1/model_99.pt --live-port 6007 --keyboard
```

打开 [机器人回放页](http://127.0.0.1:6007/)，点击机器人画面后操作。按住 `W/S` 前进或后退，按住 `A/D` 左转或右转，也可以按住页面上的方向按钮。松键释放对应方向，`Space` 将两个指令归零。浏览器失焦或控制连接断开时，运动指令会归零。拖动画面旋转视角，滚轮缩放，可以点击「复位视角」。训练监控页也可以调整视角。

回放使用 checkpoint 的指令范围，`--speed` 给前后速度设置上限。连续腿高权重可配合 `--height` 或 `--height-sequence` 使用。键盘模式默认持续运行，按 `Ctrl+C` 结束，也可以用 `--seconds` 限定时间。

网页回放的目标为 60 FPS，可用 `--live-fps` 调整。机器人画面标题旁显示渲染帧率。策略仍以 50 Hz 更新，画面在相邻状态之间插值，复位时直接切到新姿态。

## 用 TensorBoard 查看完整曲线

在本机另开一个终端，建立 6006 端口转发：

```powershell
ssh -t -L 6006:127.0.0.1:6006 hjcComputer powershell.exe -NoLogo -NoProfile
```

在远端 PowerShell 启动 TensorBoard：

```powershell
Set-Location 'D:\RL_Stuff\wheel_leg_rl_parallel\training'
uv run --locked tensorboard --logdir outputs --host 127.0.0.1 --port 6006
```

在本机打开 [TensorBoard](http://127.0.0.1:6006/)，在 **Scalars** 中查看 `Train/mean_reward`、`Train/mean_episode_length` 和 `Loss/`。TensorBoard 可以读取已经结束的实验，也可以对比 `outputs` 内的多次训练。

## 调参与实验记录

用 `--config` 指定 TOML 覆盖项，未填写的项沿用 [configs/flat.toml](configs/flat.toml)。可以将以下内容保存为 `slow_flat.toml`，再运行 `uv run --locked main.py train --config slow_flat.toml`。

```toml
[task]
forward_command = [-0.3, 0.3]

[ppo.algorithm]
learning_rate = 0.00005
```

华南虎 V14 Rough-v1 奖励的默认权重、尺度和阈值集中在 `task.scut_rough_reward` 下。要单独调整某个实验，把对应小节和参数写进该实验的 TOML。下例将 `v02_speed` 的地面俯仰二次惩罚权重从 `2.0` 调到 `3.0`。 [默认配置](configs/flat.toml)

```toml
[task.scut_rough_reward.tracking]
pitch_penalty_weight = 3.0
```

每次训练在输出目录中生成 `run_config.json`。其中包含合并后的任务与 PPO 配置，以及实际使用的机器人型号、MJCF 路径与哈希、种子、环境数、迭代数和仿真步长。省略 `--output` 时自动创建带时间戳的目录。手动指定时使用新目录或空目录，每次运行独立保存。

回放时默认使用 checkpoint 同目录的 `run_config.json`。可用 `--config` 显式指定 TOML 或 JSON 快照。复跑某次训练时，用 `--config` 读取当前格式的快照，并按 `run` 字段填写机器人、模型、种子、环境数和迭代数等启动参数。包含已删除 reward 配置的旧快照需要在原版本运行。

## 华南理工参考模型

```powershell
uv run --locked main.py demo --robot scut
uv run --locked convert_wheelbipe.py
```

转换脚本默认读取同一工作区内 `Ref_Git/wheeled-legged_RL` 的原版 USD。`--usd` 可以指定其他路径下的同一资产，`--output` 可以指定输出位置。资产的授权见 [SCUTRobotLab-LICENSE.txt](assets/SCUTRobotLab-LICENSE.txt)。
