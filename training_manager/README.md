# 轮腿训练台

在 4060 主机上管理短训、长训和回放。远端通过 Windows OpenSSH 接收任务，按次启动独立的训练进程。SSH 重连后，主机用任务编号和远端状态文件恢复监控。

## 启动

在 wheel_leg_rl_parallel 仓库根目录运行以下命令。浏览器打开 [轮腿训练台](http://127.0.0.1:8765/)。

~~~powershell
uv run --project training --locked python -m training_manager --ai --auto-followup
~~~

默认连接 SSH 别名 hjcComputer。远端工作目录默认为 D:\RL_Stuff\wheel_leg_rl_parallel\training\outputs\manager-worker。可用 --remote 和 --remote-root 修改主机别名和路径。项目快照安装到该工作目录，现有训练目录仍可按原方式使用。

只用主机时，在启动命令后加 --local-only。两种模式共用主机上的训练记录。自动选设备时，超过 30 轮的训练优先远程，其余任务优先本机。每台机器同时接收一个管理任务。调度前会检查是否已有同项目的 CUDA 训练或回放在运行。

## 提交实验

1. 在新建任务中选择训练或回放、配置和模型。填写并行环境数和训练轮数，选择执行设备。
2. 要先短训再长训，可以分别提交任务，在长训的「等待前置任务」中选择刚提交的短训。短训成功退出后长训才会开始。策略是否通过回放要按实验指标判断。
3. 在任务详情中查看滚动日志、下载产物或提交回放。续训使用原任务快照、模型和 run_config.json，输入的轮数是追加轮数。用已有模型初始化新训练时，可另选兼容配置。

每次提交都将已跟踪和未忽略的新训练文件打包，包含尚未提交的配置、机器人模型和资源。界面记录 Git 提交和整个快照的 SHA-256。模型与配置在提交时复制到输入目录，任务排队期间的工作区修改会用于下一次提交。本机可以复用与快照锁文件匹配且通过 uv 检查的训练环境。其余情况按锁文件建立独立环境。

## 产物与断线

主机的 training/outputs/manager/runs/ 下保存每次任务的输入和同步产物，任务状态记录在 training/outputs/manager/queue.sqlite3。checkpoint 与 run_config.json 一起使用。同步校验 SHA-256，checkpoint 还会检查 PyTorch 保存的 ZIP 内容。训练结束后同步完整标准输出和错误日志。在训练期间，面板显示最近的日志并回收已完成的 checkpoint。

在仓库根目录运行以下命令可对比训练曲线。

~~~powershell
uv run --project training --locked tensorboard --logdir training/outputs/manager/runs --host 127.0.0.1 --port 6006
~~~

远端不可用时，排队中的「自动选择」任务可以使用本机。明确指定远程的任务继续等待。已派发的任务在断线时标记为「失联」，重新连通后核对进程和产物。确认进程已停止后，可以从已同步的 checkpoint 续训。

## AI 排查

加 --ai 时，主机通过已登录的 Codex CLI 诊断失败任务。结束的任务也可在面板上手动排查。Codex 收到任务配置、GPU 状态、日志尾部和可用的 TensorBoard 标量摘要。使用 --auto-followup 后，Codex 建议短训时，管理器最多为一项原始失败任务派发一次复查，上限为 128 个环境、30 轮。诊断有误或需要修改配置时，可在面板上核对并提交新任务。

省略 --ai 时，排查使用本机规则，队列和同步照常工作。远端机器只使用项目已有的训练依赖和 Windows PowerShell。
