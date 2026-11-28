# V0.2 轮腿上位机部署与日志回流工具 (`deploy/host`)

负责在 PC / 机载电脑上运行 **V0.2 `WheelLegLocomotionEnv`** RL 策略，通过 **USB CDC** 与 STM32H7 底盘（500 Hz 观测上行 / 500 Hz 单片机本地纯力矩 PD）通讯，并自动沉淀用于**故障排查**与**下一轮训练 / SysID** 的结构化日志。

---

## 目录结构

```text
deploy/host/
├── configs/joint_calibration.json   # 电机索引、方向、零位偏置、PD增益标定配置
├── export_policy.py                 # 导出训练 checkpoint -> 轻量部署包 (policy.npz + policy_meta.json)
├── policy_runtime.py                # 纯 NumPy 推理 (<0.1ms)、28D×5 观测构造、动作解码与预期 PD
├── usb_link.py                      # 500 Hz USB CDC 接收线程 + 50 Hz 指令发送（复用 telemetry_capture.py）
├── deploy_logger.py                 # 异步三层日志 (raw_usb.bin / obs_500hz.npz / control_50hz.csv / dataset.npz / incidents/*.html)
├── rl_host_runner.py                # 上位机主入口（模式 2 无缝交接心跳、--dry-run、--gain-scale）
├── analyze_deploy_log.py            # 离线体检、28D OOD 核对、轮速滞后/力矩残差 SysID、新模型影子回放
├── tests/test_host_loop.py          # 无硬件闭环回归测试
├── bundles/                         # 导出的模型部署包（gitignore）
└── logs/                            # 每次上车自动生成的运行日志目录（gitignore）
```

---

## 快速上手（安全三步上车）

> 推荐使用 `deploy\wheel_leg_deploy\.venv\Scripts\python.exe`（仅需 `numpy` + `pyserial`）；导出时使用 `training\.venv\Scripts\python.exe` 可额外自动执行 **PyTorch vs NumPy 精度校验**。

### 1. 导出策略部署包
```powershell
python deploy\host\export_policy.py `
  --checkpoint training\outputs\v02_scut_stage2_scut_aligned\model_2199.pt
```
默认输出到 `deploy\host\bundles\v02_scut_stage2_scut_aligned_model_2199\`。

### 2. 阶段一：影子模式核对传感器与标定（强烈建议首次必跑）
```powershell
python deploy\host\rl_host_runner.py `
  --bundle deploy\host\bundles\v02_scut_stage2_scut_aligned_model_2199 `
  --port COM8 `
  --dry-run --tag cal_check
```
- `--dry-run` 会跑满 50 Hz 推理与日志记录，但**永远向单片机发送全零指令**。
- 结束后运行离线分析核对 `gravity`、4 腿零位误差以及 `MCU VMC vs V0.2 CAD` 腿高差异：
```powershell
python deploy\host\analyze_deploy_log.py deploy\host\logs\<timestamp>_cal_check
```
浏览器直接打开生成的 `report.html`，根据 `leg_height_comparison` 与 `ood_alerts` 调整 `configs/joint_calibration.json` 中的 `policy_to_firmware_index`、`joint_sign` 和 `leg_zero_offset_rad`。

### 3. 阶段二：吊装低增益联调（模式 2 本地起身 -> RL 接管）
```powershell
python deploy\host\rl_host_runner.py `
  --bundle deploy\host\bundles\v02_scut_stage2_scut_aligned_model_2199 `
  --port COM8 `
  --gain-scale 0.3 --tag rack_30pct
```
下位机在 `IDLE/RESCUE/GETUP` 时上位机会持续发送 50 Hz 零力心跳；进入 `CHASSIS_NORMAL` 后自动填满 5 帧历史无缝接管。Ctrl+C 退出时会自动连续发送 5 帧全零力矩包。

### 4. 阶段三：实车运行、故障排查与下一轮训练回流
```powershell
# 正常全增益运行
python deploy\host\rl_host_runner.py `
  --bundle deploy\host\bundles\v02_scut_stage2_scut_aligned_model_2199 `
  --port COM8 --tag field_run

# 生成体检报告 + SysID 建议，并用新训练的模型做离线影子回放对比
python deploy\host\analyze_deploy_log.py deploy\host\logs\<timestamp>_field_run `
  --replay-bundle deploy\host\bundles\<new_bundle>
```

### 日志文件与下一轮训练接口
- **查问题**：查看 `events.log`，并在浏览器打开 `report.html` 与 `incidents/incident_XXX.html`。
- **给下一轮训练 / SysID**：
  - `dataset.npz`：`obs_policy (T, 28)`、`obs_history (T, 140)`、`actions (T, 6)`、`q_ref_fw`、`v_ref_fw`、`tau_pd_expected_fw`、`tau_meas_fw`、`rl_active_mask (T,)`。
  - `obs_500hz.npz`：500 Hz 原始高频序列，用于标定电机延迟（`wheel_target_delay_steps`）、摩擦、阻尼与 `leg_bias`。
  - `raw_usb.bin`：原始串口字节流，随时可重新解析。

---

## 主机端无硬件闭环测试
```powershell
python deploy\host\tests\test_host_loop.py
```
