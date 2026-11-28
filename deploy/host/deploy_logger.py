"""Asynchronous 3-layer deployment logger: 500 Hz raw/obs + 50 Hz policy + incidents + ML dataset.

Designed so disk writes never block the 500 Hz USB RX thread or the 50 Hz policy loop.
Outputs inside ``directory``:
  - ``meta.json``: policy/calibration/session provenance
  - ``summary.json``: end-of-run statistics and fault counts
  - ``events.log``: human-readable timeline of mode switches, faults, warnings, and incidents
  - ``raw_usb.bin``: lossless raw USB CDC RX byte stream
  - ``obs_500hz.npz``: full-rate 500 Hz arrays for SysID (delay/friction/inertia/bias identification)
  - ``control_50hz.csv``: aligned 50 Hz policy step CSV (comparable to training/tools/sim_logger.py)
  - ``dataset.npz``: ML-ready arrays (obs_policy, obs_history, actions, pd targets, rl_active_mask)
  - ``incidents/incident_XXX.{csv,html}``: auto-captured [t-4s, t+1s] fault snapshots with offline charts
"""

from __future__ import annotations

from collections import deque
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import numpy as np

HOST_DIR = Path(__file__).resolve().parent
DEPLOY_TOOLS_DIR = HOST_DIR.parent / "wheel_leg_deploy" / "Tools"
if str(DEPLOY_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_TOOLS_DIR))

from telemetry_capture import (  # noqa: E402
    CHASSIS_MODE_NAMES,
    MOTOR_NAMES,
    Observation,
    RL_FAULT_ATTITUDE,
    RL_FAULT_INS_INVALID,
    RL_FAULT_MOTOR_OFFLINE,
    RL_FAULT_REMOTE_OFFLINE,
    RL_FAULT_RL_CMD_INVALID,
    RL_FAULT_RL_TIMEOUT,
    RL_FAULT_USB_LOST,
)

from policy_runtime import POLICY_JOINT_NAMES, PolicyStepResult  # noqa: E402

FAULT_BITS = (
    (RL_FAULT_REMOTE_OFFLINE, "REMOTE_OFFLINE"),
    (RL_FAULT_INS_INVALID, "INS_INVALID"),
    (RL_FAULT_MOTOR_OFFLINE, "MOTOR_OFFLINE"),
    (RL_FAULT_ATTITUDE, "ATTITUDE"),
    (RL_FAULT_USB_LOST, "USB_LOST"),
    (RL_FAULT_RL_TIMEOUT, "RL_TIMEOUT"),
    (RL_FAULT_RL_CMD_INVALID, "RL_CMD_INVALID"),
)


def decode_fault_flags(mask: int) -> list[str]:
    return [name for bit, name in FAULT_BITS if mask & bit]


def mode_name(mode_id: int) -> str:
    return CHASSIS_MODE_NAMES[mode_id] if 0 <= mode_id < len(CHASSIS_MODE_NAMES) else f"UNKNOWN({mode_id})"


CONTROL_COLUMNS = (
    "t_s", "step", "obs_seq", "obs_mcu_ms", "cmd_seq", "cmd_tx_ok",
    "obs_age_ms", "infer_ms", "loop_dt_ms",
    "chassis_mode", "rl_flags_rx", "rl_flags_tx", "fault_flags",
    "vx_set", "wz_set", "height_set", "cmd_vx", "cmd_wz", "cmd_height", "height_rate",
    "roll_deg", "pitch_deg", "yaw_deg",
    "wx", "wy", "wz", "gx", "gy", "gz",
    "leg_h_mcu_L", "leg_h_mcu_R", "leg_h_cad_L", "leg_h_cad_R",
    *(f"q_sim_{n}" for n in POLICY_JOINT_NAMES[:4]),
    *(f"q_nom_{n}" for n in POLICY_JOINT_NAMES[:4]),
    *(f"dq_sim_{n}" for n in POLICY_JOINT_NAMES),
    *(f"act_{n}" for n in POLICY_JOINT_NAMES),
    *(f"q_obs_{m}" for m in MOTOR_NAMES),
    *(f"dq_obs_{m}" for m in MOTOR_NAMES),
    *(f"tau_meas_{m}" for m in MOTOR_NAMES),
    *(f"q_ref_{m}" for m in MOTOR_NAMES),
    *(f"v_ref_{m}" for m in MOTOR_NAMES),
    *(f"kp_{m}" for m in MOTOR_NAMES),
    *(f"kd_{m}" for m in MOTOR_NAMES),
    *(f"tau_pd_exp_{m}" for m in MOTOR_NAMES),
    "leg_sat", "wheel_sat",
)
COL = {name: i for i, name in enumerate(CONTROL_COLUMNS)}

INCIDENT_PLOTS = (
    ("姿态 [deg]", ("pitch_deg", "roll_deg")),
    ("机体角速度 [rad/s]", ("wx", "wy", "wz")),
    ("速度与转向指令", ("cmd_vx", "cmd_wz", "wz")),
    ("腿高对比 MCU vs CAD [m]", ("cmd_height", "leg_h_mcu_L", "leg_h_mcu_R", "leg_h_cad_L", "leg_h_cad_R")),
    ("腿关节实测力矩 [N·m]", ("tau_meas_left_phi1", "tau_meas_left_phi4", "tau_meas_right_phi1", "tau_meas_right_phi4")),
    ("腿关节预期 PD 力矩 [N·m]", ("tau_pd_exp_left_phi1", "tau_pd_exp_left_phi4", "tau_pd_exp_right_phi1", "tau_pd_exp_right_phi4")),
    ("轮子力矩 实测 vs 预期 [N·m]", ("tau_meas_left_wheel", "tau_meas_right_wheel", "tau_pd_exp_left_wheel", "tau_pd_exp_right_wheel")),
    ("轮速 目标 vs 实测 [rad/s]", ("v_ref_left_wheel", "dq_obs_left_wheel", "v_ref_right_wheel", "dq_obs_right_wheel")),
    ("策略动作 (raw)", tuple(f"act_{n}" for n in POLICY_JOINT_NAMES)),
    ("通信与实时性 [ms]", ("obs_age_ms", "loop_dt_ms", "infer_ms")),
)


class _Persistence:
    def __init__(self, hold_steps: int, cooldown_steps: int):
        self.hold_steps = max(1, hold_steps)
        self.cooldown_steps = max(1, cooldown_steps)
        self.count = 0
        self.cooldown = 0

    def update(self, active: bool) -> bool:
        self.cooldown = max(0, self.cooldown - 1)
        self.count = self.count + 1 if active else 0
        if self.count == self.hold_steps and self.cooldown == 0:
            self.cooldown = self.cooldown_steps
            return True
        return False


@dataclass(frozen=True)
class ControlLogItem:
    t_s: float
    step: int
    obs: Observation
    result: PolicyStepResult
    cmd_seq: int
    cmd_tx_ok: bool
    obs_age_ms: float
    loop_dt_ms: float


class DeployLogger:
    """Async logger with bounded queue, auto incident capture, and NPZ dataset export."""

    def __init__(
        self,
        directory: Path,
        meta: dict[str, Any],
        *,
        control_dt: float = 0.02,
        leg_torque_limit: float = 40.0,
        wheel_torque_limit: float = 5.0,
        record_raw_bin: bool = True,
        history_seconds: float = 4.0,
        after_seconds: float = 1.0,
        queue_capacity: int = 16384,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "incidents").mkdir(exist_ok=True)

        self.control_dt = float(control_dt)
        self.leg_limit = float(leg_torque_limit)
        self.wheel_limit = float(wheel_torque_limit)
        self.record_raw_bin = bool(record_raw_bin)
        self.t0_ns = time.monotonic_ns()
        self.started_iso = datetime.now().isoformat(timespec="seconds")

        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=queue_capacity)
        self._dropped_items = 0
        self._closed = False
        self._incident_steps = max(2, round(history_seconds / self.control_dt))
        self._after_steps = max(1, round(after_seconds / self.control_dt))
        self._history: deque[tuple[np.ndarray, str]] = deque(maxlen=self._incident_steps)
        self._pending_incidents: list[dict[str, Any]] = []
        self._incident_count = 0
        self._counts: dict[str, int] = {}

        # State trackers inside writer thread
        self._prev_mode: int | None = None
        self._prev_rl_rx: int | None = None
        self._prev_fault: int = 0
        self._rl_active_steps = 0
        self._total_ctrl_steps = 0
        self._obs_500hz_count = 0
        self._infer_ms_list: list[float] = []
        self._loop_dt_list: list[float] = []
        self._obs_age_list: list[float] = []

        steps = lambda sec: max(1, round(sec / self.control_dt))
        cd = steps(1.0)
        self._warnings = {
            "LEG_SATURATION": _Persistence(steps(0.2), cd),
            "WHEEL_SATURATION": _Persistence(steps(0.2), cd),
            "PITCH_LARGE": _Persistence(1, cd),
            "ROLL_LARGE": _Persistence(1, cd),
            "LOOP_OVERRUN": _Persistence(2, steps(2.0)),
            "OBS_STALE": _Persistence(2, steps(2.0)),
            "PD_TORQUE_MISMATCH": _Persistence(steps(0.15), steps(2.0)),
            "LEG_HEIGHT_MISMATCH": _Persistence(steps(0.5), steps(5.0)),
        }

        # In-memory compact lists packed into obs_500hz.npz and dataset.npz on close
        self._obs500_rows: list[tuple] = []
        self._ds_t: list[float] = []
        self._ds_obs_seq: list[int] = []
        self._ds_cmd_seq: list[int] = []
        self._ds_obs_policy: list[np.ndarray] = []
        self._ds_obs_history: list[np.ndarray] = []
        self._ds_actions: list[np.ndarray] = []
        self._ds_q_sim: list[np.ndarray] = []
        self._ds_q_nom: list[np.ndarray] = []
        self._ds_dq_sim: list[np.ndarray] = []
        self._ds_q_ref: list[tuple[float, ...]] = []
        self._ds_v_ref: list[tuple[float, ...]] = []
        self._ds_kp: list[tuple[float, ...]] = []
        self._ds_kd: list[tuple[float, ...]] = []
        self._ds_tau_exp: list[tuple[float, ...]] = []
        self._ds_tau_meas: list[tuple[float, ...]] = []
        self._ds_leg_mcu: list[tuple[float, float]] = []
        self._ds_leg_cad: list[tuple[float, float]] = []
        self._ds_euler: list[tuple[float, float, float]] = []
        self._ds_gyro: list[tuple[float, ...]] = []
        self._ds_grav: list[np.ndarray] = []
        self._ds_cmds: list[tuple[float, float, float, float]] = []
        self._ds_mode: list[int] = []
        self._ds_rl_rx: list[int] = []
        self._ds_rl_tx: list[int] = []
        self._ds_fault: list[int] = []
        self._ds_active: list[bool] = []

        # Open files
        self._raw_file = (self.directory / "raw_usb.bin").open("wb") if self.record_raw_bin else None
        self._csv_file = (self.directory / "control_50hz.csv").open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(CONTROL_COLUMNS + ("mode_name",))
        self._events_file = (self.directory / "events.log").open("w", encoding="utf-8")

        meta_out = {
            "started_at": self.started_iso,
            "control_dt": self.control_dt,
            "leg_torque_limit": self.leg_limit,
            "wheel_torque_limit": self.wheel_limit,
            "control_columns": list(CONTROL_COLUMNS) + ["mode_name"],
            **meta,
        }
        (self.directory / "meta.json").write_text(
            json.dumps(meta_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._write_event(0.0, "START", f"dir={self.directory.name}")

        self._thread = threading.Thread(target=self._writer_loop, name="deploy-logger", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ producer API (non-blocking)
    def _enqueue(self, kind: str, payload: Any) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait((kind, payload))
        except queue.Full:
            self._dropped_items += 1

    def log_raw_bytes(self, data: bytes) -> None:
        if self.record_raw_bin and data:
            self._enqueue("raw", data)

    def log_obs_500hz(self, obs: Observation, lost_before: int) -> None:
        self._enqueue("obs", (obs, lost_before))

    def log_control_step(self, item: ControlLogItem) -> None:
        self._enqueue("ctrl", item)

    def log_event(self, kind: str, message: str, t_s: float | None = None) -> None:
        t = (time.monotonic_ns() - self.t0_ns) * 1e-9 if t_s is None else float(t_s)
        self._enqueue("event", (t, kind, message))

    def close(self, link_stats: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._closed:
            return {}
        self._closed = True
        self._queue.put(("stop", link_stats))
        self._thread.join(timeout=10.0)
        summary_path = self.directory / "summary.json"
        if summary_path.is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        return {}

    # ------------------------------------------------------------------ writer thread
    def _write_event(self, t_s: float, kind: str, message: str) -> None:
        self._counts[kind] = self._counts.get(kind, 0) + 1
        self._events_file.write(f"[{t_s:9.3f}s] {kind:<20} {message}\n")
        self._events_file.flush()

    def _writer_loop(self) -> None:
        last_flush = time.monotonic()
        link_stats = None
        while True:
            try:
                kind, payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                if time.monotonic() - last_flush >= 1.0:
                    self._flush_files()
                    last_flush = time.monotonic()
                continue

            if kind == "raw":
                if self._raw_file is not None:
                    self._raw_file.write(payload)
            elif kind == "obs":
                self._handle_obs_500hz(*payload)
            elif kind == "ctrl":
                self._handle_control_step(payload)
            elif kind == "event":
                self._write_event(*payload)
            elif kind == "stop":
                link_stats = payload
                break

            if time.monotonic() - last_flush >= 1.0:
                self._flush_files()
                last_flush = time.monotonic()

        self._finalize(link_stats)

    def _flush_files(self) -> None:
        self._csv_file.flush()
        self._events_file.flush()
        if self._raw_file is not None:
            self._raw_file.flush()

    def _handle_obs_500hz(self, obs: Observation, lost_before: int) -> None:
        self._obs_500hz_count += 1
        t_rx_s = (obs.received_monotonic_ns - self.t0_ns) * 1e-9
        if lost_before > 0:
            self._write_event(t_rx_s, "PACKET_LOSS", f"lost={lost_before} before seq={obs.seq}")
        self._obs500_rows.append((
            t_rx_s,
            obs.seq,
            obs.timestamp_ms,
            lost_before,
            *obs.gyro,
            *obs.quaternion,
            *(m.position_multi for m in obs.motors),
            *(m.velocity for m in obs.motors),
            *(m.torque_current for m in obs.motors),
            obs.vx_set,
            obs.vy_set,
            obs.wz_set,
            obs.height_set,
            obs.leg_height[0],
            obs.leg_height[1],
            obs.rl_flags,
            obs.chassis_mode,
            obs.fault_flags,
        ))

    def _handle_control_step(self, item: ControlLogItem) -> None:
        self._total_ctrl_steps += 1
        obs = item.obs
        res = item.result
        m_name = mode_name(obs.chassis_mode)

        tau_meas = tuple(float(m.torque_current) for m in obs.motors)
        q_obs = tuple(float(m.position_multi) for m in obs.motors)
        dq_obs = tuple(float(m.velocity) for m in obs.motors)

        leg_sat = float(
            any(abs(tau_meas[i]) >= 0.95 * self.leg_limit or abs(res.tau_pd_expected_fw[i]) >= 0.99 * self.leg_limit for i in range(4))
        )
        wheel_sat = float(
            any(abs(tau_meas[i]) >= 0.95 * self.wheel_limit or abs(res.tau_pd_expected_fw[i]) >= 0.99 * self.wheel_limit for i in range(4, 6))
        )

        row = np.array((
            item.t_s, item.step, obs.seq, obs.timestamp_ms, item.cmd_seq, float(item.cmd_tx_ok),
            item.obs_age_ms, res.infer_ms, item.loop_dt_ms,
            float(obs.chassis_mode), float(obs.rl_flags), float(res.rl_flags_tx), float(obs.fault_flags),
            obs.vx_set, obs.wz_set, obs.height_set, res.cmd_vx, res.cmd_wz, res.cmd_height, res.height_rate,
            res.roll_deg, res.pitch_deg, math.degrees(res.yaw_rad),
            *obs.gyro, *res.gravity,
            res.leg_height_mcu[0], res.leg_height_mcu[1], res.leg_height_cad[0], res.leg_height_cad[1],
            *res.q_sim_unified, *res.q_nominal_unified, *res.dq_sim_unified, *res.actions,
            *q_obs, *dq_obs, *tau_meas,
            *res.q_ref_fw, *res.v_ref_fw, *res.kp_fw, *res.kd_fw, *res.tau_pd_expected_fw,
            leg_sat, wheel_sat,
        ), dtype=np.float64)

        self._csv_writer.writerow([f"{v:.6g}" for v in row] + [m_name])

        # Accumulate dataset arrays
        rl_active = (obs.rl_flags == 1) and (res.rl_flags_tx == 1) and (obs.fault_flags == 0)
        if rl_active:
            self._rl_active_steps += 1
        self._infer_ms_list.append(res.infer_ms)
        self._loop_dt_list.append(item.loop_dt_ms)
        self._obs_age_list.append(item.obs_age_ms)

        self._ds_t.append(item.t_s)
        self._ds_obs_seq.append(obs.seq)
        self._ds_cmd_seq.append(item.cmd_seq)
        self._ds_obs_policy.append(res.obs_policy.copy())
        self._ds_obs_history.append(res.obs_history.copy())
        self._ds_actions.append(res.actions.copy())
        self._ds_q_sim.append(res.q_sim_unified.astype(np.float32))
        self._ds_q_nom.append(res.q_nominal_unified.astype(np.float32))
        self._ds_dq_sim.append(res.dq_sim_unified.astype(np.float32))
        self._ds_q_ref.append(res.q_ref_fw)
        self._ds_v_ref.append(res.v_ref_fw)
        self._ds_kp.append(res.kp_fw)
        self._ds_kd.append(res.kd_fw)
        self._ds_tau_exp.append(res.tau_pd_expected_fw)
        self._ds_tau_meas.append(tau_meas)
        self._ds_leg_mcu.append(res.leg_height_mcu)
        self._ds_leg_cad.append(res.leg_height_cad)
        self._ds_euler.append((res.roll_deg, res.pitch_deg, math.degrees(res.yaw_rad)))
        self._ds_gyro.append(obs.gyro)
        self._ds_grav.append(res.gravity.astype(np.float32))
        self._ds_cmds.append((res.cmd_vx, res.cmd_wz, res.cmd_height, res.height_rate))
        self._ds_mode.append(obs.chassis_mode)
        self._ds_rl_rx.append(obs.rl_flags)
        self._ds_rl_tx.append(res.rl_flags_tx)
        self._ds_fault.append(obs.fault_flags)
        self._ds_active.append(rl_active)

        # Transition & fault detection
        if self._prev_mode is None or obs.chassis_mode != self._prev_mode:
            prev_str = "INIT" if self._prev_mode is None else mode_name(self._prev_mode)
            self._write_event(item.t_s, "MODE", f"{prev_str} -> {m_name} (rl_rx={obs.rl_flags}, rl_tx={res.rl_flags_tx})")
        if self._prev_rl_rx is not None and obs.rl_flags != self._prev_rl_rx:
            self._write_event(item.t_s, "RL_HANDOVER", f"rl_flags_rx {self._prev_rl_rx} -> {obs.rl_flags} in {m_name}")

        new_faults = obs.fault_flags & ~self._prev_fault
        if new_faults:
            names = decode_fault_flags(new_faults)
            self._write_event(item.t_s, "FAULT", f"0x{obs.fault_flags:04X} (+{','.join(names)}) mode={m_name}")
            if self._prev_rl_rx == 1 or obs.rl_flags == 1:
                self._start_incident(item.t_s, row, m_name, [f"下位机触发安全故障: {', '.join(names)}"])
        elif self._prev_rl_rx == 1 and obs.rl_flags == 0 and obs.fault_flags == 0 and m_name != "IDLE":
            self._start_incident(item.t_s, row, m_name, [f"RL 运行中意外退出至 {m_name}"])

        self._check_warnings(item, row, tau_meas, res)

        self._prev_mode = obs.chassis_mode
        self._prev_rl_rx = obs.rl_flags
        self._prev_fault = obs.fault_flags

        self._history.append((row, m_name))
        for inc in self._pending_incidents:
            inc["after"].append((row, m_name))
        self._finish_incidents(force=False)

    def _check_warnings(
        self,
        item: ControlLogItem,
        row: np.ndarray,
        tau_meas: tuple[float, ...],
        res: PolicyStepResult,
    ) -> None:
        active_tx = (res.rl_flags_tx == 1)
        leg_gap = max(abs(res.tau_pd_expected_fw[i] - tau_meas[i]) for i in range(4))
        wheel_gap = max(abs(res.tau_pd_expected_fw[i] - tau_meas[i]) for i in range(4, 6))
        cad_ok = math.isfinite(res.leg_height_cad[0]) and math.isfinite(res.leg_height_cad[1])
        h_gap = (
            max(abs(res.leg_height_mcu[0] - res.leg_height_cad[0]), abs(res.leg_height_mcu[1] - res.leg_height_cad[1]))
            if cad_ok
            else 0.0
        )

        checks = {
            "LEG_SATURATION": (
                active_tx and row[COL["leg_sat"]] > 0.5,
                lambda: f"腿部力矩接近饱和 tau_meas={[round(v, 1) for v in tau_meas[:4]]}",
            ),
            "WHEEL_SATURATION": (
                active_tx and row[COL["wheel_sat"]] > 0.5,
                lambda: f"轮子力矩接近饱和 tau_meas={[round(v, 2) for v in tau_meas[4:6]]}",
            ),
            "PITCH_LARGE": (
                item.obs.rl_flags == 1 and abs(res.pitch_deg) > 20.0,
                lambda: f"俯仰角过大 pitch={res.pitch_deg:+.1f} deg",
            ),
            "ROLL_LARGE": (
                item.obs.rl_flags == 1 and abs(res.roll_deg) > 15.0,
                lambda: f"横滚角过大 roll={res.roll_deg:+.1f} deg",
            ),
            "LOOP_OVERRUN": (
                item.loop_dt_ms > 25.0,
                lambda: f"50Hz 控制周期超时 loop_dt={item.loop_dt_ms:.1f} ms (infer={res.infer_ms:.2f} ms)",
            ),
            "OBS_STALE": (
                item.obs_age_ms > 30.0,
                lambda: f"观测数据陈旧 obs_age={item.obs_age_ms:.1f} ms",
            ),
            "PD_TORQUE_MISMATCH": (
                active_tx and (leg_gap > 8.0 or wheel_gap > 2.0),
                lambda: f"预期PD力矩与实测偏差大 leg_gap={leg_gap:.1f}N·m wheel_gap={wheel_gap:.2f}N·m (检查方向/CAN/堵转)",
            ),
            "LEG_HEIGHT_MISMATCH": (
                cad_ok and h_gap > 0.04,
                lambda: f"MCU VMC 腿高 {res.leg_height_mcu} 与 CAD 腿高 {res.leg_height_cad} 差 {h_gap*100:.1f} cm (核对零位标定)",
            ),
        }
        for kind, (active, msg_fn) in checks.items():
            if self._warnings[kind].update(bool(active)):
                self._write_event(item.t_s, kind, msg_fn())

    def _likely_causes(self, window: list[tuple[np.ndarray, str]], trigger_row: np.ndarray) -> list[str]:
        data = np.stack([r for r, _ in window]) if window else trigger_row[None]
        c = lambda name: data[:, COL[name]]
        causes: list[str] = []
        fault_mask = int(trigger_row[COL["fault_flags"]])
        if fault_mask & RL_FAULT_RL_TIMEOUT:
            causes.append(f"下位机报 RL_TIMEOUT (最大 obs_age={np.nanmax(c('obs_age_ms')):.1f} ms, 最大 loop_dt={np.nanmax(c('loop_dt_ms')):.1f} ms)")
        if fault_mask & RL_FAULT_ATTITUDE:
            causes.append(f"姿态超限保护 (窗口峰值 |pitch|={np.nanmax(np.abs(c('pitch_deg'))):.1f}°, |roll|={np.nanmax(np.abs(c('roll_deg'))):.1f}°)")
        if fault_mask & RL_FAULT_MOTOR_OFFLINE:
            causes.append("存在底盘电机掉线或 CAN 报错 (RL_FAULT_MOTOR_OFFLINE)")
        if fault_mask & RL_FAULT_USB_LOST:
            causes.append("USB CDC 物理连接断开 (RL_FAULT_USB_LOST)")
        if np.nanmean(c("leg_sat")) > 0.25:
            causes.append(f"腿关节力矩饱和占比 {100*np.nanmean(c('leg_sat')):.0f}% (限值 {self.leg_limit:g} N·m)")
        if np.nanmean(c("wheel_sat")) > 0.25:
            causes.append(f"驱动轮力矩饱和占比 {100*np.nanmean(c('wheel_sat')):.0f}% (限值 {self.wheel_limit:g} N·m)")
        tau_err_leg = np.max(np.abs(np.stack([
            c(f"tau_pd_exp_{m}") - c(f"tau_meas_{m}") for m in MOTOR_NAMES[:4]
        ], axis=1)))
        if tau_err_leg > 10.0:
            causes.append(f"腿部预期 PD 力矩与电机实测力矩最大相差 {tau_err_leg:.1f} N·m，需核对电机零位/方向/供电限流")
        if np.nanmax(c("loop_dt_ms")) > 30.0:
            causes.append(f"上位机 50 Hz 控制循环出现卡顿，最大周期 {np.nanmax(c('loop_dt_ms')):.1f} ms")
        return causes or ["未匹配到单一启发式原因，请查看下方波形与 CSV"]

    def _start_incident(self, t_s: float, row: np.ndarray, m_name: str, reasons: list[str]) -> None:
        self._incident_count += 1
        name = f"incident_{self._incident_count:03d}"
        window = list(self._history)
        causes = self._likely_causes(window, row)
        self._write_event(t_s, "INCIDENT", f"#{self._incident_count} {'; '.join(reasons)} -> incidents/{name}.html")
        self._pending_incidents.append({
            "name": name,
            "t": t_s,
            "reasons": reasons,
            "causes": causes,
            "before": window + [(row, m_name)],
            "after": [],
        })

    def _finish_incidents(self, *, force: bool) -> None:
        keep = []
        for inc in self._pending_incidents:
            if force or len(inc["after"]) >= self._after_steps:
                self._write_incident(inc)
            else:
                keep.append(inc)
        self._pending_incidents = keep

    def _write_incident(self, inc: dict[str, Any]) -> None:
        rows = inc["before"] + inc["after"]
        base = self.directory / "incidents" / inc["name"]
        with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(CONTROL_COLUMNS + ("mode_name",))
            for r, m in rows:
                w.writerow([f"{v:.6g}" for v in r] + [m])
        series = {
            name: [None if not np.isfinite(r[COL[name]]) else round(float(r[COL[name]]), 5) for r, _ in rows]
            for name in CONTROL_COLUMNS
        }
        payload = {
            "t": series["t_s"],
            "series": series,
            "plots": INCIDENT_PLOTS,
            "event_t": inc["t"],
            "title": f"{inc['name']}  t={inc['t']:.2f}s",
            "reasons": inc["reasons"],
            "causes": inc["causes"],
            "limits": {"leg": self.leg_limit, "wheel": self.wheel_limit},
        }
        base.with_suffix(".html").write_text(
            _INCIDENT_HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False)),
            encoding="utf-8",
        )

    def _save_npz_files(self) -> None:
        if self._obs500_rows:
            arr = np.asarray(self._obs500_rows, dtype=np.float64)
            np.savez_compressed(
                self.directory / "obs_500hz.npz",
                t_rx_s=arr[:, 0].astype(np.float64),
                seq=arr[:, 1].astype(np.uint32),
                mcu_ms=arr[:, 2].astype(np.uint32),
                lost_before=arr[:, 3].astype(np.int32),
                gyro=arr[:, 4:7].astype(np.float32),
                quat=arr[:, 7:11].astype(np.float32),
                motor_pos=arr[:, 11:17].astype(np.float32),
                motor_vel=arr[:, 17:23].astype(np.float32),
                motor_tau=arr[:, 23:29].astype(np.float32),
                setpoints=arr[:, 29:33].astype(np.float32),  # vx, vy, wz, height
                leg_height=arr[:, 33:35].astype(np.float32),
                rl_flags=arr[:, 35].astype(np.uint8),
                chassis_mode=arr[:, 36].astype(np.uint8),
                fault_flags=arr[:, 37].astype(np.uint16),
            )

        if self._ds_t:
            np.savez_compressed(
                self.directory / "dataset.npz",
                t_s=np.asarray(self._ds_t, dtype=np.float64),
                obs_seq=np.asarray(self._ds_obs_seq, dtype=np.uint32),
                cmd_seq=np.asarray(self._ds_cmd_seq, dtype=np.uint32),
                obs_policy=np.stack(self._ds_obs_policy).astype(np.float32),
                obs_history=np.stack(self._ds_obs_history).astype(np.float32),
                actions=np.stack(self._ds_actions).astype(np.float32),
                q_sim_unified=np.stack(self._ds_q_sim).astype(np.float32),
                q_nominal_unified=np.stack(self._ds_q_nom).astype(np.float32),
                dq_sim_unified=np.stack(self._ds_dq_sim).astype(np.float32),
                q_ref_fw=np.asarray(self._ds_q_ref, dtype=np.float32),
                v_ref_fw=np.asarray(self._ds_v_ref, dtype=np.float32),
                kp_fw=np.asarray(self._ds_kp, dtype=np.float32),
                kd_fw=np.asarray(self._ds_kd, dtype=np.float32),
                tau_pd_expected_fw=np.asarray(self._ds_tau_exp, dtype=np.float32),
                tau_meas_fw=np.asarray(self._ds_tau_meas, dtype=np.float32),
                leg_height_mcu=np.asarray(self._ds_leg_mcu, dtype=np.float32),
                leg_height_cad=np.asarray(self._ds_leg_cad, dtype=np.float32),
                euler_deg=np.asarray(self._ds_euler, dtype=np.float32),
                gyro=np.asarray(self._ds_gyro, dtype=np.float32),
                gravity=np.stack(self._ds_grav).astype(np.float32),
                commands=np.asarray(self._ds_cmds, dtype=np.float32),
                chassis_mode=np.asarray(self._ds_mode, dtype=np.uint8),
                rl_flags_rx=np.asarray(self._ds_rl_rx, dtype=np.uint8),
                rl_flags_tx=np.asarray(self._ds_rl_tx, dtype=np.uint8),
                fault_flags=np.asarray(self._ds_fault, dtype=np.uint16),
                rl_active_mask=np.asarray(self._ds_active, dtype=np.bool_),
            )

    def _finalize(self, link_stats: dict[str, Any] | None) -> None:
        self._finish_incidents(force=True)
        self._save_npz_files()
        elapsed = (time.monotonic_ns() - self.t0_ns) * 1e-9

        def _pct(vals: list[float]) -> dict[str, float]:
            if not vals:
                return {"mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
            a = np.asarray(vals, dtype=np.float64)
            return {
                "mean": round(float(np.mean(a)), 3),
                "p95": round(float(np.percentile(a, 95)), 3),
                "p99": round(float(np.percentile(a, 99)), 3),
                "max": round(float(np.max(a)), 3),
            }

        summary = {
            "directory": str(self.directory),
            "elapsed_seconds": round(elapsed, 3),
            "obs_500hz_frames": self._obs_500hz_count,
            "control_50hz_steps": self._total_ctrl_steps,
            "rl_active_steps": self._rl_active_steps,
            "rl_active_seconds": round(self._rl_active_steps * self.control_dt, 3),
            "incidents": self._incident_count,
            "dropped_log_items": self._dropped_items,
            "infer_ms": _pct(self._infer_ms_list),
            "loop_dt_ms": _pct(self._loop_dt_list),
            "obs_age_ms": _pct(self._obs_age_list),
            "event_counts": self._counts,
            "link_stats": link_stats or {},
        }
        self._write_event(elapsed, "STOP", json.dumps(summary, ensure_ascii=False))
        (self.directory / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._flush_files()
        self._csv_file.close()
        self._events_file.close()
        if self._raw_file is not None:
            self._raw_file.close()


_INCIDENT_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>Deploy Incident</title>
<style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:16px;background:#111;color:#ddd}
h1{font-size:18px;margin-bottom:6px} ul{margin:4px 0 12px}
.plot{margin:8px 0} canvas{background:#1b1b1b;width:100%;height:175px;border-radius:4px}
.legend span{margin-right:12px;font-size:12px}
</style></head><body>
<h1 id="title"></h1><div>触发事件: <span id="reasons"></span></div><div>自动根因分析:</div><ul id="causes"></ul>
<div id="plots"></div>
<script>
const D = __DATA__;
const colors = ["#4fc3f7","#ffb74d","#81c784","#e57373","#ba68c8","#fff176"];
document.getElementById("title").textContent = D.title;
document.getElementById("reasons").textContent = D.reasons.join("; ");
const causesEl = document.getElementById("causes");
for (const c of D.causes) { const li = document.createElement("li"); li.textContent = c; causesEl.appendChild(li); }
const t = D.t;
const plotsEl = document.getElementById("plots");
for (const [label, names] of D.plots) {
  const div = document.createElement("div"); div.className = "plot";
  const legend = document.createElement("div"); legend.className = "legend";
  legend.innerHTML = "<b>" + label + "</b> " + names.map((n, i) => `<span style="color:${colors[i % colors.length]}">■ ${n}</span>`).join("");
  const canvas = document.createElement("canvas"); div.append(legend, canvas); plotsEl.appendChild(div);
  requestAnimationFrame(() => draw(canvas, names, label));
}
function draw(canvas, names, label) {
  const w = canvas.width = canvas.clientWidth * devicePixelRatio, h = canvas.height = canvas.clientHeight * devicePixelRatio;
  const g = canvas.getContext("2d"); const pad = 46 * devicePixelRatio;
  let lo = Infinity, hi = -Infinity;
  for (const n of names) for (const v of D.series[n]) if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const lim = label.startsWith("腿关节") ? D.limits.leg : label.startsWith("轮子力矩") ? D.limits.wheel : null;
  if (lim) { lo = Math.min(lo, -lim * 1.05); hi = Math.max(hi, lim * 1.05); }
  if (!isFinite(lo)) { lo = -1; hi = 1; } if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
  const x = v => pad + (v - t[0]) / Math.max(t[t.length - 1] - t[0], 1e-6) * (w - pad * 1.2);
  const y = v => h - 14 - (v - lo) / (hi - lo) * (h - 28);
  g.strokeStyle = "#333"; g.fillStyle = "#888"; g.font = `${11 * devicePixelRatio}px sans-serif`;
  for (const v of [lo, (lo + hi) / 2, hi]) { g.beginPath(); g.moveTo(pad, y(v)); g.lineTo(w, y(v)); g.stroke(); g.fillText(v.toFixed(2), 2, y(v) + 4); }
  if (lim) { g.strokeStyle = "#a33"; g.setLineDash([6, 4]); for (const v of [lim, -lim]) { g.beginPath(); g.moveTo(pad, y(v)); g.lineTo(w, y(v)); g.stroke(); } g.setLineDash([]); }
  g.strokeStyle = "#f44"; g.lineWidth = 1.5 * devicePixelRatio; g.beginPath(); g.moveTo(x(D.event_t), 0); g.lineTo(x(D.event_t), h); g.stroke();
  names.forEach((n, i) => {
    g.strokeStyle = colors[i % colors.length]; g.lineWidth = 1.5 * devicePixelRatio; g.beginPath(); let pen = false;
    D.series[n].forEach((v, k) => { if (v === null) { pen = false; return; } pen ? g.lineTo(x(t[k]), y(v)) : g.moveTo(x(t[k]), y(v)); pen = true; });
    g.stroke();
  });
  g.fillStyle = "#888"; g.fillText(`t ${t[0].toFixed(2)}s`, pad, h - 2); g.fillText(`${t[t.length - 1].toFixed(2)}s`, w - 58 * devicePixelRatio, h - 2);
}
</script></body></html>
"""
