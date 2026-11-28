#!/usr/bin/env python3
"""No-hardware closed-loop verification suite for deploy/host.

Tests:
  1. Exporting real V0.2 checkpoint -> policy.npz + policy_meta.json (pure-NumPy CAD FK & stance checks).
  2. Observation math: quaternion->gravity/euler, 5-frame history fill/shift, joint mapping & PD parity.
  3. Full closed-loop mock STM32H7 (500 Hz 142B RX <-> 50 Hz 140B TX):
     - Mode 2 GETUP -> NORMAL handover with zero-torque heartbeats during GETUP
     - Active RL PD command CRC16 / gain / clamping validation
     - Fault injection -> auto incident CSV + offline HTML generation
     - Shadow (--dry-run) mode guarantees all-zero TX while still logging policy actions
  4. Log artifacts integrity (raw_usb.bin, obs_500hz.npz, control_50hz.csv, dataset.npz)
     + analyze_deploy_log.py report.html & --replay-bundle exact parity.
"""

from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import threading
import time

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
HOST_DIR = TEST_DIR.parent
REPO_ROOT = HOST_DIR.parents[1]
DEPLOY_TOOLS_DIR = HOST_DIR.parent / "wheel_leg_deploy" / "Tools"
for p in (HOST_DIR, DEPLOY_TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from telemetry_capture import (  # noqa: E402
    FRAME,
    FrameStream,
    HEADER,
    MotorState,
    Observation,
    RL_COMMAND,
    RL_COMMAND_SIZE,
    RL_COMMAND_SOF,
    RL_FAULT_ATTITUDE,
    RL_FLAG_COMPUTED_TORQUE,
    RL_FLAG_ZERO_TORQUE,
    SOF,
    crc16,
)

from analyze_deploy_log import analyze_run  # noqa: E402
from deploy_logger import DeployLogger  # noqa: E402
from export_policy import HOME_JOINTS, LEG_SIGNS, export_bundle  # noqa: E402
from policy_runtime import (  # noqa: E402
    CalibrationConfig,
    PolicyBundle,
    WheelLegPolicyController,
    quat_to_euler_deg_and_yaw,
    quat_to_gravity_body,
    wrap_to_pi,
)
from rl_host_runner import run_host_loop  # noqa: E402

DEFAULT_CKPT = REPO_ROOT / "training" / "outputs" / "v02_scut_stage2_scut_aligned" / "model_2199.pt"


def pack_obs_packet(
    seq: int,
    mcu_ms: int,
    gyro: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    motors: list[tuple[float, float, float]],  # (vel, pos, tau) x 6
    vx_set: float,
    vy_set: float,
    wz_set: float,
    height_set: float,
    leg_height: tuple[float, float],
    rl_flags: int,
    chassis_mode: int,
    fault_flags: int,
) -> bytes:
    flat_motors = [v for m in motors for v in m]
    payload = FRAME.pack(
        seq,
        mcu_ms,
        *gyro,
        *quat,
        *flat_motors,
        vx_set,
        vy_set,
        wz_set,
        height_set,
        leg_height[0],
        leg_height[1],
        rl_flags,
        chassis_mode,
        fault_flags,
    )
    body = HEADER.pack(SOF, len(payload)) + payload
    return body + struct.pack("<H", crc16(body))


class MockStm32Transport:
    """In-memory mock STM32H7 streaming 142B obs at 500 Hz and validating 140B RL PD commands."""

    def __init__(self, bundle: PolicyBundle, *, inject_fault_at_s: float | None = 0.80, always_rl: bool = False):
        self.bundle = bundle
        self.inject_fault_at_s = inject_fault_at_s
        self.always_rl = always_rl
        self._rx_buf = bytearray()
        self._rx_lock = threading.Lock()
        self._rx_cond = threading.Condition(self._rx_lock)
        self._stop = threading.Event()
        self.tx_commands: list[tuple] = []
        self.tx_bad_crc = 0
        self.tx_bad_len = 0
        self._last_cmd_kp = [0.0] * 6
        self._last_cmd_kd = [0.0] * 6
        self._last_cmd_qref = [0.0] * 6
        self._last_cmd_vref = [0.0] * 6
        self._wheel_vel_hist: deque[tuple[float, float]] = deque([(0.0, 0.0)] * 5, maxlen=5)
        self._wheel_vel = [0.0, 0.0]
        self._thread = threading.Thread(target=self._sim_loop, name="mock-stm32", daemon=True)
        self._thread.start()

    def _sim_loop(self) -> None:
        t0 = time.perf_counter()
        seq = 0
        nom_home = self.bundle.nominal_legs(0.20)
        # Firmware order [left_phi1, left_phi4, right_phi1, right_phi4] from policy [hip_L, hip_R, hipknee_L, hipknee_R]
        q_fw = [float(nom_home[0]), float(nom_home[2]), float(nom_home[1]), float(nom_home[3]), 0.0, 0.0]
        dq_fw = [0.0] * 6
        while not self._stop.is_set():
            elapsed = time.perf_counter() - t0
            if self.always_rl:
                mode, rl_flags, fault = 3, RL_FLAG_COMPUTED_TORQUE, 0
            elif elapsed < 0.22:
                mode, rl_flags, fault = 2, RL_FLAG_ZERO_TORQUE, 0  # GETUP
            elif self.inject_fault_at_s is not None and elapsed >= self.inject_fault_at_s:
                mode, rl_flags, fault = 3, RL_FLAG_ZERO_TORQUE, RL_FAULT_ATTITUDE
            else:
                mode, rl_flags, fault = 3, RL_FLAG_COMPUTED_TORQUE, 0  # NORMAL

            # Simulate slight pitch oscillation and wheel response with 10 ms lag (5 ticks)
            pitch_rad = 0.05 * math.sin(6.0 * elapsed) if fault == 0 else math.radians(48.0)
            quat = (math.cos(0.5 * pitch_rad), 0.0, math.sin(0.5 * pitch_rad), 0.0)
            gyro = (0.0, 0.3 * math.cos(6.0 * elapsed), 0.0)

            with self._rx_lock:
                kp = list(self._last_cmd_kp)
                kd = list(self._last_cmd_kd)
                qref = list(self._last_cmd_qref)
                vref = list(self._last_cmd_vref)

            # Delayed wheel tracking
            self._wheel_vel_hist.append((vref[4], vref[5]))
            v_delayed = self._wheel_vel_hist[0]
            if rl_flags == RL_FLAG_COMPUTED_TORQUE:
                for i in range(4):
                    q_fw[i] += 0.15 * float(wrap_to_pi(qref[i] - q_fw[i]))
                self._wheel_vel[0] += 0.25 * (v_delayed[0] - self._wheel_vel[0])
                self._wheel_vel[1] += 0.25 * (v_delayed[1] - self._wheel_vel[1])
            else:
                self._wheel_vel[0] *= 0.9
                self._wheel_vel[1] *= 0.9
            dq_fw[4], dq_fw[5] = self._wheel_vel[0], self._wheel_vel[1]

            # Mirror MCU pure-torque PD for realistic feedback torques
            tau_fw = [0.0] * 6
            if rl_flags == RL_FLAG_COMPUTED_TORQUE:
                for i in range(4):
                    tau_fw[i] = float(np.clip(kp[i] * float(wrap_to_pi(qref[i] - q_fw[i])) - kd[i] * dq_fw[i], -40.0, 40.0))
                for i in range(4, 6):
                    tau_fw[i] = float(np.clip(kd[i] * (vref[i] - dq_fw[i]), -5.0, 5.0))

            motors = [(dq_fw[i], q_fw[i], tau_fw[i]) for i in range(6)]
            vx_cmd = 0.5 if mode == 3 and fault == 0 else 0.0
            pkt = pack_obs_packet(
                seq=seq,
                mcu_ms=int(elapsed * 1000.0),
                gyro=gyro,
                quat=quat,
                motors=motors,
                vx_set=vx_cmd,
                vy_set=0.0,
                wz_set=0.0,
                height_set=0.22 if elapsed > 0.45 else 0.20,
                leg_height=(0.20, 0.20),
                rl_flags=rl_flags,
                chassis_mode=mode,
                fault_flags=fault,
            )
            seq += 1
            with self._rx_cond:
                self._rx_buf.extend(pkt)
                self._rx_cond.notify_all()
            time.sleep(0.002)

    @property
    def in_waiting(self) -> int:
        with self._rx_lock:
            return len(self._rx_buf)

    def read(self, size: int) -> bytes:
        with self._rx_cond:
            if not self._rx_buf:
                self._rx_cond.wait(timeout=0.01)
            if not self._rx_buf:
                return b""
            n = min(size, len(self._rx_buf))
            out = bytes(self._rx_buf[:n])
            del self._rx_buf[:n]
            return out

    def write(self, data: bytes) -> int:
        if len(data) != RL_COMMAND_SIZE or not data.startswith(RL_COMMAND_SOF):
            self.tx_bad_len += 1
            return len(data)
        expected_crc = struct.unpack_from("<H", data, RL_COMMAND_SIZE - 2)[0]
        if crc16(data[:-2]) != expected_crc:
            self.tx_bad_crc += 1
            return len(data)
        unpacked = RL_COMMAND.unpack_from(data, 2)
        self.tx_commands.append(unpacked)
        kp = unpacked[10:16]
        kd = unpacked[16:22]
        qref = unpacked[22:28]
        vref = unpacked[28:34]
        with self._rx_lock:
            self._last_cmd_kp = list(kp)
            self._last_cmd_kd = list(kd)
            self._last_cmd_qref = list(qref)
            self._last_cmd_vref = list(vref)
        return len(data)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def test_export_and_kinematics(tmp_dir: Path) -> PolicyBundle:
    assert DEFAULT_CKPT.is_file(), f"Missing default checkpoint: {DEFAULT_CKPT}"
    bundle_dir = tmp_dir / "bundle"
    export_bundle(DEFAULT_CKPT, bundle_dir)
    bundle = PolicyBundle(bundle_dir)

    assert bundle.single_obs_dim == 28
    assert bundle.history_steps == 5
    assert bundle.input_dim == 140
    assert [w.shape for w in bundle.weights] == [(256, 140), (128, 256), (64, 128), (6, 64)]

    # Verify nominal stance at HOME_JOINTS hip-to-wheel height (0.30753273 m) matches HOME_JOINTS in unified +Y coords
    home_u = np.array([HOME_JOINTS[n] for n in ("hip_L", "hip_R", "hipknee_L", "hipknee_R")]) * LEG_SIGNS
    nom_home = bundle.nominal_legs(0.30753273)
    max_joint_err = float(np.max(np.abs(nom_home - home_u)))
    assert max_joint_err < 3e-3, f"Stance solver mismatch vs HOME_JOINTS: {max_joint_err}"

    # Verify CAD FK grid reproduces height across [0.17, 0.33] to < 2 mm
    upright_g = np.array([0.0, 0.0, -1.0])
    for h_test in (0.17, 0.20, 0.24, 0.30, 0.33):
        q_nom = bundle.nominal_legs(h_test)
        h_l, h_r = bundle.cad_leg_heights(q_nom, upright_g)
        assert abs(h_l - h_test) < 0.002 and abs(h_r - h_test) < 0.002, (
            f"CAD FK height error at {h_test}: ({h_l:.4f}, {h_r:.4f})"
        )

    # Verify pure-NumPy inference speed & finiteness
    t0 = time.perf_counter_ns()
    for _ in range(100):
        y = bundle.forward(np.zeros(140, dtype=np.float32))
    avg_ms = (time.perf_counter_ns() - t0) * 1e-6 / 100.0
    assert y.shape == (6,) and np.isfinite(y).all()
    assert avg_ms < 2.0, f"Unexpectedly slow NumPy inference: {avg_ms:.3f} ms"
    print(f"[PASS] export & kinematics (max_joint_err={max_joint_err:.2e} rad, avg_infer={avg_ms:.3f} ms)")
    return bundle


def test_observation_and_mapping(bundle: PolicyBundle) -> None:
    # 1. Quaternion -> gravity & euler sign check (REP-103: +pitch = nose down, +roll = right down)
    p = math.radians(15.0)
    q_pitch = (math.cos(p / 2), 0.0, math.sin(p / 2), 0.0)
    g_p = quat_to_gravity_body(q_pitch)
    assert abs(g_p[0] - math.sin(p)) < 1e-6 and abs(g_p[2] + math.cos(p)) < 1e-6
    r_deg, p_deg, _ = quat_to_euler_deg_and_yaw(q_pitch)
    assert abs(p_deg - 15.0) < 1e-4 and abs(r_deg) < 1e-4

    r = math.radians(10.0)
    q_roll = (math.cos(r / 2), math.sin(r / 2), 0.0, 0.0)
    g_r = quat_to_gravity_body(q_roll)
    assert abs(g_r[1] + math.sin(r)) < 1e-6
    r_deg, p_deg, _ = quat_to_euler_deg_and_yaw(q_roll)
    assert abs(r_deg - 10.0) < 1e-4 and abs(p_deg) < 1e-4

    # 2. History fill on first active frame, then shift [t, t-1, t-2, t-3, t-4]
    cal = CalibrationConfig(
        policy_to_firmware_index=(0, 2, 1, 3, 4, 5),
        joint_sign=(1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
        leg_zero_offset_rad=(0.1, -0.2, 0.3, -0.4),
    )
    ctrl = WheelLegPolicyController(bundle, cal)

    def mk_obs(seq: int, wz: float, rl_flags: int) -> Observation:
        return Observation(
            seq=seq,
            timestamp_ms=seq * 20,
            gyro=(0.0, 0.0, wz),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            motors=tuple(MotorState(0.0, 0.0, 0.0) for _ in range(6)),
            vx_set=0.0,
            vy_set=0.0,
            wz_set=0.0,
            height_set=0.20,
            leg_height=(0.20, 0.20),
            rl_flags=rl_flags,
            chassis_mode=3,
            fault_flags=0,
            received_monotonic_ns=time.monotonic_ns(),
        )

    res0 = ctrl.step(mk_obs(1, 0.11, RL_FLAG_COMPUTED_TORQUE))
    # All 5 history slices must equal obs_0 on handover!
    for k in range(5):
        assert np.allclose(res0.obs_history[k * 28 : (k + 1) * 28], res0.obs_policy), f"History slot {k} not warm-filled"

    res1 = ctrl.step(mk_obs(2, 0.22, RL_FLAG_COMPUTED_TORQUE))
    assert np.allclose(res1.obs_history[0:28], res1.obs_policy)
    assert np.allclose(res1.obs_history[28:56], res0.obs_policy)
    assert np.allclose(res1.obs_history[56:84], res0.obs_policy)
    assert abs(res1.obs_policy[6] - 0.22) < 1e-6 and abs(res1.obs_history[28 + 6] - 0.11) < 1e-6
    print("[PASS] observation math, gravity signs, and 5-frame history stacking")


def test_closed_loop_and_analysis(bundle: PolicyBundle, tmp_dir: Path) -> None:
    cal = CalibrationConfig.from_json(HOST_DIR / "configs" / "joint_calibration.json")

    # A. Normal Mode 2 handover + active RL + fault incident capture
    log_dir = tmp_dir / "run_mode2"
    transport = MockStm32Transport(bundle, inject_fault_at_s=0.75)
    logger = DeployLogger(log_dir, {"test": "mode2", "calibration": cal.to_dict()}, control_dt=bundle.control_dt)
    summary = run_host_loop(
        transport,
        bundle,
        cal,
        logger,
        dry_run=False,
        gain_scale=1.0,
        duration_s=1.05,
        print_status=False,
    )
    assert transport.tx_bad_crc == 0 and transport.tx_bad_len == 0
    assert len(transport.tx_commands) >= 40, f"Expected >=40 TX commands, got {len(transport.tx_commands)}"

    # Check early GETUP commands were zero-torque heartbeats, and middle NORMAL commands had real Kp/Kd
    early_zero = any(all(v == 0.0 for v in cmd[4:]) for cmd in transport.tx_commands[:8])
    mid_active = any(
        abs(cmd[10] - 100.0) < 1e-4 and abs(cmd[16] - 4.0) < 1e-4 and abs(cmd[20] - 0.6) < 1e-4
        for cmd in transport.tx_commands
    )
    assert early_zero, "Expected zero-torque heartbeats during GETUP"
    assert mid_active, "Expected active V0.2 PD gains (Kp=100, Kd=4/0.6) during NORMAL"

    # Check all log artifacts exist and raw_usb.bin decodes cleanly with 0 CRC errors
    for fname in ("meta.json", "summary.json", "events.log", "raw_usb.bin", "obs_500hz.npz", "control_50hz.csv", "dataset.npz"):
        assert (log_dir / fname).is_file(), f"Missing log artifact: {fname}"
    stream = FrameStream(8192)
    frames = stream.feed((log_dir / "raw_usb.bin").read_bytes())
    assert len(frames) >= 200 and stream.crc_errors == 0 and stream.length_errors == 0

    with np.load(log_dir / "dataset.npz") as ds:
        T = len(ds["t_s"])
        assert ds["obs_policy"].shape == (T, 28)
        assert ds["obs_history"].shape == (T, 140)
        assert ds["actions"].shape == (T, 6)
        assert np.any(ds["rl_active_mask"]), "Expected active RL steps in rl_active_mask"
        assert not np.all(ds["rl_active_mask"]), "Expected non-RL steps (GETUP/FAULT) masked out"

    # Fault must have auto-captured incident_001.csv & incident_001.html
    assert summary["incidents"] >= 1, f"Expected auto incident on fault, summary={summary}"
    assert (log_dir / "incidents" / "incident_001.csv").is_file()
    assert (log_dir / "incidents" / "incident_001.html").is_file()

    # Offline analyzer + replay parity (same bundle -> exact 0 diff!)
    report = analyze_run(log_dir, replay_bundle=bundle.bundle_dir, write_html=True)
    assert (log_dir / "analysis.json").is_file()
    assert (log_dir / "report.html").is_file()
    assert report["policy_replay"]["action_max_diff"] == 0.0, (
        f"Expected 0 replay diff for identical bundle, got {report['policy_replay']['action_max_diff']}"
    )
    print(
        f"[PASS] closed-loop Mode 2 + incident + analyzer "
        f"(rx={len(frames)}, tx={len(transport.tx_commands)}, active={report['frames_rl_active']}, "
        f"incidents={summary['incidents']})"
    )

    # B. Shadow (--dry-run) mode: MCU sends rl_flags=1, host logs non-zero actions but ALWAYS sends all-zero bytes
    dry_dir = tmp_dir / "run_dry"
    dry_transport = MockStm32Transport(bundle, inject_fault_at_s=None, always_rl=True)
    dry_logger = DeployLogger(dry_dir, {"test": "dry_run"}, control_dt=bundle.control_dt)
    run_host_loop(
        dry_transport,
        bundle,
        cal,
        dry_logger,
        dry_run=True,
        gain_scale=1.0,
        duration_s=0.35,
        print_status=False,
    )
    assert len(dry_transport.tx_commands) >= 10
    assert all(all(v == 0.0 for v in cmd[4:]) for cmd in dry_transport.tx_commands), (
        "dry_run transmitted a non-zero command!"
    )
    with np.load(dry_dir / "dataset.npz") as ds:
        assert np.max(np.abs(ds["actions"])) > 1e-3, "dry_run should still record live policy actions"
        assert not np.any(ds["rl_active_mask"]), "dry_run must keep rl_active_mask=False"
    print(f"[PASS] shadow --dry-run safety (tx={len(dry_transport.tx_commands)} all-zero frames, actions logged)")


def main() -> int:
    tmp_dir = Path(tempfile.mkdtemp(prefix="rl_host_test_"))
    try:
        bundle = test_export_and_kinematics(tmp_dir)
        test_observation_and_mapping(bundle)
        test_closed_loop_and_analysis(bundle, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("ALL HOST DEPLOY TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
