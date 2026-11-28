"""Pure-NumPy V0.2 RL policy runtime: observation assembly, inference, and PD command decoding.

Strictly mirrors ``training/robots/wheel_leg_locomotion.py`` (28D x H observation contract,
unified +Y joint coordinates, continuous stance lookup, height slew, friction circle,
action decoding) and ``chassis/Application/Tasks/Src/chassis_rl_control.c`` (expected MCU
500 Hz pure-torque PD law).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np

HOST_DIR = Path(__file__).resolve().parent
DEPLOY_TOOLS_DIR = HOST_DIR.parent / "wheel_leg_deploy" / "Tools"
if str(DEPLOY_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_TOOLS_DIR))

from telemetry_capture import (  # noqa: E402
    Observation,
    RL_FLAG_COMPUTED_TORQUE,
    RL_FLAG_ZERO_TORQUE,
)

POLICY_JOINT_NAMES = ("hip_L", "hip_R", "hipknee_L", "hipknee_R", "wheel_L", "wheel_R")
MAX_FORWARD_SPEED = 2.5
YAW_REFERENCE = 11.0


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(angle, dtype=np.float64)
    return arr - (2.0 * math.pi) * np.round(arr / (2.0 * math.pi))


def quat_to_gravity_body(quat_wxyz: tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Rotate world gravity (0, 0, -1) into body frame using body->world quaternion [w,x,y,z]."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-6:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    w, x, y, z = q / norm
    return np.array([
        2.0 * (w * y - x * z),
        -2.0 * (w * x + y * z),
        -(1.0 - 2.0 * (x * x + y * y)),
    ], dtype=np.float64)


def quat_to_euler_deg_and_yaw(quat_wxyz: tuple[float, ...] | np.ndarray) -> tuple[float, float, float]:
    """Return (roll_deg, pitch_deg, yaw_rad) matching V0.2 training and ins_task.c conventions."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-6:
        return 0.0, 0.0, 0.0
    w, x, y, z = q / norm
    g = quat_to_gravity_body((w, x, y, z))
    roll_deg = math.degrees(math.atan2(-g[1], -g[2]))
    pitch_deg = math.degrees(math.atan2(g[0], math.hypot(g[1], g[2])))
    yaw_rad = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll_deg, pitch_deg, yaw_rad


@dataclass(frozen=True)
class CalibrationConfig:
    """Mapping and calibration between STM32H7 motors[0..5] and V0.2 policy joints[0..5]."""

    policy_to_firmware_index: tuple[int, int, int, int, int, int] = (0, 2, 1, 3, 4, 5)
    joint_sign: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    leg_zero_offset_rad: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    leg_height_source: str = "mcu"  # "mcu" | "cad"
    apply_leg_bias: bool = True
    apply_friction_circle: bool = True
    heading_hold_enabled: bool = False
    heading_hold_gain: float = 2.5
    heading_hold_deadband_deg: float = 0.5
    heading_hold_max_yaw_rate: float = 0.4
    leg_kp: tuple[float, float, float, float] = (100.0, 100.0, 100.0, 100.0)
    leg_kd: tuple[float, float, float, float] = (4.0, 4.0, 4.0, 4.0)
    wheel_kd: tuple[float, float] = (0.6, 0.6)
    leg_torque_limit: float = 40.0
    wheel_torque_limit: float = 5.0

    @classmethod
    def from_json(cls, path: Path) -> CalibrationConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        idx = tuple(int(v) for v in raw.get("policy_to_firmware_index", (0, 2, 1, 3, 4, 5)))
        if sorted(idx) != [0, 1, 2, 3, 4, 5]:
            raise ValueError(f"policy_to_firmware_index must be a permutation of 0..5, got {idx}")
        signs = tuple(float(v) for v in raw.get("joint_sign", (1.0,) * 6))
        if len(signs) != 6 or any(s not in (-1.0, 1.0) for s in signs):
            raise ValueError(f"joint_sign must contain six +/-1.0 values, got {signs}")
        zero = tuple(float(v) for v in raw.get("leg_zero_offset_rad", (0.0,) * 4))
        if len(zero) != 4 or not all(math.isfinite(v) for v in zero):
            raise ValueError("leg_zero_offset_rad must contain 4 finite radians")
        hh = raw.get("heading_hold", {})
        pd = raw.get("pd_gains", {})
        lim = raw.get("torque_limits_nm", {})
        source = str(raw.get("leg_height_source", "mcu")).lower()
        if source not in ("mcu", "cad"):
            raise ValueError(f"leg_height_source must be 'mcu' or 'cad', got {source!r}")
        return cls(
            policy_to_firmware_index=idx,
            joint_sign=signs,
            leg_zero_offset_rad=zero,
            leg_height_source=source,
            apply_leg_bias=bool(raw.get("apply_leg_bias", True)),
            apply_friction_circle=bool(raw.get("apply_friction_circle", True)),
            heading_hold_enabled=bool(hh.get("enabled", False)),
            heading_hold_gain=float(hh.get("gain", 2.5)),
            heading_hold_deadband_deg=float(hh.get("deadband_deg", 0.5)),
            heading_hold_max_yaw_rate=float(hh.get("max_yaw_rate", 0.4)),
            leg_kp=tuple(float(v) for v in pd.get("leg_kp", (100.0,) * 4)),
            leg_kd=tuple(float(v) for v in pd.get("leg_kd", (4.0,) * 4)),
            wheel_kd=tuple(float(v) for v in pd.get("wheel_kd", (0.6, 0.6))),
            leg_torque_limit=float(lim.get("leg", 40.0)),
            wheel_torque_limit=float(lim.get("wheel", 5.0)),
        )

    def to_dict(self) -> dict:
        return {
            "policy_to_firmware_index": list(self.policy_to_firmware_index),
            "joint_sign": list(self.joint_sign),
            "leg_zero_offset_rad": list(self.leg_zero_offset_rad),
            "leg_height_source": self.leg_height_source,
            "apply_leg_bias": self.apply_leg_bias,
            "apply_friction_circle": self.apply_friction_circle,
            "heading_hold": {
                "enabled": self.heading_hold_enabled,
                "gain": self.heading_hold_gain,
                "deadband_deg": self.heading_hold_deadband_deg,
                "max_yaw_rate": self.heading_hold_max_yaw_rate,
            },
            "pd_gains": {
                "leg_kp": list(self.leg_kp),
                "leg_kd": list(self.leg_kd),
                "wheel_kd": list(self.wheel_kd),
            },
            "torque_limits_nm": {
                "leg": self.leg_torque_limit,
                "wheel": self.wheel_torque_limit,
            },
        }


@dataclass(frozen=True)
class PolicyStepResult:
    """All host-side signals for one 50 Hz control tick (for USB TX and logging)."""

    rl_flags_tx: int
    obs_policy: np.ndarray        # (28,) float32
    obs_history: np.ndarray       # (28 * H,) float32
    actions: np.ndarray           # (6,) float32 in policy order
    cmd_vx: float
    cmd_wz: float
    cmd_height: float
    height_rate: float
    roll_deg: float
    pitch_deg: float
    yaw_rad: float
    gravity: np.ndarray           # (3,) float64
    leg_height_mcu: tuple[float, float]
    leg_height_cad: tuple[float, float]
    leg_height_used: tuple[float, float]
    q_sim_unified: np.ndarray     # (4,) float64 in policy order
    q_nominal_unified: np.ndarray # (4,) float64 in policy order
    dq_sim_unified: np.ndarray    # (6,) float64 in policy order
    # Downlink command arrays in STM32H7 firmware order [0..5]
    torque_ff_fw: tuple[float, ...]
    kp_fw: tuple[float, ...]
    kd_fw: tuple[float, ...]
    q_ref_fw: tuple[float, ...]
    v_ref_fw: tuple[float, ...]
    tau_pd_expected_fw: tuple[float, ...]
    infer_ms: float


class PolicyBundle:
    """Loaded lightweight deploy bundle (NumPy MLP + pose table + optional CAD FK grid)."""

    def __init__(self, bundle_dir: Path, *, backend: str = "numpy"):
        self.bundle_dir = Path(bundle_dir).resolve()
        self.meta_path = self.bundle_dir / "policy_meta.json"
        self.npz_path = self.bundle_dir / "policy.npz"
        if not self.meta_path.is_file() or not self.npz_path.is_file():
            raise FileNotFoundError(f"Incomplete deploy bundle (need policy.npz & policy_meta.json): {self.bundle_dir}")

        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.contract = self.meta["contract"]
        self.activation = str(self.contract["activation"]).lower()
        self.single_obs_dim = int(self.contract["single_obs_dim"])
        self.history_steps = int(self.contract["actor_history_steps"])
        self.input_dim = int(self.contract["actor_input_dim"])
        self.control_dt = float(self.contract["control_dt"])
        self.height_low = float(self.contract["height_range"][0])
        self.height_high = float(self.contract["height_range"][1])
        self.height_nominal = float(self.contract["height_nominal"])
        self.height_rate_max = float(self.contract["height_rate"])
        self.height_accel = float(self.contract["height_acceleration"])
        self.wheel_radius = float(self.contract["wheel_radius"])
        self.half_track = float(self.contract["half_track"])
        self.leg_action_scale = float(self.contract["leg_action_scale"])
        self.leg_offset_limit = float(self.contract["leg_offset_limit"])
        self.wheel_action_scale = float(self.contract["wheel_action_scale"])
        self.wheel_target_limit = float(self.contract["wheel_target_limit"])
        self.forward_bounds = (
            float(self.contract["forward_command"][0]),
            float(self.contract["forward_command"][1]),
        )
        self.yaw_bounds = (
            float(self.contract["yaw_command"][0]),
            float(self.contract["yaw_command"][1]),
        )

        with np.load(self.npz_path) as data:
            n_layers = int(data["num_layers"])
            self.weights = [np.array(data[f"w_{i}"], dtype=np.float32) for i in range(n_layers)]
            self.biases = [np.array(data[f"b_{i}"], dtype=np.float32) for i in range(n_layers)]
            self.pose_heights = np.array(data["pose_heights"], dtype=np.float64)
            self.nominal_legs_unified = np.array(data["nominal_legs_unified"], dtype=np.float64)
            self.leg_bias_unified = np.array(data["leg_bias_unified"], dtype=np.float64)
            self.norm_mean = np.array(data["obs_norm_mean"], dtype=np.float32) if "obs_norm_mean" in data else None
            self.norm_std = np.array(data["obs_norm_std"], dtype=np.float32) if "obs_norm_std" in data else None
            self.norm_eps = float(data["obs_norm_eps"]) if "obs_norm_eps" in data else 1e-2
            if "cad_hip_grid" in data:
                self.cad_hip_grid = np.array(data["cad_hip_grid"], dtype=np.float64)
                self.cad_hk_grid = np.array(data["cad_hk_grid"], dtype=np.float64)
                self.cad_vec_x = np.array(data["cad_vec_x"], dtype=np.float64)
                self.cad_vec_z = np.array(data["cad_vec_z"], dtype=np.float64)
            else:
                self.cad_hip_grid = None
                self.cad_hk_grid = None
                self.cad_vec_x = None
                self.cad_vec_z = None

        self.backend = backend.lower()
        self._torch_net = None
        if self.backend == "torch":
            self._init_torch_backend()
        elif self.backend != "numpy":
            raise ValueError(f"Unsupported backend: {backend!r} (expected 'numpy' or 'torch')")

    def _init_torch_backend(self) -> None:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        act_map = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}
        act_cls = act_map[self.activation]
        layers = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            lin = nn.Linear(w.shape[1], w.shape[0])
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(w))
                lin.bias.copy_(torch.from_numpy(b))
            layers.append(lin)
            if i < len(self.weights) - 1:
                layers.append(act_cls())
        self._torch_net = nn.Sequential(*layers).eval()
        self._torch = torch

    def nominal_legs(self, height: float) -> np.ndarray:
        """Linearly interpolate unified +Y nominal leg angles [hip_L, hip_R, hipknee_L, hipknee_R]."""
        span = max(self.height_high - self.height_low, 1e-8)
        pos = float(np.clip((height - self.height_low) / span, 0.0, 1.0)) * (len(self.nominal_legs_unified) - 1)
        lo = min(int(math.floor(pos)), len(self.nominal_legs_unified) - 2)
        frac = pos - lo
        return (1.0 - frac) * self.nominal_legs_unified[lo] + frac * self.nominal_legs_unified[lo + 1]

    def cad_leg_vector_xz(self, hip_u: float, hipknee_u: float) -> tuple[float, float] | None:
        """Bilinearly interpolate V0.2 CAD (hip - wheel) [dx, dz] in base_link frame."""
        if self.cad_hip_grid is None or self.cad_hk_grid is None:
            return None
        hg, kg = self.cad_hip_grid, self.cad_hk_grid
        if not (hg[0] <= hip_u <= hg[-1] and kg[0] <= hipknee_u <= kg[-1]):
            return None
        ih = int(np.searchsorted(hg, hip_u) - 1)
        ik = int(np.searchsorted(kg, hipknee_u) - 1)
        ih = max(0, min(ih, len(hg) - 2))
        ik = max(0, min(ik, len(kg) - 2))
        th = (hip_u - hg[ih]) / max(hg[ih + 1] - hg[ih], 1e-12)
        tk = (hipknee_u - kg[ik]) / max(kg[ik + 1] - kg[ik], 1e-12)
        vx_patch = self.cad_vec_x[ih:ih + 2, ik:ik + 2]
        vz_patch = self.cad_vec_z[ih:ih + 2, ik:ik + 2]
        if not (np.isfinite(vx_patch).all() and np.isfinite(vz_patch).all()):
            return None
        w00 = (1.0 - th) * (1.0 - tk)
        w01 = (1.0 - th) * tk
        w10 = th * (1.0 - tk)
        w11 = th * tk
        dx = float(w00 * vx_patch[0, 0] + w01 * vx_patch[0, 1] + w10 * vx_patch[1, 0] + w11 * vx_patch[1, 1])
        dz = float(w00 * vz_patch[0, 0] + w01 * vz_patch[0, 1] + w10 * vz_patch[1, 0] + w11 * vz_patch[1, 1])
        return dx, dz

    def cad_leg_heights(self, q_sim_u: np.ndarray, gravity: np.ndarray) -> tuple[float, float]:
        """Compute V0.2 CAD sagittal projected leg heights matching wheel_leg_locomotion._leg_projections."""
        up = -gravity
        up_x = float(up[0])
        up_z = float(math.hypot(up[1], up[2]))
        out = []
        for h_idx, hk_idx in ((0, 2), (1, 3)):
            vec = self.cad_leg_vector_xz(float(q_sim_u[h_idx]), float(q_sim_u[hk_idx]))
            if vec is None:
                out.append(float("nan"))
            else:
                out.append(vec[0] * up_x + vec[1] * up_z)
        return (out[0], out[1])

    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        x = np.asarray(obs_vec, dtype=np.float32).reshape(self.input_dim)
        if self.norm_mean is not None and self.norm_std is not None:
            x = (x - self.norm_mean) / (self.norm_std + np.float32(self.norm_eps))
        if self._torch_net is not None:
            with self._torch.no_grad():
                return self._torch_net(self._torch.from_numpy(x)).numpy().astype(np.float32)
        h = x
        act = self.activation
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w.T + b
            if idx < len(self.weights) - 1:
                if act == "elu":
                    h = np.where(h > 0.0, h, np.expm1(np.clip(h, -80.0, 0.0))).astype(np.float32)
                elif act == "relu":
                    h = np.maximum(h, 0.0).astype(np.float32)
                elif act == "tanh":
                    h = np.tanh(h).astype(np.float32)
                elif act in ("silu", "swish"):
                    h = (h / (1.0 + np.exp(-np.clip(h, -80.0, 80.0)))).astype(np.float32)
        return h.astype(np.float32)


class WheelLegPolicyController:
    """Stateful 50 Hz policy controller handling Mode 2 handover, history stacking, and PD command generation."""

    def __init__(
        self,
        bundle: PolicyBundle,
        calibration: CalibrationConfig,
        *,
        dry_run: bool = False,
        gain_scale: float = 1.0,
    ):
        if not 0.0 <= gain_scale <= 1.0:
            raise ValueError(f"gain_scale must be in [0.0, 1.0], got {gain_scale}")
        self.bundle = bundle
        self.cal = calibration
        self.dry_run = dry_run
        self.gain_scale = float(gain_scale)

        self.last_actions = np.zeros(6, dtype=np.float32)
        self.history: np.ndarray | None = (
            np.zeros((bundle.history_steps - 1, bundle.single_obs_dim), dtype=np.float32)
            if bundle.history_steps > 1
            else None
        )
        self.history_needs_fill = True
        self.cmd_height = bundle.height_nominal
        self.height_rate = 0.0
        self.heading_target = 0.0
        self.heading_turning = False
        self.last_cmd_wz = 0.0
        self.prev_rl_active = False

    def reset_for_handover(self, obs: Observation) -> None:
        self.last_actions.fill(0.0)
        if self.history is not None:
            self.history.fill(0.0)
        self.history_needs_fill = True
        h_init = float(obs.height_set)
        if not math.isfinite(h_init):
            h_init = self.bundle.height_nominal
        self.cmd_height = float(np.clip(h_init, self.bundle.height_low, self.bundle.height_high))
        self.height_rate = 0.0
        _, _, yaw = quat_to_euler_deg_and_yaw(obs.quaternion)
        self.heading_target = yaw
        self.heading_turning = False
        self.last_cmd_wz = 0.0

    def _condition_commands(
        self,
        obs: Observation,
        yaw_rad: float,
        leg_heights_used: tuple[float, float],
        dt: float,
    ) -> tuple[float, float, float, float]:
        vx_req = float(np.clip(obs.vx_set, *self.bundle.forward_bounds))
        wz_req = float(np.clip(obs.wz_set, *self.bundle.yaw_bounds))
        h_req = float(np.clip(obs.height_set, self.bundle.height_low, self.bundle.height_high))

        # Optional heading hold (off by default when MCU gimbal-follow computes wz_set)
        if self.cal.heading_hold_enabled:
            turning = abs(wz_req) > 1e-6
            settling = self.heading_turning and (abs(self.last_cmd_wz) > 0.05 or abs(obs.gyro[2]) > 0.10)
            if turning or self.heading_turning:
                self.heading_target = yaw_rad
            self.heading_turning = turning or settling
            if not turning:
                if settling:
                    wz_req = 0.0
                else:
                    err = float(wrap_to_pi(self.heading_target - yaw_rad))
                    db = math.radians(self.cal.heading_hold_deadband_deg)
                    corr = self.cal.heading_hold_gain * math.copysign(max(abs(err) - db, 0.0), err)
                    wz_req = float(np.clip(corr, -self.cal.heading_hold_max_yaw_rate, self.cal.heading_hold_max_yaw_rate))

        # Optional friction-circle / wheel-speed-bound scaling (mirrors wheel_leg_locomotion._advance_commands)
        if self.cal.apply_friction_circle:
            mean_wheel_v = 0.5 * (
                self.cal.joint_sign[4] * obs.motors[self.cal.policy_to_firmware_index[4]].velocity
                + self.cal.joint_sign[5] * obs.motors[self.cal.policy_to_firmware_index[5]].velocity
            ) * self.bundle.wheel_radius
            lateral_accel = abs(mean_wheel_v * float(obs.gyro[2]))
            cur_h = float(np.clip(0.5 * (leg_heights_used[0] + leg_heights_used[1]), 0.10, 0.335))
            radius = float(np.clip(1.0 - lateral_accel * cur_h / (9.81 * self.bundle.half_track), 0.05, 1.0))
            demand = math.hypot(vx_req / MAX_FORWARD_SPEED, wz_req / YAW_REFERENCE)
            outer_wheel = abs(vx_req) + self.bundle.half_track * abs(wz_req)
            scale = min(
                1.0,
                radius / max(demand, 1e-6),
                (self.bundle.wheel_target_limit * self.bundle.wheel_radius) / max(outer_wheel, 1e-6),
            )
            vx_cmd = vx_req * scale
            wz_cmd = wz_req * scale
        else:
            vx_cmd, wz_cmd = vx_req, wz_req

        # Second-order height slew matching wheel_leg_locomotion._advance_commands
        h_err = h_req - self.cmd_height
        h_step_max = self.bundle.height_rate_max * dt
        if self.bundle.height_accel > 0.0:
            dv_max = self.bundle.height_accel * dt
            stop_rate = math.sqrt(dv_max * dv_max + 2.0 * self.bundle.height_accel * abs(h_err))
            des_rate = math.copysign(min(stop_rate - dv_max, self.bundle.height_rate_max), h_err)
            final_rate = h_err / dt
            if abs(final_rate) <= dv_max and abs(final_rate - self.height_rate) <= dv_max:
                des_rate = final_rate
            h_rate = self.height_rate + float(np.clip(des_rate - self.height_rate, -dv_max, dv_max))
            next_h = float(np.clip(self.cmd_height + h_rate * dt, self.bundle.height_low, self.bundle.height_high))
            h_delta = next_h - self.cmd_height
        else:
            h_delta = float(np.clip(h_err, -h_step_max, h_step_max))
            next_h = float(np.clip(self.cmd_height + h_delta, self.bundle.height_low, self.bundle.height_high))
            h_delta = next_h - self.cmd_height

        self.cmd_height = next_h
        self.height_rate = h_delta / dt
        self.last_cmd_wz = wz_cmd
        return vx_cmd, wz_cmd, self.cmd_height, self.height_rate

    def _expected_mcu_pd_torque(
        self,
        obs: Observation,
        torque_ff: tuple[float, ...],
        kp: tuple[float, ...],
        kd: tuple[float, ...],
        q_ref: tuple[float, ...],
        v_ref: tuple[float, ...],
    ) -> tuple[float, ...]:
        """Mirror chassis_rl_control.c ChassisRL_StepPD() exactly for torque-residual diagnostics."""
        out = [0.0] * 6
        for i in range(4):
            kp_c = float(np.clip(kp[i], 0.0, 200.0))
            kd_c = float(np.clip(kd[i], 0.0, 10.0))
            pos_err = float(wrap_to_pi(q_ref[i] - obs.motors[i].position_multi))
            vel_err = v_ref[i] - obs.motors[i].velocity
            tau = torque_ff[i] + kp_c * pos_err + kd_c * vel_err
            out[i] = float(np.clip(tau, -self.cal.leg_torque_limit, self.cal.leg_torque_limit))
        for i in range(4, 6):
            kd_c = float(np.clip(kd[i], 0.0, 5.0))
            v_c = float(np.clip(v_ref[i], -52.0, 52.0))
            tau = torque_ff[i] + kd_c * (v_c - obs.motors[i].velocity)
            out[i] = float(np.clip(tau, -self.cal.wheel_torque_limit, self.cal.wheel_torque_limit))
        return tuple(out)

    def step(self, obs: Observation, *, dt: float | None = None) -> PolicyStepResult:
        """Process latest Observation at 50 Hz and return downlink PD command + diagnostic snapshot."""
        import time

        step_dt = self.bundle.control_dt if dt is None or dt <= 0.0 else float(dt)
        mcu_wants_rl = (obs.rl_flags == RL_FLAG_COMPUTED_TORQUE)
        if not mcu_wants_rl:
            self.reset_for_handover(obs)
        elif not self.prev_rl_active:
            self.reset_for_handover(obs)
        self.prev_rl_active = mcu_wants_rl

        roll_deg, pitch_deg, yaw_rad = quat_to_euler_deg_and_yaw(obs.quaternion)
        gravity = quat_to_gravity_body(obs.quaternion)

        # Map firmware motor observations into unified +Y V0.2 policy order
        idx = self.cal.policy_to_firmware_index
        signs = np.asarray(self.cal.joint_sign, dtype=np.float64)
        zeros = np.asarray(self.cal.leg_zero_offset_rad, dtype=np.float64)
        q_obs_policy_legs = np.array([obs.motors[idx[j]].position_multi for j in range(4)], dtype=np.float64)
        dq_obs_policy = np.array([obs.motors[idx[j]].velocity for j in range(6)], dtype=np.float64)

        q_sim_u = wrap_to_pi(signs[:4] * (q_obs_policy_legs - zeros))
        dq_sim_u = signs * dq_obs_policy

        leg_h_mcu = (float(obs.leg_height[0]), float(obs.leg_height[1]))
        leg_h_cad = self.bundle.cad_leg_heights(q_sim_u, gravity)
        if (
            self.cal.leg_height_source == "cad"
            and math.isfinite(leg_h_cad[0])
            and math.isfinite(leg_h_cad[1])
        ):
            leg_h_used = leg_h_cad
        else:
            leg_h_used = leg_h_mcu

        cmd_vx, cmd_wz, cmd_height, height_rate = self._condition_commands(obs, yaw_rad, leg_h_used, step_dt)
        nominal_u = self.bundle.nominal_legs(cmd_height)
        joint_pos_obs = wrap_to_pi(q_sim_u - nominal_u)

        # Assemble 28D single-frame policy observation (exact V0.2 order)
        obs_policy = np.concatenate((
            np.array([cmd_vx, cmd_wz, cmd_height * 5.0, height_rate], dtype=np.float32),
            np.asarray(obs.gyro, dtype=np.float32),
            gravity.astype(np.float32),
            joint_pos_obs.astype(np.float32),
            (dq_sim_u * 0.1).astype(np.float32),
            self.last_actions.astype(np.float32),
            np.array([leg_h_used[0] * 5.0, leg_h_used[1] * 5.0], dtype=np.float32),
        ))

        if self.history is not None:
            if self.history_needs_fill:
                self.history[:] = obs_policy[None, :]
                self.history_needs_fill = False
            obs_history = np.concatenate((obs_policy, self.history.reshape(-1))).astype(np.float32)
        else:
            obs_history = obs_policy.copy()

        # Run inference even in dry_run / standby so shadow logs capture policy behavior
        t0 = time.perf_counter_ns()
        if np.isfinite(obs_history).all():
            raw_actions = self.bundle.forward(obs_history)
            actions = np.clip(raw_actions, -10.0, 10.0).astype(np.float32)
        else:
            actions = np.zeros(6, dtype=np.float32)
        infer_ms = (time.perf_counter_ns() - t0) * 1e-6

        if not np.isfinite(actions).all():
            actions = np.zeros(6, dtype=np.float32)

        # Update history & last_actions when active (or in dry_run shadow evaluation while MCU is active)
        if mcu_wants_rl:
            if self.history is not None:
                if self.history.shape[0] > 1:
                    self.history[1:] = self.history[:-1]
                self.history[0] = obs_policy
            self.last_actions[:] = actions

        # Decode actions -> firmware-order PD command arrays
        offsets_u = np.clip(actions[:4].astype(np.float64) * self.bundle.leg_action_scale,
                            -self.bundle.leg_offset_limit, self.bundle.leg_offset_limit)
        bias_u = self.bundle.leg_bias_unified if self.cal.apply_leg_bias else np.zeros(4, dtype=np.float64)
        q_target_sim_u = nominal_u + offsets_u + bias_u
        q_ref_policy = wrap_to_pi(zeros + signs[:4] * q_target_sim_u)

        v_target_sim_u = np.clip(
            actions[4:6].astype(np.float64) * self.bundle.wheel_action_scale,
            -self.bundle.wheel_target_limit,
            self.bundle.wheel_target_limit,
        )
        v_ref_policy = signs[4:6] * v_target_sim_u

        # Scatter from policy order into STM32H7 firmware order [0..5]
        torque_ff_fw = [0.0] * 6
        kp_fw = [0.0] * 6
        kd_fw = [0.0] * 6
        q_ref_fw = [0.0] * 6
        v_ref_fw = [0.0] * 6
        for j in range(4):
            fw = idx[j]
            kp_fw[fw] = self.cal.leg_kp[j] * self.gain_scale
            kd_fw[fw] = self.cal.leg_kd[j] * self.gain_scale
            q_ref_fw[fw] = float(q_ref_policy[j])
            v_ref_fw[fw] = 0.0
        for j in range(2):
            fw = idx[4 + j]
            kp_fw[fw] = 0.0
            kd_fw[fw] = self.cal.wheel_kd[j] * self.gain_scale
            q_ref_fw[fw] = 0.0
            v_ref_fw[fw] = float(v_ref_policy[j])

        tx_flag = (
            RL_FLAG_COMPUTED_TORQUE
            if (mcu_wants_rl and not self.dry_run and np.isfinite(obs_history).all())
            else RL_FLAG_ZERO_TORQUE
        )
        if tx_flag == RL_FLAG_ZERO_TORQUE:
            tau_pd_expected = (0.0,) * 6
        else:
            tau_pd_expected = self._expected_mcu_pd_torque(
                obs, tuple(torque_ff_fw), tuple(kp_fw), tuple(kd_fw), tuple(q_ref_fw), tuple(v_ref_fw)
            )

        return PolicyStepResult(
            rl_flags_tx=tx_flag,
            obs_policy=obs_policy,
            obs_history=obs_history,
            actions=actions,
            cmd_vx=cmd_vx,
            cmd_wz=cmd_wz,
            cmd_height=cmd_height,
            height_rate=height_rate,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_rad=yaw_rad,
            gravity=gravity,
            leg_height_mcu=leg_h_mcu,
            leg_height_cad=leg_h_cad,
            leg_height_used=leg_h_used,
            q_sim_unified=q_sim_u,
            q_nominal_unified=nominal_u,
            dq_sim_unified=dq_sim_u,
            torque_ff_fw=tuple(torque_ff_fw),
            kp_fw=tuple(kp_fw),
            kd_fw=tuple(kd_fw),
            q_ref_fw=tuple(q_ref_fw),
            v_ref_fw=tuple(v_ref_fw),
            tau_pd_expected_fw=tau_pd_expected,
            infer_ms=infer_ms,
        )
