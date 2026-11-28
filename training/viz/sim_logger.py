"""Automatic run logger for Genesis demos / keyboard driving.

Every control step of env 0 is appended to ``telemetry.csv``. Notable situations (falls, resets,
mode switches, torque saturation, wheel lift-off, large tracking errors, slow simulation, ...)
are written to ``events.log``. For every fall an ``incidents/fall_XXX.csv`` slice and an offline
``fall_XXX.html`` chart of the seconds before the fall are saved, together with a heuristic list
of likely causes, so loss of control during driving can be analysed afterwards.
"""

from __future__ import annotations

import csv
import json
from collections import deque
from datetime import datetime
from math import degrees, nan
from pathlib import Path
from time import monotonic

import numpy as np
import torch

LEG_NAMES = ("hip_L", "hip_R", "knee_L", "knee_R")

COLUMNS = (
    "t", "step",
    # Operator input and the planned (sloped) commands.
    "key_fwd", "key_turn", "key_height", "vx_target", "wz_target", "height_goal",
    "cmd_vx", "cmd_wz", "cmd_height",
    # Base state (velocities in the body frame).
    "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg",
    "vx", "vy", "vz", "wx", "wy", "wz",
    "clearance", "leg_h_L", "leg_h_R",
    # Joints (raw Genesis joint frame), applied torques and policy actions.
    *(f"q_{n}" for n in LEG_NAMES), *(f"dq_{n}" for n in LEG_NAMES), *(f"tau_{n}" for n in LEG_NAMES),
    "w_wheel_L", "w_wheel_R", "tau_wheel_L", "tau_wheel_R",
    *(f"act_{n}" for n in LEG_NAMES), "act_wheel_L", "act_wheel_R",
    # Contacts and flags.
    "fz_wheel_L", "fz_wheel_R", "body_force",
    "leg_sat", "wheel_sat", "fallen", "done", "timeout", "rtf",
)
COL = {name: i for i, name in enumerate(COLUMNS)}
INCIDENT_PLOTS = (
    ("姿态 [deg]", ("pitch_deg", "roll_deg")),
    ("前向速度 [m/s]", ("vx_target", "cmd_vx", "vx")),
    ("偏航角速度 [rad/s]", ("wz_target", "cmd_wz", "wz")),
    ("按键", ("key_fwd", "key_turn", "key_height")),
    ("腿部力矩 [N·m]", tuple(f"tau_{n}" for n in LEG_NAMES)),
    ("轮子力矩 [N·m]", ("tau_wheel_L", "tau_wheel_R")),
    ("接触力 [N]", ("fz_wheel_L", "fz_wheel_R", "body_force")),
    ("高度 [m]", ("clearance", "cmd_height", "leg_h_L", "leg_h_R")),
)


def _quat_to_euler(w, x, y, z):
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


class _Persistence:
    """Debounced rising-edge detector with a cooldown, per warning kind."""

    def __init__(self, hold_steps, cooldown_steps):
        self.hold_steps = hold_steps
        self.cooldown_steps = cooldown_steps
        self.count = 0
        self.cooldown = 0

    def update(self, active):
        self.cooldown = max(0, self.cooldown - 1)
        self.count = self.count + 1 if active else 0
        if self.count == self.hold_steps and self.cooldown == 0:
            self.cooldown = self.cooldown_steps
            return True
        return False


class SimLogger:
    def __init__(self, directory, env, options=None, *, history_seconds=4.0, incident_seconds=3.0,
                 after_seconds=1.0, flush_seconds=1.0):
        self.env = env
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "incidents").mkdir(exist_ok=True)
        self.dt = env.control_dt
        self.leg_limit = float(getattr(env, "leg_torque_limit", 12.0))
        self.wheel_limit = float(getattr(env, "wheel_torque_limit", 5.0))
        self.history = deque(maxlen=max(2, round(history_seconds / self.dt)))
        self.incident_steps = max(1, round(incident_seconds / self.dt))
        self.after_steps = max(0, round(after_seconds / self.dt))
        self.flush_steps = max(1, round(flush_seconds / self.dt))
        self.mode_source = None
        self.mode = "NORMAL"
        self.mode_since = 0.0
        self.step_index = 0
        self.t = 0.0
        self.last_wall = None
        self.wall_window = deque(maxlen=max(2, round(0.5 / self.dt)))
        self.last_fallen = False
        self.last_keys = None
        self.falls = 0
        self.pending = []  # incidents waiting for their post-fall frames
        self.counts = {}
        self.started = datetime.now()

        steps = lambda seconds: max(1, round(seconds / self.dt))
        cooldown = steps(1.0)
        self.warnings = {
            "LEG_SATURATION": _Persistence(steps(0.2), cooldown),
            "WHEEL_SATURATION": _Persistence(steps(0.2), cooldown),
            "WHEEL_LIFTOFF": _Persistence(steps(0.1), cooldown),
            "PITCH_LARGE": _Persistence(1, cooldown),
            "ROLL_LARGE": _Persistence(1, cooldown),
            "SPEED_TRACKING": _Persistence(steps(0.5), steps(2.0)),
            "YAW_TRACKING": _Persistence(steps(0.5), steps(2.0)),
            "BODY_CONTACT": _Persistence(1, cooldown),
            "SLOW_SIM": _Persistence(steps(1.0), steps(5.0)),
        }

        self.csv_file = (self.directory / "telemetry.csv").open("w", newline="", encoding="utf-8")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(COLUMNS + ("mode",))
        self.events_file = (self.directory / "events.log").open("w", encoding="utf-8")
        meta = {
            "started": self.started.isoformat(timespec="seconds"),
            "robot": getattr(env, "label", type(env).__name__),
            "control_dt": self.dt,
            "num_envs": env.num_envs,
            "logged_env": 0,
            "leg_torque_limit": self.leg_limit,
            "wheel_torque_limit": self.wheel_limit,
            "terrain": env.terrain_description() if hasattr(env, "terrain_description") else None,
            "options": {k: str(v) for k, v in vars(options).items()} if options is not None else {},
            "columns": list(COLUMNS) + ["mode"],
            "notes": "Velocities in body frame; joint q/dq/tau in raw Genesis joint frame; "
                     "leg_sat/wheel_sat = |tau| >= 99% of the limit.",
        }
        (self.directory / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self.event("START", f"robot={meta['robot']} dt={self.dt:g}s terrain={meta['terrain']}")

    # ------------------------------------------------------------------ recording
    def event(self, kind, message="", t=None):
        t = self.t if t is None else t
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.events_file.write(f"[{t:9.2f}s] {kind:<17} {message}\n")
        self.events_file.flush()

    def _gather(self, actions, extras, done):
        env, robot = self.env, self.env.robot
        lin_vel, ang_vel, gravity, _ = env._base_state()
        contact = robot.get_links_net_contact_force()[0]
        leg_heights = extras.get("leg_heights")
        nan2 = torch.full((2,), nan, device=env.device)
        act = actions[0].float().reshape(-1)[:6]
        act = torch.cat((act, act.new_full((6 - act.numel(),), nan)))
        parts = (
            robot.get_pos()[0], robot.get_quat()[0], lin_vel[0], ang_vel[0], gravity[0],
            env.body_clearance[0:1], leg_heights[0] if leg_heights is not None else nan2,
            robot.get_dofs_position(env.leg_dofs)[0], robot.get_dofs_velocity(env.leg_dofs)[0],
            robot.get_dofs_control_force(env.leg_dofs)[0],
            robot.get_dofs_velocity(env.wheel_dofs)[0], robot.get_dofs_control_force(env.wheel_dofs)[0],
            act, env.commands[0, :3],
            contact[env.wheel_links, 2], contact[env.body_links].norm(dim=-1).amax().reshape(1),
            extras["fallen"][0:1], done[0:1], extras["time_outs"][0:1],
        )
        return torch.cat([p.reshape(-1).float() for p in parts]).cpu().numpy()

    def step(self, t, actions, extras, done, inputs=(0.0, 0.0, 0.0, nan, nan, nan)):
        """Record one control step of env 0 (call right after env.step)."""
        self.t = t
        now = monotonic()
        if self.last_wall is not None:
            self.wall_window.append(now - self.last_wall)
        self.last_wall = now
        rtf = (len(self.wall_window) * self.dt / max(sum(self.wall_window), 1e-6)) if self.wall_window else nan

        raw = self._gather(actions, extras, done)
        pos, quat, lin, ang, _grav = raw[0:3], raw[3:7], raw[7:10], raw[10:13], raw[13:16]
        clearance, leg_h = raw[16], raw[17:19]
        q, dq, tau, w_wheel, tau_wheel = raw[19:23], raw[23:27], raw[27:31], raw[31:33], raw[33:35]
        act, cmd, fz, body_force = raw[35:41], raw[41:44], raw[44:46], raw[46]
        fallen, is_done, timeout = raw[47] > 0.5, raw[48] > 0.5, raw[49] > 0.5
        roll, pitch, yaw = _quat_to_euler(*quat)
        leg_sat = float((np.abs(tau) >= 0.99 * self.leg_limit).any())
        wheel_sat = float((np.abs(tau_wheel) >= 0.99 * self.wheel_limit).any())
        row = np.array((t, self.step_index, *inputs, *cmd, *pos, roll, pitch, yaw, *lin, *ang, clearance, *leg_h,
                        *q, *dq, *tau, *w_wheel, *tau_wheel, *act, *fz, body_force,
                        leg_sat, wheel_sat, float(fallen), float(is_done), float(timeout), rtf), dtype=np.float64)

        mode = self.mode_source() if self.mode_source is not None else "NORMAL"
        if mode != self.mode:
            self.event("MODE", f"{self.mode} -> {mode} (在 {self.mode} 停留 {t - self.mode_since:.2f}s)")
            self.mode, self.mode_since = mode, t

        self._input_events(row)
        if not is_done:
            self._warning_events(row, mode)
        if fallen and not self.last_fallen and mode == "NORMAL":
            self._on_fall(row, is_done)
        self.last_fallen = bool(fallen) and not is_done
        if is_done:
            reason = "超时/出界" if timeout else ("摔倒" if fallen else "未知")
            self.event("RESET", f"环境被重置: {reason}")

        self.history.append((row, mode))
        for incident in self.pending:
            incident["after"].append((row, mode))
        self._finish_incidents()
        self.csv.writerow([f"{v:.6g}" for v in row] + [mode])
        self.step_index += 1
        if self.step_index % self.flush_steps == 0:
            self.csv_file.flush()

    def _input_events(self, row):
        keys = tuple(row[COL[c]] for c in ("key_fwd", "key_turn", "key_height"))
        if keys != self.last_keys:
            names = []
            if keys[0]:
                names.append("W" if keys[0] > 0 else "S")
            if keys[1]:
                names.append("A" if keys[1] > 0 else "D")
            if keys[2]:
                names.append("Z" if keys[2] > 0 else "X")
            self.event("INPUT", f"按键={'+'.join(names) or '无'} vx_target={row[COL['vx_target']]:+.2f} "
                                f"wz_target={row[COL['wz_target']]:+.2f} 实际vx={row[COL['vx']]:+.2f}")
            self.last_keys = keys

    def _warning_events(self, row, mode):
        r = lambda name: row[COL[name]]
        normal = mode == "NORMAL"
        fz_min = min(r("fz_wheel_L"), r("fz_wheel_R"))
        checks = {
            "LEG_SATURATION": (normal and r("leg_sat") > 0,
                               lambda: "腿部电机力矩饱和 " + " ".join(f"{n}={r('tau_' + n):+.1f}" for n in LEG_NAMES)),
            "WHEEL_SATURATION": (normal and r("wheel_sat") > 0,
                                 lambda: f"轮子力矩饱和 L={r('tau_wheel_L'):+.2f} R={r('tau_wheel_R'):+.2f} "
                                         f"(限值 {self.wheel_limit:g})，可能无法维持平衡"),
            "WHEEL_LIFTOFF": (normal and fz_min < 1.0,
                              lambda: f"轮子离地 fzL={r('fz_wheel_L'):.1f}N fzR={r('fz_wheel_R'):.1f}N"),
            "PITCH_LARGE": (normal and abs(r("pitch_deg")) > 20.0, lambda: f"俯仰角过大 pitch={r('pitch_deg'):+.1f}deg"),
            "ROLL_LARGE": (normal and abs(r("roll_deg")) > 15.0,
                           lambda: f"横滚角过大 roll={r('roll_deg'):+.1f}deg wz={r('wz'):+.2f}"),
            "SPEED_TRACKING": (normal and abs(r("vx") - r("cmd_vx")) > 0.6,
                               lambda: f"速度跟踪误差大 cmd={r('cmd_vx'):+.2f} 实际={r('vx'):+.2f}m/s"),
            "YAW_TRACKING": (normal and abs(r("wz") - r("cmd_wz")) > 1.5,
                             lambda: f"偏航跟踪误差大 cmd={r('cmd_wz'):+.2f} 实际={r('wz'):+.2f}rad/s"),
            "BODY_CONTACT": (normal and r("body_force") > 10.0, lambda: f"车身接触 {r('body_force'):.0f}N"),
            "SLOW_SIM": (r("rtf") < 0.7, lambda: f"仿真慢于实时 rtf={r('rtf'):.2f}x，键盘输入会有延迟感"),
        }
        for kind, (active, message) in checks.items():
            if self.warnings[kind].update(bool(active)):
                self.event(kind, message())

    # ------------------------------------------------------------------ falls
    def _fall_reasons(self, row):
        env, r = self.env, lambda name: row[COL[name]]
        reasons = []
        if r("body_force") > 10.0:
            reasons.append(f"车身接触 {r('body_force'):.0f}N")
        if r("clearance") < 0.045:
            reasons.append(f"离地高度过低 {r('clearance') * 100:.1f}cm")
        tilt = degrees(np.arccos(np.clip(np.cos(np.radians(r("roll_deg"))) * np.cos(np.radians(r("pitch_deg"))), -1, 1)))
        if tilt > 49.5:
            reasons.append(f"倾斜 {tilt:.0f}deg > 49.5deg")
        instant = degrees(getattr(env, "instant_pitch_rad", np.inf))
        if abs(r("pitch_deg")) > instant:
            reasons.append(f"|pitch| {abs(r('pitch_deg')):.0f}deg > {instant:.0f}deg")
        return reasons or [f"持续俯仰超限 (>{degrees(getattr(env, 'sustained_pitch_rad', nan)):.0f}deg)"]

    def _likely_causes(self, window, fall_row):
        data = np.stack([row for row, _ in window]) if window else fall_row[None]
        modes = [mode for _, mode in window]
        c = lambda name: data[:, COL[name]]
        last = max(1, round(1.0 / self.dt))
        recent = data[-last:]
        rc = lambda name: recent[:, COL[name]]
        causes = []
        if rc("leg_sat").mean() > 0.3:
            causes.append(f"腿部力矩饱和占最后1s的 {100 * rc('leg_sat').mean():.0f}% (限值 {self.leg_limit:g}N·m)，腿撑不住")
        if rc("wheel_sat").mean() > 0.3:
            causes.append(f"轮子力矩饱和占最后1s的 {100 * rc('wheel_sat').mean():.0f}% (限值 {self.wheel_limit:g}N·m)，"
                          "轮子已无余量去纠正俯仰")
        lift = (np.minimum(c("fz_wheel_L"), c("fz_wheel_R")) < 1.0)
        if lift.sum() * self.dt >= 0.1:
            first = c("t")[lift.argmax()]
            causes.append(f"摔倒前有 {lift.sum() * self.dt:.2f}s 单轮/双轮离地 (首次 t={first:.2f}s)，可能被地形弹起或侧倾")
        speed_err = np.abs(c("vx") - c("cmd_vx"))
        if speed_err.max() > 0.6:
            causes.append(f"速度跟踪误差最大 {speed_err.max():.2f}m/s，策略跟不上指令 (指令过激或地形阻碍)")
        key = c("key_fwd")
        nonzero = key[key != 0]
        if nonzero.size and (np.diff(np.sign(nonzero)) != 0).any():
            causes.append("窗口内出现前进/后退反向操作 (急刹或反打)，俯仰冲击大")
        if np.abs(c("vx")).max() > 1.5 and (np.diff(c("cmd_vx")) * np.sign(c("vx")[1:]) < -0.02).any():
            causes.append(f"高速 ({np.abs(c('vx')).max():.2f}m/s) 时减速，刹车产生大俯仰力矩")
        if np.abs(c("wz")).max() > 2.5 and np.abs(c("roll_deg")).max() > 10.0:
            causes.append(f"转向过快 (|wz|最大 {np.abs(c('wz')).max():.1f}rad/s) 导致侧倾 {np.abs(c('roll_deg')).max():.0f}deg")
        if np.abs(c("pitch_deg")).max() > 20.0:
            idx = np.abs(c("pitch_deg")) > 20.0
            causes.append(f"俯仰角在 t={c('t')[idx.argmax()]:.2f}s 首次超过20deg，之后发散")
        touch = c("body_force") > 10.0
        if touch.any():
            k = touch.argmax()
            if abs(c("pitch_deg")[k]) < 25.0 and abs(c("roll_deg")[k]) < 25.0:
                causes.append(f"车身在姿态还正常时就碰到了东西 (t={c('t')[k]:.2f}s)，可能是撞到障碍/台阶边缘或底盘太低")
        if np.nanmin(c("cmd_height")) < 0.15 and touch.any():
            causes.append("身高指令偏低，底盘离地间隙不足")
        if any(mode != "NORMAL" for mode in modes):
            since = self.t - self.mode_since if self.mode == "NORMAL" else 0.0
            causes.append(f"窗口内存在自起过程，可能是自起交接后策略不稳定 (交接后 {since:.2f}s 摔倒)")
        if np.nanmin(c("rtf")) < 0.8:
            causes.append(f"仿真实时率最低 {np.nanmin(c('rtf')):.2f}x，按键响应可能滞后")
        return causes or ["未匹配到已知模式，请查看 incident 图表和 CSV"]

    def _on_fall(self, row, is_done):
        self.falls += 1
        window = list(self.history)[-self.incident_steps:]
        # With reset-on-fall the current frame is already the respawned state.
        reference = window[-1][0] if is_done and window else row
        reasons = self._fall_reasons(reference)
        causes = self._likely_causes(window, reference)
        name = f"fall_{self.falls:03d}"
        self.event("FALL", f"#{self.falls} 判定: {'; '.join(reasons)}" + (" (重置前最后一帧)" if is_done else ""))
        for cause in causes:
            self.events_file.write(f"{'':31}可能原因: {cause}\n")
        self.events_file.write(f"{'':31}详情: incidents/{name}.csv / incidents/{name}.html\n")
        self.events_file.flush()
        print(f"[sim-log] t={self.t:.2f}s 摔倒 #{self.falls}: {'; '.join(reasons)} | 可能原因: {causes[0]}", flush=True)
        self.pending.append({"name": name, "before": window, "fall_row": (row, self.mode), "after": [],
                             "t": self.t, "reasons": reasons, "causes": causes, "include_fall": not is_done})

    def _finish_incidents(self, force=False):
        keep = []
        for incident in self.pending:
            if force or len(incident["after"]) >= self.after_steps:
                self._write_incident(incident)
            else:
                keep.append(incident)
        self.pending = keep

    def _write_incident(self, incident):
        rows = incident["before"] + incident["after"]
        path = self.directory / "incidents" / incident["name"]
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(COLUMNS + ("mode",))
            for row, mode in rows:
                writer.writerow([f"{v:.6g}" for v in row] + [mode])
        series = {name: [None if not np.isfinite(row[COL[name]]) else round(float(row[COL[name]]), 5)
                         for row, _ in rows] for name in COLUMNS}
        payload = {"t": series["t"], "series": series, "plots": INCIDENT_PLOTS, "fall_t": incident["t"],
                   "title": f"{incident['name']}  t={incident['t']:.2f}s",
                   "reasons": incident["reasons"], "causes": incident["causes"],
                   "limits": {"leg": self.leg_limit, "wheel": self.wheel_limit}}
        path.with_suffix(".html").write_text(_HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False)),
                                             encoding="utf-8")

    # ------------------------------------------------------------------ shutdown
    def close(self):
        if self.csv_file.closed:
            return
        self._finish_incidents(force=True)
        duration = self.t
        summary = {
            "duration_s": round(duration, 2),
            "steps": self.step_index,
            "falls": self.falls,
            "event_counts": self.counts,
            "directory": str(self.directory),
        }
        self.event("STOP", json.dumps(summary, ensure_ascii=False))
        (self.directory / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        self.csv_file.close()
        self.events_file.close()
        warnings = {k: v for k, v in self.counts.items() if k not in ("START", "STOP", "INPUT", "MODE")}
        print(f"[sim-log] {duration:.1f}s, {self.falls} 次摔倒, 事件统计 {warnings}. 日志目录: {self.directory}", flush=True)


_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>Sim incident</title>
<style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:16px;background:#111;color:#ddd}
h1{font-size:18px} ul{margin:4px 0 12px} .plot{margin:6px 0} canvas{background:#1b1b1b;width:100%;height:170px}
.legend span{margin-right:12px;font-size:12px}
</style></head><body>
<h1 id="title"></h1><div>判定: <span id="reasons"></span></div><div>可能原因:</div><ul id="causes"></ul>
<div id="plots"></div>
<script>
const D = __DATA__;
const colors = ["#4fc3f7","#ffb74d","#81c784","#e57373","#ba68c8","#fff176"];
document.getElementById("title").textContent = D.title;
document.getElementById("reasons").textContent = D.reasons.join("; ");
for (const c of D.causes) { const li = document.createElement("li"); li.textContent = c; causes.appendChild(li); }
const t = D.t;
for (const [label, names] of D.plots) {
  const div = document.createElement("div"); div.className = "plot";
  const legend = document.createElement("div"); legend.className = "legend";
  legend.innerHTML = "<b>" + label + "</b> " + names.map((n, i) => `<span style="color:${colors[i]}">■ ${n}</span>`).join("");
  const canvas = document.createElement("canvas"); div.append(legend, canvas); plots.appendChild(div);
  requestAnimationFrame(() => draw(canvas, names, label));
}
function draw(canvas, names, label) {
  const w = canvas.width = canvas.clientWidth * devicePixelRatio, h = canvas.height = canvas.clientHeight * devicePixelRatio;
  const g = canvas.getContext("2d"); const pad = 40 * devicePixelRatio;
  let lo = Infinity, hi = -Infinity;
  for (const n of names) for (const v of D.series[n]) if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const lim = label.startsWith("腿部") ? D.limits.leg : label.startsWith("轮子") ? D.limits.wheel : null;
  if (lim) { lo = Math.min(lo, -lim * 1.05); hi = Math.max(hi, lim * 1.05); }
  if (!isFinite(lo)) { lo = -1; hi = 1; } if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
  const x = v => pad + (v - t[0]) / Math.max(t[t.length - 1] - t[0], 1e-6) * (w - pad * 1.2);
  const y = v => h - 12 - (v - lo) / (hi - lo) * (h - 24);
  g.strokeStyle = "#333"; g.fillStyle = "#888"; g.font = `${11 * devicePixelRatio}px sans-serif`;
  for (const v of [lo, (lo + hi) / 2, hi]) { g.beginPath(); g.moveTo(pad, y(v)); g.lineTo(w, y(v)); g.stroke(); g.fillText(v.toFixed(2), 2, y(v) + 4); }
  if (lim) { g.strokeStyle = "#a33"; g.setLineDash([6, 4]); for (const v of [lim, -lim]) { g.beginPath(); g.moveTo(pad, y(v)); g.lineTo(w, y(v)); g.stroke(); } g.setLineDash([]); }
  g.strokeStyle = "#f44"; g.beginPath(); g.moveTo(x(D.fall_t), 0); g.lineTo(x(D.fall_t), h); g.stroke();
  names.forEach((n, i) => {
    g.strokeStyle = colors[i]; g.lineWidth = 1.5 * devicePixelRatio; g.beginPath(); let pen = false;
    D.series[n].forEach((v, k) => { if (v === null) { pen = false; return; } pen ? g.lineTo(x(t[k]), y(v)) : g.moveTo(x(t[k]), y(v)); pen = true; });
    g.stroke();
  });
  g.fillStyle = "#888"; g.fillText(`t ${t[0].toFixed(2)}s`, pad, h - 1); g.fillText(`${t[t.length - 1].toFixed(2)}s`, w - 50 * devicePixelRatio, h - 1);
}
</script></body></html>
"""
