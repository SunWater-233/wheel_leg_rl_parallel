#!/usr/bin/env python3
"""Offline diagnostic, Sim2Real SysID analyzer, and new-policy replay tool for deploy logs.

Reads a recorded run directory (``dataset.npz``, ``obs_500hz.npz``, ``meta.json``,
``summary.json``) and produces:
  1. Terminal diagnostic summary + ``analysis.json``
  2. Self-contained interactive ``report.html``
  3. Actuator lag & PD torque residual SysID metrics (for next-round training DR)
  4. 28D observation OOD check against V0.2 training ranges
  5. Optional ``--replay-bundle`` shadow comparison of a newly trained checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

HOST_DIR = Path(__file__).resolve().parent
DEPLOY_TOOLS_DIR = HOST_DIR.parent / "wheel_leg_deploy" / "Tools"
for p in (HOST_DIR, DEPLOY_TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from telemetry_capture import MOTOR_NAMES  # noqa: E402
from policy_runtime import POLICY_JOINT_NAMES, PolicyBundle  # noqa: E402

OBS_28_SPEC = (
    ("cmd_vx", -2.6, 2.6),
    ("cmd_wz", -10.5, 10.5),
    ("cmd_height_x5", 0.80, 1.70),
    ("height_rate", -1.2, 1.2),
    ("gyro_x", -6.0, 6.0),
    ("gyro_y", -6.0, 6.0),
    ("gyro_z", -12.0, 12.0),
    ("grav_x", -0.75, 0.75),
    ("grav_y", -0.75, 0.75),
    ("grav_z", -1.05, -0.50),
    ("q_err_hip_L", -0.60, 0.60),
    ("q_err_hip_R", -0.60, 0.60),
    ("q_err_hipknee_L", -0.60, 0.60),
    ("q_err_hipknee_R", -0.60, 0.60),
    ("dq_hip_L_x0.1", -1.5, 1.5),
    ("dq_hip_R_x0.1", -1.5, 1.5),
    ("dq_hipknee_L_x0.1", -1.5, 1.5),
    ("dq_hipknee_R_x0.1", -1.5, 1.5),
    ("dq_wheel_L_x0.1", -5.5, 5.5),
    ("dq_wheel_R_x0.1", -5.5, 5.5),
    ("last_act_hip_L", -10.0, 10.0),
    ("last_act_hip_R", -10.0, 10.0),
    ("last_act_hipknee_L", -10.0, 10.0),
    ("last_act_hipknee_R", -10.0, 10.0),
    ("last_act_wheel_L", -10.0, 10.0),
    ("last_act_wheel_R", -10.0, 10.0),
    ("leg_h_L_x5", 0.75, 1.75),
    ("leg_h_R_x5", 0.75, 1.75),
)


def _estimate_lag_ms(ref: np.ndarray, meas: np.ndarray, dt_s: float, max_lag_steps: int = 8) -> tuple[float | None, float | None]:
    """Estimate delay (meas lagging ref) in milliseconds via normalized cross-correlation."""
    ref = np.asarray(ref, dtype=np.float64)
    meas = np.asarray(meas, dtype=np.float64)
    if len(ref) < 20 or np.std(ref) < 0.2 or np.std(meas) < 0.2:
        return None, None
    r0 = ref - np.mean(ref)
    m0 = meas - np.mean(meas)
    best_lag = 0
    best_corr = -2.0
    for lag in range(0, min(max_lag_steps + 1, len(ref) // 3)):
        a = r0[: len(ref) - lag] if lag > 0 else r0
        b = m0[lag:] if lag > 0 else m0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-9:
            continue
        corr = float(np.dot(a, b) / denom)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    if best_corr < -1.0:
        return None, None
    return best_lag * dt_s * 1e3, best_corr


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 5 or np.std(a) < 1e-4 or np.std(b) < 1e-4:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze_run(
    log_dir: Path,
    *,
    replay_bundle: Path | None = None,
    write_html: bool = True,
) -> dict[str, Any]:
    log_dir = Path(log_dir).resolve()
    ds_path = log_dir / "dataset.npz"
    if not ds_path.is_file():
        raise FileNotFoundError(f"dataset.npz not found in {log_dir}")

    meta = json.loads((log_dir / "meta.json").read_text(encoding="utf-8")) if (log_dir / "meta.json").is_file() else {}
    summary = json.loads((log_dir / "summary.json").read_text(encoding="utf-8")) if (log_dir / "summary.json").is_file() else {}
    dt_s = float(meta.get("control_dt", 0.02))

    with np.load(ds_path) as ds:
        t_s = ds["t_s"]
        obs_policy = ds["obs_policy"]
        obs_history = ds["obs_history"]
        actions = ds["actions"]
        q_ref = ds["q_ref_fw"]
        v_ref = ds["v_ref_fw"]
        tau_exp = ds["tau_pd_expected_fw"]
        tau_meas = ds["tau_meas_fw"]
        leg_mcu = ds["leg_height_mcu"]
        leg_cad = ds["leg_height_cad"]
        euler_deg = ds["euler_deg"]
        gyro = ds["gyro"]
        gravity = ds["gravity"]
        cmds = ds["commands"]
        dq_sim = ds["dq_sim_unified"]
        rl_active = ds["rl_active_mask"].astype(bool)
        fault_flags = ds["fault_flags"]

    obs500_stats: dict[str, Any] = {}
    obs500_path = log_dir / "obs_500hz.npz"
    if obs500_path.is_file():
        with np.load(obs500_path) as o5:
            t5 = o5["t_rx_s"]
            if len(t5) >= 2:
                dt5 = np.diff(t5) * 1e3
                obs500_stats = {
                    "frames": int(len(t5)),
                    "avg_rate_hz": round(float((len(t5) - 1) / max(t5[-1] - t5[0], 1e-6)), 1),
                    "dt_ms_mean": round(float(np.mean(dt5)), 3),
                    "dt_ms_p99": round(float(np.percentile(dt5, 99)), 3),
                    "total_lost": int(np.sum(o5["lost_before"])),
                }

    # Analyze active RL segment if available, otherwise fall back to full run (e.g., dry-run)
    mask = rl_active if np.any(rl_active) else np.ones(len(t_s), dtype=bool)
    scope = "rl_active" if np.any(rl_active) else "all_frames_shadow"

    # 1. 28D Observation OOD diagnostics
    ood_alerts: list[str] = []
    obs_channels: list[dict[str, Any]] = []
    obs_sub = obs_policy[mask]
    for i, (name, lo, hi) in enumerate(OBS_28_SPEC):
        col = obs_sub[:, i]
        finite_frac = float(np.mean(np.isfinite(col)))
        c_min = float(np.nanmin(col))
        c_max = float(np.nanmax(col))
        c_mean = float(np.nanmean(col))
        c_std = float(np.nanstd(col))
        oob = (c_min < lo) or (c_max > hi) or (finite_frac < 1.0)
        obs_channels.append({
            "idx": i,
            "name": name,
            "min": round(c_min, 4),
            "mean": round(c_mean, 4),
            "std": round(c_std, 4),
            "max": round(c_max, 4),
            "expected_range": [lo, hi],
            "ood": oob,
        })
        if oob:
            ood_alerts.append(f"obs[{i}] {name}: [{c_min:.3f}, {c_max:.3f}] 超出期望区间 [{lo}, {hi}]")

    grav_norm = np.linalg.norm(gravity[mask], axis=1)
    if np.max(np.abs(grav_norm - 1.0)) > 0.05:
        ood_alerts.append(f"重力向量模长异常: [{float(np.min(grav_norm)):.3f}, {float(np.max(grav_norm)):.3f}]")

    # 2. Actuator SysID (PD expected vs measured torque, wheel lag, saturation)
    leg_lim = float(meta.get("leg_torque_limit", 40.0))
    wheel_lim = float(meta.get("wheel_torque_limit", 5.0))
    actuator_sysid: dict[str, Any] = {}
    recommendations: list[str] = []

    for i, mname in enumerate(MOTOR_NAMES):
        te = tau_exp[mask, i]
        tm = tau_meas[mask, i]
        err = te - tm
        corr = _safe_corr(te, tm)
        lim = leg_lim if i < 4 else wheel_lim
        sat_pct = float(100.0 * np.mean(np.abs(tm) >= 0.95 * lim))
        entry = {
            "tau_mae_nm": round(float(np.mean(np.abs(err))), 3),
            "tau_rmse_nm": round(float(np.sqrt(np.mean(err * err))), 3),
            "tau_max_meas_nm": round(float(np.max(np.abs(tm))), 3),
            "tau_exp_meas_corr": round(corr, 3) if corr is not None else None,
            "saturation_percent": round(sat_pct, 2),
        }
        actuator_sysid[mname] = entry
        if scope == "rl_active" and corr is not None and corr < 0.5:
            recommendations.append(
                f"{mname}: 预期 PD 力矩与实测相关性仅 {corr:.2f} (MAE={entry['tau_mae_nm']} N·m)，请优先排查方向/零位/堵转"
            )
        if sat_pct > 15.0:
            recommendations.append(
                f"{mname}: 力矩饱和占比 {sat_pct:.1f}%，建议在下一轮训练加强力矩惩罚或检查实车机械阻尼/偏置"
            )

    # Wheel velocity tracking lag estimation -> recommends wheel_target_delay_steps
    wheel_lags_ms = []
    wheel_lag_detail: dict[str, Any] = {}
    for w_i, (fw_idx, p_idx, wname) in enumerate(((4, 4, "left_wheel"), (5, 5, "right_wheel"))):
        del w_i
        lag_ms, corr = _estimate_lag_ms(v_ref[mask, fw_idx], dq_sim[mask, p_idx], dt_s)
        wheel_lag_detail[wname] = {
            "estimated_lag_ms": round(lag_ms, 1) if lag_ms is not None else None,
            "corr": round(corr, 3) if corr is not None else None,
        }
        if lag_ms is not None and corr is not None and corr > 0.5:
            wheel_lags_ms.append(lag_ms)
    rec_delay_steps = int(round(float(np.mean(wheel_lags_ms)) / 2.0)) if wheel_lags_ms else None
    if rec_delay_steps is not None:
        recommendations.append(
            f"轮速响应估计滞后 ≈ {float(np.mean(wheel_lags_ms)):.1f} ms -> 下一轮训练建议参考 wheel_target_delay_steps = {rec_delay_steps} (2ms步)"
        )

    # 3. Leg height comparison (MCU VMC vs V0.2 CAD FK)
    cad_valid = np.isfinite(leg_cad[mask]).all(axis=1)
    if np.any(cad_valid):
        diff_h = leg_mcu[mask][cad_valid] - leg_cad[mask][cad_valid]
        leg_height_cmp = {
            "valid_fraction": round(float(np.mean(cad_valid)), 3),
            "mean_diff_cm_L": round(float(np.mean(diff_h[:, 0]) * 100.0), 2),
            "mean_diff_cm_R": round(float(np.mean(diff_h[:, 1]) * 100.0), 2),
            "rmse_cm": round(float(np.sqrt(np.mean(diff_h * diff_h)) * 100.0), 2),
        }
        if leg_height_cmp["rmse_cm"] > 2.5:
            recommendations.append(
                f"MCU VMC 腿高与 V0.2 CAD 正解 RMSE = {leg_height_cmp['rmse_cm']:.2f} cm "
                f"(L={leg_height_cmp['mean_diff_cm_L']:+.2f}, R={leg_height_cmp['mean_diff_cm_R']:+.2f} cm)，"
                "建议标定 leg_zero_offset_rad 或切换 leg_height_source='cad'"
            )
    else:
        leg_height_cmp = {"valid_fraction": 0.0, "note": "CAD FK grid out-of-range or absent"}

    # 4. Optional shadow replay of a new candidate policy on recorded real obs_history
    replay_result: dict[str, Any] | None = None
    if replay_bundle is not None:
        cand = PolicyBundle(replay_bundle)
        if cand.input_dim != obs_history.shape[1]:
            raise ValueError(
                f"Replay bundle expects {cand.input_dim}D input, but dataset has {obs_history.shape[1]}D"
            )
        new_acts = np.stack([np.clip(cand.forward(obs_history[k]), -10.0, 10.0) for k in range(len(t_s))])
        diff = new_acts[mask] - actions[mask]
        old_jerk = float(np.mean(np.abs(np.diff(actions[mask], axis=0)))) if np.sum(mask) > 1 else 0.0
        new_jerk = float(np.mean(np.abs(np.diff(new_acts[mask], axis=0)))) if np.sum(mask) > 1 else 0.0
        replay_result = {
            "candidate_bundle": str(cand.bundle_dir),
            "candidate_checkpoint_sha256": cand.meta.get("checkpoint_sha256"),
            "action_mae_per_joint": {
                POLICY_JOINT_NAMES[j]: round(float(np.mean(np.abs(diff[:, j]))), 4) for j in range(6)
            },
            "action_max_diff": round(float(np.max(np.abs(diff))), 4),
            "old_action_step_delta_mean": round(old_jerk, 4),
            "new_action_step_delta_mean": round(new_jerk, 4),
        }

    report: dict[str, Any] = {
        "log_dir": str(log_dir),
        "analysis_scope": scope,
        "frames_total": int(len(t_s)),
        "frames_rl_active": int(np.sum(rl_active)),
        "duration_s": round(float(t_s[-1] - t_s[0]), 2) if len(t_s) >= 2 else 0.0,
        "fault_frames": int(np.sum(fault_flags != 0)),
        "attitude_peaks_deg": {
            "pitch_abs_max": round(float(np.max(np.abs(euler_deg[mask, 1]))), 2),
            "roll_abs_max": round(float(np.max(np.abs(euler_deg[mask, 0]))), 2),
        },
        "obs_500hz": obs500_stats,
        "timing_summary": {
            "infer_ms": summary.get("infer_ms", {}),
            "loop_dt_ms": summary.get("loop_dt_ms", {}),
            "obs_age_ms": summary.get("obs_age_ms", {}),
            "link_stats": summary.get("link_stats", {}),
        },
        "leg_height_comparison": leg_height_cmp,
        "wheel_lag_sysid": wheel_lag_detail,
        "recommended_wheel_target_delay_steps": rec_delay_steps,
        "actuator_sysid": actuator_sysid,
        "ood_alerts": ood_alerts,
        "obs_28_channels": obs_channels,
        "recommendations": recommendations,
        "policy_replay": replay_result,
    }

    (log_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if write_html:
        _write_report_html(log_dir, report, t_s, euler_deg, gyro, cmds, leg_mcu, leg_cad, tau_meas, tau_exp, v_ref, dq_sim, actions, leg_lim, wheel_lim)

    return report


def _downsample_indices(n: int, max_points: int = 2500) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points, dtype=int)


def _to_json_list(arr: np.ndarray) -> list[float | None]:
    return [None if not math.isfinite(float(v)) else round(float(v), 5) for v in arr]


def _write_report_html(
    log_dir: Path,
    report: dict[str, Any],
    t_s: np.ndarray,
    euler_deg: np.ndarray,
    gyro: np.ndarray,
    cmds: np.ndarray,
    leg_mcu: np.ndarray,
    leg_cad: np.ndarray,
    tau_meas: np.ndarray,
    tau_exp: np.ndarray,
    v_ref: np.ndarray,
    dq_sim: np.ndarray,
    actions: np.ndarray,
    leg_lim: float,
    wheel_lim: float,
) -> None:
    idx = _downsample_indices(len(t_s), 2500)
    series = {
        "pitch_deg": _to_json_list(euler_deg[idx, 1]),
        "roll_deg": _to_json_list(euler_deg[idx, 0]),
        "wx": _to_json_list(gyro[idx, 0]),
        "wy": _to_json_list(gyro[idx, 1]),
        "wz": _to_json_list(gyro[idx, 2]),
        "cmd_vx": _to_json_list(cmds[idx, 0]),
        "cmd_wz": _to_json_list(cmds[idx, 1]),
        "cmd_height": _to_json_list(cmds[idx, 2]),
        "leg_h_mcu_L": _to_json_list(leg_mcu[idx, 0]),
        "leg_h_mcu_R": _to_json_list(leg_mcu[idx, 1]),
        "leg_h_cad_L": _to_json_list(leg_cad[idx, 0]),
        "leg_h_cad_R": _to_json_list(leg_cad[idx, 1]),
        "v_ref_L": _to_json_list(v_ref[idx, 4]),
        "v_meas_L": _to_json_list(dq_sim[idx, 4]),
        "v_ref_R": _to_json_list(v_ref[idx, 5]),
        "v_meas_R": _to_json_list(dq_sim[idx, 5]),
    }
    for i, m in enumerate(MOTOR_NAMES):
        series[f"tau_meas_{m}"] = _to_json_list(tau_meas[idx, i])
        series[f"tau_exp_{m}"] = _to_json_list(tau_exp[idx, i])
    for i, p in enumerate(POLICY_JOINT_NAMES):
        series[f"act_{p}"] = _to_json_list(actions[idx, i])

    plots = [
        ("姿态 [deg]", ["pitch_deg", "roll_deg"]),
        ("机体角速度与指令 [rad/s]", ["wx", "wy", "wz", "cmd_wz"]),
        ("腿高 MCU vs CAD [m]", ["cmd_height", "leg_h_mcu_L", "leg_h_mcu_R", "leg_h_cad_L", "leg_h_cad_R"]),
        ("腿关节实测力矩 [N·m]", [f"tau_meas_{m}" for m in MOTOR_NAMES[:4]]),
        ("腿关节预期 PD 力矩 [N·m]", [f"tau_exp_{m}" for m in MOTOR_NAMES[:4]]),
        ("驱动轮力矩 实测 vs 预期 [N·m]", ["tau_meas_left_wheel", "tau_meas_right_wheel", "tau_exp_left_wheel", "tau_exp_right_wheel"]),
        ("轮速 目标 vs 实测 [rad/s]", ["v_ref_L", "v_meas_L", "v_ref_R", "v_meas_R"]),
        ("策略原始动作", [f"act_{p}" for p in POLICY_JOINT_NAMES]),
    ]
    payload = {
        "title": f"Deploy Report — {log_dir.name}",
        "t": _to_json_list(t_s[idx]),
        "series": series,
        "plots": plots,
        "limits": {"leg": leg_lim, "wheel": wheel_lim},
        "meta_summary": {
            "scope": report["analysis_scope"],
            "duration_s": report["duration_s"],
            "frames_rl_active": report["frames_rl_active"],
            "fault_frames": report["fault_frames"],
            "pitch_peak_deg": report["attitude_peaks_deg"]["pitch_abs_max"],
            "roll_peak_deg": report["attitude_peaks_deg"]["roll_abs_max"],
            "recommended_wheel_delay_steps": report["recommended_wheel_target_delay_steps"],
        },
        "ood_alerts": report["ood_alerts"],
        "recommendations": report["recommendations"],
    }
    (log_dir / "report.html").write_text(
        _REPORT_HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False)),
        encoding="utf-8",
    )


def print_summary(report: dict[str, Any]) -> None:
    print(f"=== Deploy Analysis: {Path(report['log_dir']).name} ({report['analysis_scope']}) ===")
    print(
        f"  Duration: {report['duration_s']:.2f}s | 50Hz steps: {report['frames_total']} "
        f"(RL active: {report['frames_rl_active']}) | fault_frames: {report['fault_frames']}"
    )
    print(
        f"  Attitude peaks: |pitch|={report['attitude_peaks_deg']['pitch_abs_max']:.1f}° "
        f"|roll|={report['attitude_peaks_deg']['roll_abs_max']:.1f}°"
    )
    if report["obs_500hz"]:
        o5 = report["obs_500hz"]
        print(
            f"  500Hz RX: {o5['frames']} frames ({o5['avg_rate_hz']} Hz, "
            f"dt p99={o5['dt_ms_p99']} ms, lost={o5['total_lost']})"
        )
    print(f"  Leg height MCU vs CAD: {report['leg_height_comparison']}")
    print(f"  Wheel lag SysID: {report['wheel_lag_sysid']} -> rec delay steps: {report['recommended_wheel_target_delay_steps']}")
    if report["ood_alerts"]:
        print("  [OOD ALERTS]")
        for a in report["ood_alerts"]:
            print(f"    - {a}")
    if report["recommendations"]:
        print("  [RECOMMENDATIONS FOR NEXT TRAINING / CALIBRATION]")
        for r in report["recommendations"]:
            print(f"    - {r}")
    if report["policy_replay"]:
        rp = report["policy_replay"]
        print(
            f"  [POLICY REPLAY] {Path(rp['candidate_bundle']).name}: "
            f"max_diff={rp['action_max_diff']:.4f}, "
            f"step_delta {rp['old_action_step_delta_mean']:.4f} -> {rp['new_action_step_delta_mean']:.4f}"
        )
    print(f"  Report written -> {Path(report['log_dir']) / 'report.html'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="Recorded deploy log directory")
    parser.add_argument("--replay-bundle", type=Path, help="Optional candidate policy bundle to replay on recorded obs_history")
    parser.add_argument("--no-html", action="store_true", help="Skip generating report.html")
    args = parser.parse_args()

    report = analyze_run(args.log_dir, replay_bundle=args.replay_bundle, write_html=not args.no_html)
    print_summary(report)
    return 0


_REPORT_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>Deploy Report</title>
<style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:16px;background:#111;color:#ddd}
h1{font-size:19px;margin-bottom:8px} h2{font-size:14px;color:#81d4fa;margin:12px 0 4px}
.kv{display:flex;flex-wrap:wrap;gap:14px;background:#1b1b1b;padding:10px 14px;border-radius:6px;font-size:13px}
.kv span b{color:#fff} ul{margin:4px 0 10px;font-size:13px}
.warn{color:#ffb74d} .plot{margin:8px 0} canvas{background:#1b1b1b;width:100%;height:175px;border-radius:4px}
.legend span{margin-right:12px;font-size:12px}
</style></head><body>
<h1 id="title"></h1>
<div class="kv" id="summary"></div>
<h2>OOD 与对齐告警</h2><ul id="ood"></ul>
<h2>下一轮训练与标定建议</h2><ul id="recs"></ul>
<div id="plots"></div>
<script>
const D = __DATA__;
const colors = ["#4fc3f7","#ffb74d","#81c784","#e57373","#ba68c8","#fff176"];
document.getElementById("title").textContent = D.title;
const sumEl = document.getElementById("summary");
for (const [k, v] of Object.entries(D.meta_summary)) {
  const s = document.createElement("span"); s.innerHTML = `${k}: <b>${v}</b>`; sumEl.appendChild(s);
}
const fillList = (id, items, empty) => {
  const el = document.getElementById(id);
  if (!items.length) { const li = document.createElement("li"); li.textContent = empty; el.appendChild(li); return; }
  for (const x of items) { const li = document.createElement("li"); li.className = "warn"; li.textContent = x; el.appendChild(li); }
};
fillList("ood", D.ood_alerts, "全部 28 维观测均在期望区间内");
fillList("recs", D.recommendations, "暂未检出明显硬件偏差或饱和问题");
const t = D.t, plotsEl = document.getElementById("plots");
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
  const lim = label.startsWith("腿关节") ? D.limits.leg : label.startsWith("驱动轮力矩") ? D.limits.wheel : null;
  if (lim) { lo = Math.min(lo, -lim * 1.05); hi = Math.max(hi, lim * 1.05); }
  if (!isFinite(lo)) { lo = -1; hi = 1; } if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
  const x = v => pad + (v - t[0]) / Math.max(t[t.length - 1] - t[0], 1e-6) * (w - pad * 1.2);
  const y = v => h - 14 - (v - lo) / (hi - lo) * (h - 28);
  g.strokeStyle = "#333"; g.fillStyle = "#888"; g.font = `${11 * devicePixelRatio}px sans-serif`;
  for (const v of [lo, (lo + hi) / 2, hi]) { g.beginPath(); g.moveTo(pad, y(v)); g.lineTo(w, y(v)); g.stroke(); g.fillText(v.toFixed(2), 2, y(v) + 4); }
  if (lim) { g.strokeStyle = "#a33"; g.setLineDash([6, 4]); for (const v of [lim, -lim]) { g.beginPath(); g.moveTo(pad, y(v)); g.lineTo(w, y(v)); g.stroke(); } g.setLineDash([]); }
  names.forEach((n, i) => {
    g.strokeStyle = colors[i % colors.length]; g.lineWidth = 1.5 * devicePixelRatio; g.beginPath(); let pen = false;
    D.series[n].forEach((v, k) => { if (v === null) { pen = false; return; } pen ? g.lineTo(x(t[k]), y(v)) : g.moveTo(x(t[k]), y(v)); pen = true; });
    g.stroke();
  });
  g.fillStyle = "#888"; g.fillText(`t ${t[0].toFixed(2)}s`, pad, h - 2); g.fillText(`${t[t.length - 1].toFixed(2)}s`, w - 58 * devicePixelRatio, h - 2);
}
</script></body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
