#!/usr/bin/env python3
"""STM32H7 USB CDC 50 Hz RL policy host runner with Mode 2 handover and async logging."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys
import time

HOST_DIR = Path(__file__).resolve().parent
DEPLOY_TOOLS_DIR = HOST_DIR.parent / "wheel_leg_deploy" / "Tools"
for p in (HOST_DIR, DEPLOY_TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from telemetry_capture import RL_FLAG_ZERO_TORQUE  # noqa: E402

from deploy_logger import ControlLogItem, DeployLogger, decode_fault_flags, mode_name  # noqa: E402
from policy_runtime import CalibrationConfig, PolicyBundle, WheelLegPolicyController  # noqa: E402
from usb_link import ByteTransport, SerialPortTransport, UsbLink  # noqa: E402


class _HighResTimer:
    """Request 1 ms Windows multimedia timer resolution (no-op elsewhere)."""

    def __init__(self) -> None:
        self._winmm = None

    def __enter__(self) -> "_HighResTimer":
        if sys.platform == "win32":
            try:
                import ctypes  # local import so sandboxed tests never fail

                winmm = ctypes.WinDLL("winmm")
                if winmm.timeBeginPeriod(1) == 0:
                    self._winmm = winmm
            except Exception:
                self._winmm = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._winmm is not None:
            try:
                self._winmm.timeEndPeriod(1)
            except Exception:
                pass
            self._winmm = None


def _sleep_until_ns(target_ns: int) -> None:
    while True:
        remaining_ns = target_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 2_000_000:
            time.sleep((remaining_ns - 1_500_000) * 1e-9)
        elif remaining_ns > 200_000:
            time.sleep(0.0)


def run_host_loop(
    transport: ByteTransport,
    bundle: PolicyBundle,
    calibration: CalibrationConfig,
    logger: DeployLogger,
    *,
    dry_run: bool = False,
    gain_scale: float = 1.0,
    duration_s: float = 0.0,
    obs_stale_ms: float = 50.0,
    print_status: bool = True,
) -> dict:
    """Run the 500 Hz RX + 50 Hz Policy/TX + Async Logging closed loop."""
    controller = WheelLegPolicyController(
        bundle,
        calibration,
        dry_run=dry_run,
        gain_scale=gain_scale,
    )
    link = UsbLink(
        transport,
        on_raw_bytes=logger.log_raw_bytes,
        on_observation=logger.log_obs_500hz,
    )
    link.start()

    period_ns = round(bundle.control_dt * 1e9)
    start_perf = time.perf_counter_ns()
    next_tick_ns = start_perf
    prev_tick_ns = start_perf
    next_status_s = 1.0
    step_idx = 0
    zeros = (0.0,) * 6

    try:
        with _HighResTimer():
            while True:
                now_perf = time.perf_counter_ns()
                elapsed_s = (now_perf - start_perf) * 1e-9
                if duration_s > 0.0 and elapsed_s >= duration_s:
                    break

                loop_dt_ms = (now_perf - prev_tick_ns) * 1e-6 if step_idx > 0 else bundle.control_dt * 1e3
                prev_tick_ns = now_perf

                obs, _ = link.get_latest_observation()
                now_rx_clock_ns = time.perf_counter_ns()

                if obs is None:
                    # No observation yet: keep sending zero-torque heartbeat if link alive
                    link.send_command(
                        torque=zeros,
                        kp=zeros,
                        kd=zeros,
                        q_ref=zeros,
                        v_ref=zeros,
                        source_seq=0,
                        source_timestamp_ms=0,
                        rl_flags=RL_FLAG_ZERO_TORQUE,
                    )
                else:
                    obs_age_ms = max(0.0, (now_rx_clock_ns - obs.received_monotonic_ns) * 1e-6)
                    stale = obs_age_ms > obs_stale_ms
                    result = controller.step(obs, dt=bundle.control_dt)
                    tx_flag = RL_FLAG_ZERO_TORQUE if stale else result.rl_flags_tx
                    if stale:
                        controller.reset_for_handover(obs)

                    cmd_seq, _, tx_ok = link.send_command(
                        torque=result.torque_ff_fw,
                        kp=result.kp_fw,
                        kd=result.kd_fw,
                        q_ref=result.q_ref_fw,
                        v_ref=result.v_ref_fw,
                        source_seq=obs.seq,
                        source_timestamp_ms=obs.timestamp_ms,
                        rl_flags=tx_flag,
                    )

                    logger.log_control_step(
                        ControlLogItem(
                            t_s=elapsed_s,
                            step=step_idx,
                            obs=obs,
                            result=result,
                            cmd_seq=cmd_seq,
                            cmd_tx_ok=tx_ok,
                            obs_age_ms=obs_age_ms,
                            loop_dt_ms=loop_dt_ms,
                        )
                    )
                    step_idx += 1

                    if print_status and elapsed_s >= next_status_s:
                        st = link.stats()
                        rx_hz = st.rx_frames / max(elapsed_s, 1e-3)
                        faults = ",".join(decode_fault_flags(obs.fault_flags)) or "NONE"
                        print(
                            f"t={elapsed_s:6.1f}s mode={mode_name(obs.chassis_mode):<6} "
                            f"rl={obs.rl_flags}->{tx_flag} "
                            f"pitch={result.pitch_deg:+5.1f}° roll={result.roll_deg:+5.1f}° "
                            f"h_mcu=({result.leg_height_mcu[0]:.3f},{result.leg_height_mcu[1]:.3f}) "
                            f"h_cad=({result.leg_height_cad[0]:.3f},{result.leg_height_cad[1]:.3f}) "
                            f"rx={rx_hz:5.1f}Hz loss={st.loss_percent:4.2f}% "
                            f"infer={result.infer_ms:4.2f}ms dt={loop_dt_ms:4.1f}ms "
                            f"fault={faults}",
                            flush=True,
                        )
                        next_status_s = elapsed_s + 1.0

                if not link.stats().connected:
                    logger.log_event("LINK_LOST", f"USB link disconnected: {link.stats().last_error}", t_s=elapsed_s)
                    break

                next_tick_ns = max(time.perf_counter_ns() - period_ns, next_tick_ns + period_ns)
                _sleep_until_ns(next_tick_ns)
    except KeyboardInterrupt:
        logger.log_event("USER_STOP", "Ctrl+C received, sending zero-torque burst")
    finally:
        link.stop(send_zero_burst=True)
        summary = logger.close(link_stats=asdict(link.stats()))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Exported policy bundle directory")
    parser.add_argument("--port", required=True, help="STM32H7 USB CDC port (e.g. COM8 or /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate (default: 921600)")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=HOST_DIR / "configs" / "joint_calibration.json",
        help="Joint mapping & calibration JSON",
    )
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy", help="Inference backend")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Shadow mode: run full inference & logging but ALWAYS transmit zero-torque commands",
    )
    parser.add_argument(
        "--gain-scale",
        type=float,
        default=1.0,
        help="Scale factor for PD gains in [0.0, 1.0] (e.g. 0.3 for hanging-rack test)",
    )
    parser.add_argument("--leg-height-source", choices=("mcu", "cad"), help="Override calibration leg_height_source")
    parser.add_argument("--heading-hold", action="store_true", help="Enable host-side heading hold")
    parser.add_argument("--duration", type=float, default=0.0, help="Run seconds; 0 = until Ctrl+C")
    parser.add_argument("--obs-stale-ms", type=float, default=50.0, help="Max observation age before zeroing TX (ms)")
    parser.add_argument("--log-dir", type=Path, help="Explicit output log directory")
    parser.add_argument("--tag", default="", help="Optional suffix tag for automatic log directory name")
    parser.add_argument("--no-raw-bin", action="store_true", help="Disable raw_usb.bin byte dump")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.gain_scale <= 1.0:
        raise SystemExit("--gain-scale must be in [0.0, 1.0]")

    bundle = PolicyBundle(args.bundle, backend=args.backend)
    cal = CalibrationConfig.from_json(args.calibration)
    if args.leg_height_source or args.heading_hold:
        cal_dict = cal.to_dict()
        if args.leg_height_source:
            cal_dict["leg_height_source"] = args.leg_height_source
        if args.heading_hold:
            cal_dict["heading_hold"]["enabled"] = True
        cal = CalibrationConfig(
            policy_to_firmware_index=tuple(cal_dict["policy_to_firmware_index"]),
            joint_sign=tuple(cal_dict["joint_sign"]),
            leg_zero_offset_rad=tuple(cal_dict["leg_zero_offset_rad"]),
            leg_height_source=cal_dict["leg_height_source"],
            apply_leg_bias=cal_dict["apply_leg_bias"],
            apply_friction_circle=cal_dict["apply_friction_circle"],
            heading_hold_enabled=cal_dict["heading_hold"]["enabled"],
            heading_hold_gain=cal_dict["heading_hold"]["gain"],
            heading_hold_deadband_deg=cal_dict["heading_hold"]["deadband_deg"],
            heading_hold_max_yaw_rate=cal_dict["heading_hold"]["max_yaw_rate"],
            leg_kp=tuple(cal_dict["pd_gains"]["leg_kp"]),
            leg_kd=tuple(cal_dict["pd_gains"]["leg_kd"]),
            wheel_kd=tuple(cal_dict["pd_gains"]["wheel_kd"]),
            leg_torque_limit=cal_dict["torque_limits_nm"]["leg"],
            wheel_torque_limit=cal_dict["torque_limits_nm"]["wheel"],
        )

    tag = args.tag or ("dryrun" if args.dry_run else f"scale{int(round(args.gain_scale * 100))}")
    log_dir = args.log_dir or (HOST_DIR / "logs" / f"{datetime.now():%Y%m%d_%H%M%S}_{tag}")

    meta = {
        "port": args.port,
        "baud": args.baud,
        "backend": args.backend,
        "dry_run": args.dry_run,
        "gain_scale": args.gain_scale,
        "obs_stale_ms": args.obs_stale_ms,
        "bundle_dir": str(bundle.bundle_dir),
        "checkpoint_path": bundle.meta.get("checkpoint_path"),
        "checkpoint_sha256": bundle.meta.get("checkpoint_sha256"),
        "model_xml_sha256": bundle.meta.get("model_xml_sha256"),
        "calibration_path": str(args.calibration.resolve()),
        "calibration": cal.to_dict(),
    }

    print(
        f"[host] Starting RL host runner: port={args.port} bundle={bundle.bundle_dir.name} "
        f"backend={args.backend} dry_run={args.dry_run} gain_scale={args.gain_scale:g}\n"
        f"[host] Logging to: {log_dir}",
        flush=True,
    )
    transport = SerialPortTransport(args.port, args.baud)
    logger = DeployLogger(
        log_dir,
        meta,
        control_dt=bundle.control_dt,
        leg_torque_limit=cal.leg_torque_limit,
        wheel_torque_limit=cal.wheel_torque_limit,
        record_raw_bin=not args.no_raw_bin,
    )
    summary = run_host_loop(
        transport,
        bundle,
        cal,
        logger,
        dry_run=args.dry_run,
        gain_scale=args.gain_scale,
        duration_s=args.duration,
        obs_stale_ms=args.obs_stale_ms,
        print_status=True,
    )
    print(
        f"\n[host] Session ended: elapsed={summary.get('elapsed_seconds', 0):.1f}s "
        f"rl_active={summary.get('rl_active_seconds', 0):.1f}s "
        f"incidents={summary.get('incidents', 0)}\n"
        f"[host] Analyze with: python {HOST_DIR / 'analyze_deploy_log.py'} {log_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
