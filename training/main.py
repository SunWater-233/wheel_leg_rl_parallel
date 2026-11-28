"""Run a Genesis demo or train a wheel-leg PPO policy."""

import argparse
from contextlib import ExitStack
from datetime import datetime
from hashlib import sha256
from math import isfinite
from os import environ
from pathlib import Path

import genesis as gs

from configs.schema import MAX_FORWARD_SPEED, SNAPSHOT_NAME, TRAINING_DIR, load_config, load_snapshot
from rl.checkpoint import add_actor_history_inputs, add_actor_speed_input, load_checkpoint
from rl.evaluator import attach_local_view, demo, drive_target, scripted_actions
from rl.runner import RobotRunner
from rl.trainer import train
from robots.scut.adapter import WheelbipeFlatEnv
from robots.v02.adapter import WheelLegFlatEnv
from tasks.locomotion_env import WheelLegLocomotionEnv
from tasks.terrains import (
    SCUT_GROUND_COMPOUND_SURFACES,
    SCUT_GROUND_SINGLE_SIDE_SURFACES,
    SCUT_GROUND_SPEED_BUMP,
    SCUT_V1_SURFACES,
    TERRAIN_LANES,
)
from viz.live_view import TrainingView

__all__ = [
    "RobotRunner",
    "add_actor_history_inputs",
    "add_actor_speed_input",
    "attach_local_view",
    "build_parser",
    "demo",
    "drive_target",
    "load_checkpoint",
    "main",
    "scripted_actions",
    "train",
]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("demo", "train"), default="demo")
    parser.add_argument("--num-envs", type=int, help="Default: 4 for demo and 64 for training")
    parser.add_argument("--backend", choices=("cuda", "cpu"), help="Simulation device (default: cuda)")
    parser.add_argument("--robot", choices=("v0.2", "scut"), help="Robot to simulate (default: v0.2)")
    parser.add_argument("--model", type=Path, help="Override the MJCF file for the selected robot")
    parser.add_argument("--seed", type=int, help="Random seed (default: 42)")
    parser.add_argument("--config", type=Path, help="Load TOML task settings or a run snapshot")
    parser.add_argument("--seconds", type=float, help="Demo duration (default: 5 s, unlimited with --keyboard)")
    parser.add_argument("--speed", type=float, help="Demo forward speed in m/s (2.5 for native keyboard, 0.35 otherwise)")
    parser.add_argument("--accel", type=float, help="Keyboard forward acceleration in m/s^2 (default: 1.8 for --keyboard)")
    parser.add_argument("--brake-accel", type=float, help="Keyboard braking/reversal deceleration in m/s^2 (default: 3.0 for --keyboard)")
    parser.add_argument("--yaw-accel", type=float, help="Keyboard yaw angular acceleration in rad/s^2 (default: 8.0 for --keyboard; firmware MAX_WZ_ACC is 24.0)")
    parser.add_argument("--yaw-brake-accel", type=float, help="Keyboard yaw braking/reversal deceleration in rad/s^2 (default: same as --yaw-accel)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Demo yaw-rate request in rad/s")
    heights = parser.add_mutually_exclusive_group()
    heights.add_argument("--body-height", type=float, help="Demo base_link clearance above the ground in meters")
    heights.add_argument("--body-height-sequence", type=float, nargs="+", help="Successive demo body clearances in meters")
    heights.add_argument("--height", type=float, help="Demo hip-to-wheel height in meters")
    heights.add_argument("--height-sequence", type=float, nargs="+", help="Successive demo leg-height targets in meters")
    parser.add_argument("--height-hold", type=float, default=2.0, help="Seconds between demo height targets")
    parser.add_argument(
        "--surface",
        choices=(
            SCUT_V1_SURFACES + SCUT_GROUND_COMPOUND_SURFACES + SCUT_GROUND_SINGLE_SIDE_SURFACES
            + (SCUT_GROUND_SPEED_BUMP, "transitions")
        ),
        help="Spawn on this lane when replaying a terrain policy",
    )
    parser.add_argument("--terrain-variant", type=int, help="Select a 1-based lane difficulty when replaying")
    parser.add_argument("--terrain-seed", type=int, help="Override the mixed-terrain layout seed")
    parser.add_argument("--forward-limit", type=float, help="Override the training speed range with this symmetric limit")
    parser.add_argument("--viewer", action="store_true", help="Show the live Genesis scene in demo mode")
    parser.add_argument("--viewer-fps", type=int, help="Native Genesis viewer refresh rate (default: 30 for --keyboard, 60 otherwise)")
    parser.add_argument("--live-port", type=int, help="Serve training or demo on this loopback port")
    parser.add_argument("--live-fps", type=int, help="Live frame rate target (default: 8 for streamed training, 60 otherwise)")
    parser.add_argument("--live-stream", action="store_true", help="Stream training frames to the browser instead of opening Genesis")
    parser.add_argument("--live-telemetry", action="store_true", help="Use a Genesis window and a telemetry-only dashboard for demo")
    parser.add_argument("--keyboard", action="store_true", help="Drive a PPO demo in Genesis, or in the browser with --live-port")
    parser.add_argument(
        "--free-drive",
        action="store_true",
        help="Drive across all terrain lanes without a time limit in the native Genesis viewer",
    )
    parser.add_argument(
        "--fall-reset",
        action="store_true",
        help="Keyboard demo: reset on fall instead of the firmware RESCUE/GETUP self-righting",
    )
    sim_logs = parser.add_mutually_exclusive_group()
    sim_logs.add_argument(
        "--sim-log",
        type=Path,
        help="Demo run log directory (default: outputs/sim_logs/<timestamp>, always on in demo)",
    )
    sim_logs.add_argument("--no-sim-log", action="store_true", help="Disable the automatic demo run log")
    parser.add_argument("--video", type=Path, help="Save a replay of env 0 as an MP4")
    checkpoint_args = parser.add_mutually_exclusive_group()
    checkpoint_args.add_argument("--checkpoint", type=Path, help="Replay a PPO checkpoint in demo mode")
    checkpoint_args.add_argument("--resume", type=Path, help="Continue PPO training from a checkpoint in train mode")
    checkpoint_args.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Start a new run from compatible weights with a fresh optimizer",
    )
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--output", type=Path, help="New or empty directory for the PPO run")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    body_heights = args.body_height_sequence or ([] if args.body_height is None else [args.body_height])
    if args.mode != "train" and (args.resume or args.init_checkpoint or args.forward_limit is not None):
        parser.error("--resume, --init-checkpoint and --forward-limit require train mode")
    if args.mode == "train" and (args.viewer or args.video or args.checkpoint or args.height is not None or body_heights
                                 or args.height_sequence or args.surface or args.terrain_variant is not None or args.yaw):
        parser.error("Viewer, recording, checkpoint and manual geometry/command options require demo mode")
    if args.keyboard and (args.mode != "demo" or args.checkpoint is None):
        parser.error("--keyboard requires demo mode and --checkpoint")
    if args.live_telemetry and (args.mode != "demo" or args.live_port is None):
        parser.error("--live-telemetry requires demo mode and --live-port")
    native_keyboard = args.keyboard and (args.live_port is None or args.live_telemetry)
    if args.free_drive and not (native_keyboard and args.live_port is None):
        parser.error("--free-drive requires native --keyboard without --live-port")
    if args.live_telemetry:
        args.viewer = True
    if native_keyboard:
        args.viewer = True
    if args.speed is None:
        args.speed = 2.5 if native_keyboard else 0.35
    if args.keyboard:
        if args.accel is None:
            args.accel = 1.8
        if args.brake_accel is None:
            args.brake_accel = 3.0
        if args.yaw_accel is None:
            args.yaw_accel = 8.0
        if args.yaw_brake_accel is None:
            args.yaw_brake_accel = args.yaw_accel
    elif (
        args.accel is not None
        or args.brake_accel is not None
        or args.yaw_accel is not None
        or args.yaw_brake_accel is not None
    ):
        parser.error("--accel, --brake-accel, --yaw-accel and --yaw-brake-accel require --keyboard")
    for name, value in (
        ("--accel", args.accel),
        ("--brake-accel", args.brake_accel),
        ("--yaw-accel", args.yaw_accel),
        ("--yaw-brake-accel", args.yaw_brake_accel),
    ):
        if value is not None and (not isfinite(value) or value <= 0.0):
            parser.error(f"{name} must be finite and positive")
    if args.keyboard and args.video and args.seconds is None:
        parser.error("Recording a keyboard demo requires a finite --seconds")
    if not isfinite(args.height_hold) or args.height_hold <= 0:
        parser.error("--height-hold must be finite and positive")
    if args.seconds is None and not args.keyboard:
        sequence = args.body_height_sequence or args.height_sequence
        args.seconds = len(sequence) * args.height_hold if sequence else 5.0

    checkpoint = args.resume or args.checkpoint or args.init_checkpoint
    run = {}
    config_path = args.config
    settings = None
    checkpoint_settings = None
    try:
        if checkpoint is not None:
            if not checkpoint.is_file():
                raise ValueError(f"Checkpoint not found: {checkpoint}")
            snapshot = checkpoint.with_name(SNAPSHOT_NAME)
            if snapshot.is_file():
                settings, run = load_snapshot(snapshot)
                checkpoint_settings = settings
                if not isinstance(run, dict):
                    raise ValueError(f"Expected run metadata in {snapshot}")
        if config_path is not None or settings is None:
            settings = load_config(config_path)
            if args.mode == "demo" and checkpoint_settings is not None:
                src = checkpoint_settings["task"]["locomotion"]
                dst = settings["task"]["locomotion"]
                if src["enabled"] and dst["enabled"]:
                    for key in ("actor_history_steps", "actor_true_forward_speed", "critic_terrain_observations"):
                        dst[key] = src[key]
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if "robot" in run and args.robot is not None and args.robot != run["robot"]:
        parser.error(f"Checkpoint uses --robot {run['robot']}; selected --robot {args.robot}")
    args.robot = args.robot or run.get("robot", "v0.2")
    args.backend = args.backend or run.get("backend", "cuda")
    args.seed = args.seed if args.seed is not None else run.get("seed", 42)
    if args.model is None and "model" in run:
        args.model = TRAINING_DIR / run["model"]
    default_envs = 64 if args.mode == "train" else (1 if args.keyboard else 4)
    resumed_envs = run.get("num_envs", default_envs) if args.resume else default_envs
    num_envs = args.num_envs if args.num_envs is not None else resumed_envs
    env_class = WheelLegFlatEnv if args.robot == "v0.2" else WheelbipeFlatEnv
    task = settings["task"]
    locomotion = task["locomotion"]
    if locomotion["enabled"]:
        if args.robot != "v0.2":
            parser.error("Continuous leg-height locomotion requires --robot v0.2")
        env_class = WheelLegLocomotionEnv
    if checkpoint and "task" in run and run["task"] != env_class.task_name:
        parser.error("Checkpoint and selected configuration use different observation contracts")
    if args.forward_limit is not None:
        if not locomotion["enabled"] or not 0 < args.forward_limit <= MAX_FORWARD_SPEED:
            parser.error("--forward-limit requires locomotion training and must be in (0, 2.5] m/s")
        task["forward_command"] = [-args.forward_limit, args.forward_limit]
    requested_heights = args.height_sequence or ([] if args.height is None else [args.height])
    if requested_heights:
        low, high = locomotion["height_range"]
        if not locomotion["enabled"] or any(not low <= height <= high for height in requested_heights):
            parser.error(f"Height demo requires a locomotion checkpoint and targets in [{low:g}, {high:g}] m")
    if body_heights and (not locomotion["enabled"] or any(not isfinite(height) for height in body_heights)):
        parser.error("Body heights require a locomotion checkpoint and finite targets")
    lane_class = TERRAIN_LANES.get(locomotion["terrain"])
    if args.free_drive and (not locomotion["enabled"] or lane_class is None or locomotion["cliff_drop_heights"]):
        parser.error("--free-drive requires terrain locomotion without separate cliff decks")
    if args.surface:
        if not locomotion["enabled"] or lane_class is None or args.surface not in lane_class.surface_names(locomotion):
            parser.error("--surface must name a lane in the selected locomotion terrain")
        if args.surface == "cliff" and locomotion["cliff_drop_heights"] and args.speed < 0 and not args.keyboard:
            parser.error("The cliff drop requires a positive forward speed")
    if args.terrain_variant is not None:
        if not args.surface:
            parser.error("--terrain-variant requires --surface")
        if not 1 <= args.terrain_variant <= lane_class.variants:
            parser.error(f"--terrain-variant must be in [1, {lane_class.variants}]")
    if args.terrain_seed is not None:
        if not locomotion["enabled"] or locomotion["terrain"] == "plane" or args.terrain_seed < 0:
            parser.error("--terrain-seed requires a terrain configuration and a nonnegative seed")
        locomotion["terrain_seed"] = args.terrain_seed
    control_dt = env_class.physics_dt * env_class.decimation
    if num_envs < 1 or args.iterations < 1 or (
        args.seconds is not None and (not isfinite(args.seconds) or args.seconds < control_dt)
    ):
        parser.error(f"Use at least one environment, one iteration and {control_dt:g} seconds")
    if not isfinite(args.speed) or not isfinite(args.yaw):
        parser.error("--speed and --yaw must be finite")
    if args.live_port is not None and not 1 <= args.live_port <= 65535:
        parser.error("--live-port must be between 1 and 65535")
    if args.live_fps is not None and (args.live_port is None or not 1 <= args.live_fps <= 60):
        parser.error("--live-fps requires --live-port and must be between 1 and 60")
    if args.viewer_fps is not None and not 1 <= args.viewer_fps <= 120:
        parser.error("--viewer-fps must be between 1 and 120")
    if args.live_stream and (args.mode != "train" or args.live_port is None):
        parser.error("--live-stream requires train mode and --live-port")
    ssh_session = any(environ.get(name) for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"))
    local_training_view = args.mode == "train" and args.live_port is not None and not (args.live_stream or ssh_session)
    native_live_view = local_training_view or args.live_telemetry
    training_view_fps = (args.live_fps or 60) if native_live_view else None
    args.model = args.model or env_class.model_path
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")
    if args.mode == "demo" and not args.no_sim_log and args.sim_log is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.checkpoint.parent.name if args.checkpoint is not None else "scripted"
        args.sim_log = TRAINING_DIR / "outputs" / "sim_logs" / f"{stamp}_{run_name}"
    if args.no_sim_log:
        args.sim_log = None

    try:
        if "model_sha256" in run and sha256(args.model.read_bytes()).hexdigest() != run["model_sha256"]:
            raise ValueError("Model differs from the checkpoint's run configuration. Use --model with the original MJCF.")
        if args.mode == "train":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output = args.output or TRAINING_DIR / "outputs" / f"ppo_{timestamp}"
            if output.is_dir() and any(output.iterdir()):
                raise ValueError(f"Output directory already contains files: {output}. Choose a new --output path.")
            ppo = settings["ppo"]
            if ppo["algorithm"]["num_mini_batches"] > num_envs * ppo["num_steps_per_env"]:
                raise ValueError("PPO mini batches outnumber the transitions collected per iteration")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    with ExitStack() as stack:
        live_view = None
        try:
            if args.live_port is not None:
                live_view = stack.enter_context(TrainingView(
                    args.live_port, iterations=args.iterations if args.mode == "train" else None,
                    robot=env_class.label, num_envs=num_envs, keyboard=args.keyboard, fps=args.live_fps,
                    stream=not native_live_view,
                ))
        except (OSError, ValueError) as error:
            parser.error(str(error))

        gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu, seed=args.seed, logging_level="warning")
        stack.callback(gs.destroy)
        if live_view is not None:
            stack.callback(live_view.stop_rendering)
        print(f"Building {num_envs} Genesis environments on {gs.device}...", flush=True)
        preview_options = {}
        if native_keyboard and args.surface and not (args.free_drive or locomotion["cliff_drop_heights"]):
            preview_options["preview_lane"] = (args.surface, (args.terrain_variant or 1) - 1)
        env = env_class(args.model, num_envs, viewer=args.viewer or local_training_view,
                        task_config=settings["task"], training_view_fps=training_view_fps, keyboard_view=native_keyboard,
                        viewer_fps=args.viewer_fps, **preview_options)
        if body_heights:
            low = env.height_low + env.body_height_offset
            high = env.height_high + env.body_height_offset
            if any(height < low - 0.00005 or height > high + 0.00005 for height in body_heights):
                parser.error(f"Body height targets must be in [{low:.4f}, {high:.4f}] m")
            heights = [min(env.height_high, max(env.height_low, h - env.body_height_offset)) for h in body_heights]
            if args.body_height_sequence:
                args.height_sequence = heights
            else:
                args.height = heights[0]
        if local_training_view:
            from viz.viewer_drive import ViewerFollow

            camera = env.scene.viewer.add_plugin(ViewerFollow())
            attach_local_view(env, live_view, camera, training_view_fps)
        elif live_view is not None and live_view.stream:
            env.on_step = lambda reset: live_view.capture(env, args.model, reset)
        print(f"{env.label}: {env.robot.n_links} rigid links, {env.robot.n_equalities} equality constraints, "
              f"{env.num_actions} actions at {1 / env.control_dt:.0f} Hz", flush=True)
        if args.mode == "train":
            train(env, args, output, settings, live_view)
        else:
            try:
                demo(env, args, settings["ppo"], live_view)
            except gs.GenesisException as error:
                if not args.viewer or str(error) != "Viewer closed.":
                    raise
            except KeyboardInterrupt:
                if not args.keyboard:
                    raise
                print("Keyboard demo stopped.")


if __name__ == "__main__":
    main()
