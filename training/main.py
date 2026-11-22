"""Run a Genesis demo or train a wheel-leg PPO policy."""

import argparse
from contextlib import ExitStack
from datetime import datetime
from hashlib import sha256
from itertools import count
from math import cos, hypot, isfinite, sin
from os import environ
from pathlib import Path
from statistics import mean
from time import monotonic, sleep

import genesis as gs
from rsl_rl.runners import OnPolicyRunner
import torch

from configs.experiment import (MAX_FORWARD_SPEED, SNAPSHOT_NAME, TRAINING_DIR, load_config, load_snapshot,
                                runner_config, save_run_config)
from envs.terrain import SCUT_V1_SURFACES, SURFACES
from robots.scut import WheelbipeFlatEnv
from robots.wheel_leg import WheelLegFlatEnv
from robots.wheel_leg_locomotion import WheelLegLocomotionEnv
from tools.live_view import TrainingView


class RobotRunner(OnPolicyRunner):
    live_view = None

    def save(self, path, infos=None):
        super().save(path, {**(infos or {}), "robot": self.env.label, "task": self.env.task_name})

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        if self.live_view is None:
            return
        batch_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        seconds = locs["collection_time"] + locs["learn_time"]
        rewards, lengths = locs["rewbuffer"], locs["lenbuffer"]
        losses = locs["loss_dict"]
        self.live_view.add_metrics(
            iteration=locs["it"] - locs["start_iter"] + 1,
            steps=self.tot_timesteps,
            steps_per_second=int(batch_size / seconds),
            elapsed=self.tot_time,
            reward=mean(rewards) if rewards else None,
            episode_length=mean(lengths) if lengths else None,
            value_loss=losses["value_function"],
            policy_loss=losses["surrogate"],
        )


def load_checkpoint(runner, path, *, load_optimizer):
    info = runner.load(str(path), load_optimizer=load_optimizer, map_location=str(runner.env.device)) or {}
    # Checkpoints before robot metadata was added only used the SCUT model.
    checkpoint_robot = info.get("robot", WheelbipeFlatEnv.label)
    if checkpoint_robot != runner.env.label:
        raise ValueError(f"Checkpoint was trained on {checkpoint_robot}; selected robot is {runner.env.label}")
    if info.get("task", "flat") != runner.env.task_name:
        raise ValueError("Checkpoint and selected configuration use different observation contracts")


def train(env, options, output, settings, live_view=None):
    if isinstance(env, WheelLegLocomotionEnv):
        env.enable_observation_noise()
    runner = RobotRunner(env, runner_config(settings["ppo"]), log_dir=str(output), device=str(env.device))
    runner.live_view = live_view
    if options.resume is not None:
        load_checkpoint(runner, options.resume, load_optimizer=True)
        # RSL-RL restores the optimizer's rate, but PPO also tracks it for the adaptive schedule.
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
        runner.current_learning_iteration += 1
        print(f"Resuming at iteration {runner.current_learning_iteration} from {options.resume}", flush=True)
    elif options.init_checkpoint is not None:
        load_checkpoint(runner, options.init_checkpoint, load_optimizer=False)
        runner.current_learning_iteration = 0
        print(f"Initializing from {options.init_checkpoint} with a fresh optimizer", flush=True)
    save_run_config(output, settings, options, type(env), env.num_envs)
    try:
        runner.learn(num_learning_iterations=options.iterations, init_at_random_ep_len=True)
    finally:
        if runner.writer is not None:
            runner.writer.close()
    model = output / f"model_{runner.current_learning_iteration}.pt"
    print(f"PPO completed {options.iterations} iterations. Checkpoint: {model.resolve()}")


def attach_local_view(env, live_view, camera, fps):
    viewer = env.scene.viewer
    update_viewer = viewer.update

    def refresh_viewer(*args, **kwargs):
        nonlocal viewer
        if viewer is None:
            return
        try:
            update_viewer(*args, **kwargs)
        except gs.GenesisException as error:
            if str(error) != "Viewer closed.":
                raise
            viewer = None

    # Genesis also calls update from its geometry setters when a robot resets.
    viewer.update = refresh_viewer
    env_zero = (0,)
    start_x, start_y = env.robot.get_pos(env_zero)[0, :2].tolist()
    speed, yaw = env.commands[0, :2].tolist()
    reference_x = reference_y = heading = 0.0
    next_frame = next_sample = 0.0
    use_leg_height = env.task_name == "locomotion"

    def show_terrain():
        terrain = env.terrain_description()
        live_view.set_terrain(terrain)
        camera.show_terrain(terrain)

    show_terrain()

    def capture(reset):
        nonlocal next_frame, next_sample
        nonlocal start_x, start_y, speed, yaw, reference_x, reference_y, heading
        position = env.robot.get_pos(env_zero)[0, :2]
        x, y, next_speed, next_yaw, height_target, restarted = torch.cat((
            position, env.commands[0], reset.reshape(1),
        )).tolist()
        if restarted:
            start_x, start_y = x, y
            reference_x = reference_y = heading = 0.0
            show_terrain()
        else:
            # Integrate the command used on this step, before any update for the next step.
            half_turn = yaw * env.control_dt / 2.0
            chord = speed * env.control_dt * (sin(half_turn) / half_turn if half_turn else 1.0)
            reference_x += chord * cos(heading + half_turn)
            reference_y += chord * sin(heading + half_turn)
            heading += 2.0 * half_turn
        speed, yaw = next_speed, next_yaw

        now = monotonic()
        if now >= next_sample:
            next_sample = now + 0.1
            lin_vel, ang_vel, gravity, base_height = env._base_state()
            height = env.leg_heights(gravity)[0].mean() if use_leg_height else base_height[0]
            gx, gy, gz = gravity[0].unbind()
            roll = torch.rad2deg(torch.atan2(-gy, -gz))
            pitch = torch.rad2deg(torch.atan2(gx, torch.hypot(gy, gz)))
            vx, wz, measured_height, roll_angle, pitch_angle = torch.stack(
                (lin_vel[0, 0], ang_vel[0, 2], height, roll, pitch),
            ).tolist()
            live_view.set_telemetry(
                speed_target=speed, speed_actual=vx, yaw_target=yaw, yaw_actual=wz,
                height_target=height_target, height_actual=measured_height,
                distance_target=hypot(reference_x, reference_y), distance_actual=hypot(x - start_x, y - start_y),
                roll_actual=roll_angle, pitch_actual=pitch_angle,
            )
        if viewer is not None and now >= next_frame:
            next_frame = now + 1 / fps
            camera.follow_position(x, y)
            refresh_viewer(auto_refresh=True, force=True)

    env.on_step = capture


def scripted_actions(env, velocity, angular_velocity, gravity, speed_integral):
    actions = torch.zeros_like(env.last_actions)
    wheel_speed = env.scripted_wheel_speed(velocity, angular_velocity, gravity, speed_integral)
    actions[:, 4:] = (wheel_speed / 10.0)[:, None]
    return actions


def drive_target(axis, limits, magnitude=None):
    low, high = limits
    limit = max(0.0, high) if axis > 0 else max(0.0, -low)
    return axis * (limit if magnitude is None else min(limit, magnitude))


@torch.inference_mode()
def demo(env, options, ppo, live_view=None):
    policy = None
    if options.checkpoint is not None:
        runner = OnPolicyRunner(env, runner_config(ppo), device=str(env.device))
        load_checkpoint(runner, options.checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)

    viewer = env.scene.viewer if options.viewer else None
    motion_input = live_view
    native_drive = None
    camera_follow = None
    drive_speed = abs(options.speed)
    forward_limits = env.cfg["forward_command"]
    locomotion = isinstance(env, WheelLegLocomotionEnv)
    if options.keyboard and (live_view is None or not live_view.stream):
        from tools.viewer_drive import ViewerDrive

        native_drive = viewer.add_plugin(ViewerDrive(enable_height=locomotion))
        camera_follow = native_drive
        motion_input = native_drive
        if not locomotion:
            forward_limits = (-drive_speed, drive_speed)
        print(f"Genesis keyboard drive: W/S forward/back, A/D steer, Space stop, "
              f"forward speed limit {drive_speed:g} m/s", flush=True)
        if locomotion:
            print("Z/X raise/lower the continuous leg-height target", flush=True)
    elif viewer is not None:
        from tools.viewer_drive import ViewerFollow

        camera_follow = viewer.add_plugin(ViewerFollow())

    control_dt = env.control_dt
    steps = round(options.seconds / control_dt) if options.seconds is not None else None
    if locomotion:
        default_height = env.drop_approach_height if env.forward_drop and options.surface == "cliff" else env.height_nominal
        height_sequence = options.height_sequence or [options.height or default_height]
        env.begin_demo(height_sequence[0], options.surface or "flat", reverse=options.speed < 0.0)
        height_request = height_sequence[0]
    else:
        env.reset()
    native_monitor = live_view is not None and not live_view.stream
    if native_monitor:
        attach_local_view(env, live_view, camera_follow, options.live_fps or 60)
    elif camera_follow is not None:
        camera_follow.show_terrain(env.terrain_description())
    last_x = env.robot.get_pos()[0, 0].item()
    travelled = 0.0
    resets = 0
    falls = 0
    height_error = 0.0
    saturated_steps = 0
    if locomotion:
        posture_max = env.commands.new_zeros(3)
        pitch_peak = env.commands.new_zeros(())
    wheel_contact_frames = 0
    speed_integral = torch.zeros_like(env.commands[:, 0])
    frames = []
    control_steps_per_frame = 2
    next_step = monotonic()
    for i in count() if steps is None else range(steps):
        if viewer is not None and not viewer.is_alive():
            return
        seconds = i * control_dt
        if options.keyboard:
            forward, turn = motion_input.motion_axes()
            vx_cmd = drive_target(forward, forward_limits, drive_speed)
            yaw_cmd = drive_target(turn, env.cfg["yaw_command"])
        else:
            vx_cmd = options.speed if seconds >= 0.5 else 0.0
            yaw_cmd = options.yaw if seconds >= 0.5 else 0.0
        if locomotion:
            if options.height_sequence:
                index = min(int(seconds / options.height_hold), len(height_sequence) - 1)
                height_request = height_sequence[index]
            elif native_drive is not None:
                height_request += 0.10 * native_drive.height_axis() * control_dt
                height_request = max(env.height_low, min(height_request, env.height_high))
            env.set_commands(vx_cmd, yaw_cmd, height_request)
            vx_cmd, yaw_cmd, height_reference = env.commands[0].tolist()
        else:
            env.commands[:, 0] = vx_cmd
            env.commands[:, 1] = yaw_cmd
        lin_vel, ang_vel, gravity, _ = env._base_state()
        if policy is not None:
            actions = policy(env.get_observations())
        else:
            speed_integral.add_(control_dt * (vx_cmd - lin_vel[:, 0])).clamp_(-0.5, 0.5)
            actions = scripted_actions(env, lin_vel, ang_vel, gravity, speed_integral)
        obs, reward, done, extras = env.step(actions)
        qpos = env.robot.get_qpos()
        if not (torch.isfinite(qpos).all() & torch.isfinite(reward).all() & torch.isfinite(obs["policy"]).all()):
            raise RuntimeError(f"Nonfinite simulation state at step {i + 1}")
        resets += int(done.sum())
        if locomotion:
            falls += int(extras["fallen"].sum())
            saturated_steps += int(extras["leg_saturated"][0])
            left, right = extras["leg_heights"][0].tolist()
            height_error += abs((left + right) * 0.5 - height_reference)
            posture_max = torch.maximum(posture_max, extras["posture"][0])
            pitch_peak = torch.maximum(pitch_peak, extras["pitch_deg"][0].abs())
        speed_integral.masked_fill_(done, 0.0)
        wheel_contact_frames += int(extras["wheel_contacts"][0].all())
        vx = lin_vel[0, 0].item()
        position = env.robot.get_pos()[0]
        if camera_follow is not None and not native_monitor:
            x, y = position[:2].tolist()
            camera_follow.follow_position(x, y)
        else:
            x = position[0].item()
        if not done[0]:
            travelled += x - last_x
        last_x = x
        if options.video is not None and i % control_steps_per_frame == 0:
            frames.append((qpos[0].cpu().numpy().copy(), seconds + control_dt, vx_cmd, vx))
        if (i + 1) % round(1.0 / control_dt) == 0 or i + 1 == steps:
            yaw = (f" wz={ang_vel[0, 2].item():+.2f}rad/s yaw_target={yaw_cmd:+.2f}rad/s"
                   if options.keyboard or options.yaw else "")
            height = (f" hL/hR={left * 100:.1f}/{right * 100:.1f}cm "
                      f"ref={height_reference * 100:.1f}cm goal={height_request * 100:.1f}cm "
                      f"pitch={extras['pitch_deg'][0].item():+.1f}deg"
                      if locomotion else "")
            print(f"t={seconds + control_dt:.2f}s x={x:+.3f}m z={position[2].item():.3f}m "
                  f"vx={vx:+.3f}m/s target={vx_cmd:+.2f}m/s{yaw}{height} resets={resets}", flush=True)
        if live_view is not None:
            now = monotonic()
            next_step = max(now, next_step + control_dt)
            if next_step > now:
                sleep(next_step - now)

    print(f"Demo: {steps} control steps x {env.num_envs} envs, {resets} resets, "
          f"env 0 forward displacement {travelled:+.3f} m, both wheels in contact "
          f"for {wheel_contact_frames}/{steps} steps.")
    if locomotion:
        print(f"Falls: {falls}; env 0 leg-height MAE: {100 * height_error / steps:.2f} cm; "
              f"leg torque saturation: {100 * saturated_steps / steps:.1f}% of steps")
        roll, fore_aft, height_gap = posture_max.tolist()
        print(f"Env 0 peaks: |roll| {roll:.2f} deg, |pitch| {pitch_peak.item():.2f} deg, "
              f"left/right fore-aft gap {fore_aft:.2f} cm, "
              f"left/right height gap {height_gap:.2f} cm")
    if frames:
        from tools.render_wheelbipe import record

        record(options.model, env.robot, frames, options.video, round(1 / (control_dt * control_steps_per_frame)),
               env.label, "PPO policy" if policy is not None else "Scripted controller",
               terrain=getattr(env, "lanes", None))


def main():
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
    parser.add_argument("--yaw", type=float, default=0.0, help="Demo yaw-rate request in rad/s")
    heights = parser.add_mutually_exclusive_group()
    heights.add_argument("--height", type=float, help="Demo hip-to-wheel height in meters")
    heights.add_argument("--height-sequence", type=float, nargs="+", help="Successive demo leg-height targets in meters")
    parser.add_argument("--height-hold", type=float, default=2.0, help="Seconds between demo height targets")
    parser.add_argument("--surface", choices=SCUT_V1_SURFACES, help="Spawn on this lane when replaying a terrain policy")
    parser.add_argument("--terrain-seed", type=int, help="Override the mixed-terrain layout seed")
    parser.add_argument("--forward-limit", type=float, help="Override the training speed range with this symmetric limit")
    parser.add_argument("--viewer", action="store_true", help="Show the live Genesis scene in demo mode")
    parser.add_argument("--live-port", type=int, help="Serve training or demo on this loopback port")
    parser.add_argument("--live-fps", type=int, help="Live frame rate target (default: 8 for streamed training, 60 otherwise)")
    parser.add_argument("--live-stream", action="store_true", help="Stream training frames to the browser instead of opening Genesis")
    parser.add_argument("--live-telemetry", action="store_true", help="Use a Genesis window and a telemetry-only dashboard for demo")
    parser.add_argument("--keyboard", action="store_true", help="Drive a PPO demo in Genesis, or in the browser with --live-port")
    parser.add_argument("--video", type=Path, help="Save a replay of env 0 as an MP4")
    checkpoint_args = parser.add_mutually_exclusive_group()
    checkpoint_args.add_argument("--checkpoint", type=Path, help="Replay a PPO checkpoint in demo mode")
    checkpoint_args.add_argument("--resume", type=Path, help="Continue PPO training from a checkpoint in train mode")
    checkpoint_args.add_argument("--init-checkpoint", type=Path,
                                 help="Start a new run from compatible weights with a fresh optimizer")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--output", type=Path, help="New or empty directory for the PPO run")
    args = parser.parse_args()
    if args.mode != "train" and (args.resume or args.init_checkpoint or args.forward_limit is not None):
        parser.error("--resume, --init-checkpoint and --forward-limit require train mode")
    if args.mode == "train" and (args.viewer or args.video or args.checkpoint or args.height is not None
                                 or args.height_sequence or args.surface or args.yaw):
        parser.error("Viewer, recording, checkpoint and manual geometry/command options require demo mode")
    if args.keyboard and (args.mode != "demo" or args.checkpoint is None):
        parser.error("--keyboard requires demo mode and --checkpoint")
    if args.live_telemetry and (args.mode != "demo" or args.live_port is None):
        parser.error("--live-telemetry requires demo mode and --live-port")
    native_keyboard = args.keyboard and (args.live_port is None or args.live_telemetry)
    if args.live_telemetry:
        args.viewer = True
    if native_keyboard:
        args.viewer = True
    if args.speed is None:
        args.speed = 2.5 if native_keyboard else 0.35
    if args.keyboard and args.video and args.seconds is None:
        parser.error("Recording a keyboard demo requires a finite --seconds")
    if not isfinite(args.height_hold) or args.height_hold <= 0:
        parser.error("--height-hold must be finite and positive")
    if args.seconds is None and not args.keyboard:
        args.seconds = len(args.height_sequence) * args.height_hold if args.height_sequence else 5.0

    checkpoint = args.resume or args.checkpoint or args.init_checkpoint
    run = {}
    config_path = args.config
    settings = None
    try:
        if checkpoint is not None:
            if not checkpoint.is_file():
                raise ValueError(f"Checkpoint not found: {checkpoint}")
            snapshot = checkpoint.with_name(SNAPSHOT_NAME)
            if snapshot.is_file():
                settings, run = load_snapshot(snapshot)
                if not isinstance(run, dict):
                    raise ValueError(f"Expected run metadata in {snapshot}")
        if config_path is not None or settings is None:
            settings = load_config(config_path)
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
    if args.surface:
        surfaces = SCUT_V1_SURFACES if locomotion["terrain"] == "scut_v1" else SURFACES
        if not locomotion["enabled"] or locomotion["terrain"] == "plane" or args.surface not in surfaces:
            parser.error("--surface must name a lane in the selected locomotion terrain")
        if args.surface == "cliff" and locomotion["cliff_drop_heights"] and args.speed < 0 and not args.keyboard:
            parser.error("The cliff drop requires a positive forward speed")
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
    if args.live_stream and (args.mode != "train" or args.live_port is None):
        parser.error("--live-stream requires train mode and --live-port")
    ssh_session = any(environ.get(name) for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"))
    local_training_view = args.mode == "train" and args.live_port is not None and not (args.live_stream or ssh_session)
    native_live_view = local_training_view or args.live_telemetry
    training_view_fps = (args.live_fps or 60) if native_live_view else None
    args.model = args.model or env_class.model_path
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")

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
        env = env_class(args.model, num_envs, viewer=args.viewer or local_training_view,
                        task_config=settings["task"], training_view_fps=training_view_fps)
        if local_training_view:
            from tools.viewer_drive import ViewerFollow

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
