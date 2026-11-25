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
from envs.terrain import (SCUT_GROUND_COMPOUND_SURFACES, SCUT_GROUND_SINGLE_SIDE_SURFACES, SCUT_GROUND_SPEED_BUMP,
                          SCUT_V1_SURFACES, TERRAIN_LANES)
from robots.scut import WheelbipeFlatEnv
from robots.wheel_leg import WheelLegFlatEnv
from robots.wheel_leg_locomotion import WheelLegLocomotionEnv
from tools.live_view import TrainingView


class RobotRunner(OnPolicyRunner):
    live_view = None
    capped_action_std = None

    def save(self, path, infos=None):
        super().save(path, {**(infos or {}), "robot": self.env.label, "task": self.env.task_name})

    def limit_exploration(self, maximum):
        if not maximum:
            return
        std = getattr(self.alg.policy, "std", None)
        if not isinstance(std, torch.nn.Parameter):
            raise ValueError("ppo.max_action_std requires a per-action standard deviation parameter")

        @torch.no_grad()
        def clamp_std(*_):
            std.clamp_(min=1e-6, max=maximum)

        clamp_std()
        self.alg.optimizer.register_step_post_hook(clamp_std)
        self.capped_action_std = std
        print(f"Per-action exploration std capped at {maximum:g}", flush=True)

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        if getattr(self.env, "transition_course", False):
            peak = self.env.transition_peak_command.masked_fill(self.env.surface == 0, 0.0).max().item()
            self.writer.add_scalar("Transition/max_live_command_m_s", peak, locs["it"])
        if getattr(self.env, "recovery_fraction", 0.0):
            attempts, successes = self.env.recovery_counts.tolist()
            self.env.recovery_counts.zero_()
            if attempts:
                self.writer.add_scalar("Recovery/attempts", attempts, locs["it"])
                self.writer.add_scalar("Recovery/success_rate", successes / attempts, locs["it"])
                print(f"Low recovery stable: {successes}/{attempts} ({successes / attempts:.1%})", flush=True)
        if self.capped_action_std is not None:
            self.writer.add_scalar("Policy/max_noise_std", self.capped_action_std.detach().max().item(), locs["it"])
        elif hasattr(self.alg.policy, "std"):
            std_detached = self.alg.policy.std.detach()
            self.writer.add_scalar("Policy/max_noise_std", std_detached.max().item(), locs["it"])
            self.writer.add_scalar("Policy/mean_noise_std", std_detached.mean().item(), locs["it"])
        if getattr(self.env, "terrain_curriculum_enabled", False):
            mean_lvl = self.env.terrain_level.float().mean().item()
            maxed_frac = self.env.terrain_maxed.float().mean().item()
            self.writer.add_scalar("Curriculum/mean_terrain_level", mean_lvl, locs["it"])
            self.writer.add_scalar("Curriculum/maxed_fraction", maxed_frac, locs["it"])
            print(
                f"Terrain curriculum: mean level {mean_lvl:.2f}/{self.env.lanes.variants - 1}, "
                f"maxed {maxed_frac:.1%}",
                flush=True,
            )
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


def add_actor_speed_input(module, state, prefix, *_):
    name = f"{prefix}actor.0.weight"
    old = state.get(name)
    weight = module.actor[0].weight
    if old is not None and old.shape == (weight.shape[0], weight.shape[1] - 1):
        state[name] = torch.nn.functional.pad(old, (0, 1))
        print("Initialized actor speed input with zero weights", flush=True)


def add_actor_history_inputs(module, state, prefix, *_):
    name = f"{prefix}actor.0.weight"
    old = state.get(name)
    weight = module.actor[0].weight
    if old is None or old.shape[0] != weight.shape[0]:
        return
    extra_dims = weight.shape[1] - old.shape[1]
    if extra_dims > 0:
        state[name] = torch.nn.functional.pad(old, (0, extra_dims))
        print(f"Initialized {extra_dims} actor history inputs with zero weights", flush=True)


def load_checkpoint(runner, path, *, load_optimizer, initialize_actor_speed=False, initialize_actor_history=False):
    def add_critic_inputs(module, state, prefix, *_):
        name = f"{prefix}critic.0.weight"
        old = state.get(name)
        weight = module.critic[0].weight
        extra_dims = runner.env.critic_terrain_dims
        if old is None or old.shape != (weight.shape[0], weight.shape[1] - extra_dims):
            return
        if load_optimizer:
            raise ValueError("Use --init-checkpoint to add terrain observations to an existing critic")
        state[name] = torch.nn.functional.pad(old, (0, extra_dims))
        print(f"Initialized {extra_dims} terrain critic inputs with zero weights", flush=True)

    with ExitStack() as stack:
        if initialize_actor_speed:
            stack.enter_context(runner.alg.policy.register_load_state_dict_pre_hook(add_actor_speed_input))
        if initialize_actor_history:
            if load_optimizer:
                raise ValueError("Use --init-checkpoint to add history observations to an existing actor")
            stack.enter_context(runner.alg.policy.register_load_state_dict_pre_hook(add_actor_history_inputs))
        if isinstance(runner.env, WheelLegLocomotionEnv) and runner.env.critic_scan_offsets is not None:
            stack.enter_context(runner.alg.policy.register_load_state_dict_pre_hook(add_critic_inputs))
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
    actor_speed = getattr(env, "actor_true_forward_speed", False)
    history_steps = getattr(env, "actor_history_steps", 1)
    runner = RobotRunner(
        env,
        runner_config(settings["ppo"], actor_true_forward_speed=actor_speed, actor_history_steps=history_steps),
        log_dir=str(output),
        device=str(env.device),
    )
    runner.live_view = live_view
    if options.resume is not None:
        load_checkpoint(runner, options.resume, load_optimizer=True)
        # RSL-RL restores the optimizer's rate, but PPO also tracks it for the adaptive schedule.
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
        runner.current_learning_iteration += 1
        print(f"Resuming at iteration {runner.current_learning_iteration} from {options.resume}", flush=True)
    elif options.init_checkpoint is not None:
        load_checkpoint(
            runner,
            options.init_checkpoint,
            load_optimizer=False,
            initialize_actor_speed=actor_speed,
            initialize_actor_history=history_steps > 1,
        )
        runner.current_learning_iteration = 0
        min_std = settings["ppo"].get("init_checkpoint_min_std", 0.0)
        if min_std > 0.0 and isinstance(getattr(runner.alg.policy, "std", None), torch.nn.Parameter):
            with torch.no_grad():
                runner.alg.policy.std.clamp_min_(min_std)
            print(
                f"Raised policy std floor to {min_std:g} on --init-checkpoint "
                f"(range {runner.alg.policy.std.min().item():.3f}-{runner.alg.policy.std.max().item():.3f})",
                flush=True,
            )
        print(f"Initializing from {options.init_checkpoint} with a fresh optimizer", flush=True)
    runner.limit_exploration(settings["ppo"]["max_action_std"])
    if isinstance(env, WheelLegLocomotionEnv):
        env.start_height_curriculum(runner.current_learning_iteration, runner.num_steps_per_env)
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
    terrain_locomotion = isinstance(env, WheelLegLocomotionEnv)

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
            height = base_height[0]
            if terrain_locomotion:
                height = height - env.ground_height_at(env.robot.get_pos())[0].nan_to_num()
                height_target += env.body_height_offset
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


def demo(env, options, ppo, live_view=None):
    sim_log = None
    if options.sim_log is not None and (len(env.leg_dofs), len(env.wheel_dofs)) != (4, 2):
        print("Simulation log skipped: it expects the 4-leg-motor / 2-wheel V0.2 layout", flush=True)
    elif options.sim_log is not None:
        from tools.sim_logger import SimLogger

        sim_log = SimLogger(options.sim_log, env, options)
        print(f"Simulation log: {sim_log.directory} (telemetry.csv / events.log / incidents/)", flush=True)
    try:
        _demo(env, options, ppo, live_view, sim_log)
    finally:
        if sim_log is not None:
            sim_log.close()


@torch.inference_mode()
def _demo(env, options, ppo, live_view=None, sim_log=None):
    policy = None
    if options.checkpoint is not None:
        cfg = runner_config(
            ppo,
            actor_true_forward_speed=getattr(env, "actor_true_forward_speed", False),
            actor_history_steps=getattr(env, "actor_history_steps", 1),
        )
        runner = OnPolicyRunner(env, cfg, device=str(env.device))
        load_checkpoint(runner, options.checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)

    viewer = env.scene.viewer if options.viewer else None
    motion_input = live_view
    native_drive = None
    camera_follow = None
    command_slope = None
    drive_speed = abs(options.speed)
    forward_limits = env.cfg["forward_command"]
    locomotion = isinstance(env, WheelLegLocomotionEnv)
    # Keyboard driving keeps a fallen robot in the scene and stands it up with the firmware RESCUE/GETUP port.
    use_self_righting = options.keyboard and locomotion and not options.fall_reset
    if options.keyboard and (live_view is None or not live_view.stream):
        from tools.viewer_drive import ViewerDrive

        native_drive = viewer.add_plugin(
            ViewerDrive(
                enable_height=locomotion,
                enable_reset=options.free_drive or use_self_righting,
                vx_accel=options.accel,
                vx_decel=options.brake_accel,
                wz_accel=options.yaw_accel,
                wz_decel=options.yaw_brake_accel,
            )
        )
        camera_follow = native_drive
        motion_input = native_drive
        command_slope = native_drive.command_slope
        if not locomotion:
            forward_limits = (-drive_speed, drive_speed)
        print(
            f"Genesis keyboard drive: W/S forward/back, A/D steer, Space brake/stop, "
            f"speed {drive_speed:g} m/s, "
            f"vx slope {options.accel:g}/{options.brake_accel:g} m/s^2 (real-first), "
            f"yaw slope {options.yaw_accel:g}/{options.yaw_brake_accel:g} rad/s^2 (planning-first)",
            flush=True,
        )
        if locomotion:
            print("Z/X raise/lower the continuous body-height target", flush=True)
        if options.free_drive:
            print("Free drive: full course, no episode timer; R returns to spawn", flush=True)
        elif use_self_righting:
            print("R returns to spawn", flush=True)
    elif options.keyboard:
        from tools.command_slope import KeyboardCommandSlope

        command_slope = KeyboardCommandSlope(
            vx_accel=options.accel,
            vx_decel=options.brake_accel,
            wz_accel=options.yaw_accel,
            wz_decel=options.yaw_brake_accel,
        )
    elif viewer is not None:
        from tools.viewer_drive import ViewerFollow

        camera_follow = viewer.add_plugin(ViewerFollow())

    self_righting = None
    if use_self_righting:
        from robots.wheel_leg_rescue import FallSelfRighting

        self_righting = FallSelfRighting(env, options.model)
        if sim_log is not None:
            sim_log.mode_source = self_righting.mode_name
        print("Fall handling: no reset on fall; firmware RESCUE -> GETUP self-righting, then the policy resumes "
              "(use --fall-reset for the old behaviour)", flush=True)
    elif options.keyboard and options.fall_reset:
        print("Fall handling: reset on fall (--fall-reset)", flush=True)
    control_dt = env.control_dt
    steps = round(options.seconds / control_dt) if options.seconds is not None else None
    if locomotion:
        if options.keyboard:
            if options.accel is not None:
                env.manual_forward_acceleration = options.accel
            if options.brake_accel is not None:
                env.manual_brake_acceleration = options.brake_accel
        default_height = env.drop_approach_height if env.forward_drop and options.surface == "cliff" else env.height_nominal
        height_sequence = options.height_sequence or [options.height or default_height]
        env.begin_demo(height_sequence[0], options.surface or "flat", reverse=options.speed < 0.0,
                       variant=(options.terrain_variant or 1) - 1, free_drive=options.free_drive)
        if env.lanes is not None:
            print(f"Terrain: {env.terrain_description()}", flush=True)
        height_request = height_sequence[0]
        if command_slope is not None:
            command_slope.reset(height=height_request)
    else:
        env.reset()
        if command_slope is not None:
            command_slope.reset()
    native_monitor = live_view is not None and not live_view.stream
    if native_monitor:
        attach_local_view(env, live_view, camera_follow, options.live_fps or 60)
    elif camera_follow is not None:
        camera_follow.show_terrain("Free drive" if options.free_drive else env.terrain_description())
    last_x = env.robot.get_pos()[0, 0].clone()
    travelled = env.commands.new_zeros(())
    resets = torch.zeros((), dtype=torch.long, device=env.device)
    falls = torch.zeros((), dtype=torch.long, device=env.device)
    height_error = env.commands.new_zeros(())
    saturated_steps = torch.zeros((), dtype=torch.long, device=env.device)
    finite_ok = torch.ones((), dtype=torch.bool, device=env.device)
    if locomotion:
        posture_max = env.commands.new_zeros(3)
        pitch_peak = env.commands.new_zeros(())
    wheel_contact_frames = torch.zeros((), dtype=torch.long, device=env.device)
    speed_integral = torch.zeros_like(env.commands[:, 0])
    frames = []
    control_steps_per_frame = 2
    log_interval = round(1.0 / control_dt)
    viewer_fps = options.viewer_fps or (30 if native_drive is not None else 60)
    render_interval = 1.0 / viewer_fps
    next_step = monotonic()
    next_render = next_step
    window_wall = next_step
    window_steps = 0
    for i in count() if steps is None else range(steps):
        if viewer is not None and not viewer.is_alive():
            return
        seconds = i * control_dt
        is_log_step = (i + 1) % log_interval == 0 or i + 1 == steps
        is_video_step = options.video is not None and i % control_steps_per_frame == 0
        if native_drive is not None and native_drive.take_reset_request():
            env.reset()
            if self_righting is not None:
                self_righting.cancel()
            if sim_log is not None:
                sim_log.event("USER_RESET", "按 R 回到出生点", t=seconds)
            height_request = height_sequence[0]
            if command_slope is not None:
                command_slope.reset(height=height_request)
            last_x = env.robot.get_pos()[0, 0].clone()
            resets += 1
            next_step = monotonic()
        need_base_state = policy is None or is_log_step or is_video_step or command_slope is not None
        if need_base_state:
            lin_vel, ang_vel, gravity, _ = env._base_state()
        if options.keyboard:
            forward, turn = motion_input.motion_axes()
            vx_target = drive_target(forward, forward_limits, drive_speed)
            yaw_cap = abs(options.yaw) if options.yaw else min(4.5, max(abs(v) for v in env.cfg["yaw_command"]))
            yaw_target = drive_target(turn, env.cfg["yaw_command"], yaw_cap)
            vx_real = lin_vel[0, 0].item()
            wz_real = ang_vel[0, 2].item()
            if locomotion:
                if options.height_sequence:
                    index = min(int(seconds / options.height_hold), len(height_sequence) - 1)
                    height_request = height_sequence[index]
                elif native_drive is not None:
                    height_request += 0.10 * native_drive.height_axis() * control_dt
                    height_request = max(env.height_low, min(height_request, env.height_high))
                if self_righting is not None and self_righting.active and bool(self_righting.enabled[0]):
                    # Like the firmware RESCUE/GETUP modes: hold zero motion and restart the slopes from
                    # the low get-up stance so the policy takes over smoothly afterwards.
                    command_slope.reset(height=env.height_low)
                    vx_cmd, yaw_cmd, height_cmd = 0.0, 0.0, env.height_low
                else:
                    vx_cmd, yaw_cmd, height_cmd = command_slope.calculate(
                        vx_target,
                        yaw_target,
                        control_dt,
                        vx_real=vx_real,
                        wz_real=wz_real,
                        height_target=height_request,
                    )
                env.set_commands(vx_cmd, yaw_cmd, height_cmd, slew=False)
            else:
                vx_cmd, yaw_cmd = command_slope.calculate(
                    vx_target, yaw_target, control_dt, vx_real=vx_real, wz_real=wz_real
                )
                env.commands[:, 0] = vx_cmd
                env.commands[:, 1] = yaw_cmd
        else:
            vx_cmd = options.speed if seconds >= 0.5 else 0.0
            yaw_cmd = options.yaw if seconds >= 0.5 else 0.0
            if locomotion:
                if options.height_sequence:
                    index = min(int(seconds / options.height_hold), len(height_sequence) - 1)
                    height_request = height_sequence[index]
                env.set_commands(vx_cmd, yaw_cmd, height_request)
            else:
                env.commands[:, 0] = vx_cmd
                env.commands[:, 1] = yaw_cmd
        if policy is not None:
            actions = policy(env.get_observations())
            if self_righting is not None:
                actions = self_righting.mask_actions(actions)
        else:
            speed_integral.add_(control_dt * (vx_cmd - lin_vel[:, 0])).clamp_(-0.5, 0.5)
            actions = scripted_actions(env, lin_vel, ang_vel, gravity, speed_integral)
        obs, reward, done, extras = env.step(actions)
        if self_righting is not None:
            started, finished = self_righting.update(extras, done)
            falls += started.sum()
            if started[0].item():
                print(f"t={seconds + control_dt:.2f}s fall detected: firmware RESCUE/GETUP engaged", flush=True)
            if finished[0].item():
                print(f"t={seconds + control_dt:.2f}s self-righting finished: policy control resumed", flush=True)
        if sim_log is not None:
            if options.keyboard:
                key_height = native_drive.height_axis() if native_drive is not None and locomotion else 0
                inputs = (forward, turn, key_height, vx_target, yaw_target,
                          height_request if locomotion else float("nan"))
            else:
                inputs = (0, 0, 0, vx_cmd, yaw_cmd, height_request if locomotion else float("nan"))
            sim_log.step(seconds + control_dt, actions, extras, done, inputs)
        if command_slope is not None and done[0].item():
            command_slope.reset(height=height_request if locomotion else None)
        qpos = env.robot.get_qpos()
        finite_ok &= torch.isfinite(qpos).all() & torch.isfinite(reward).all() & torch.isfinite(obs["policy"]).all()
        resets += done.sum()
        if locomotion:
            if self_righting is None:
                falls += extras["fallen"].sum()
            saturated_steps += extras["leg_saturated"][0].long()
            body_height_t = extras["body_clearance"][0]
            body_ref_t = env.commands[0, 2] + env.body_height_offset
            height_error += (body_height_t - body_ref_t).abs()
            posture_max = torch.maximum(posture_max, extras["posture"][0])
            pitch_peak = torch.maximum(pitch_peak, extras["pitch_deg"][0].abs())
        speed_integral.masked_fill_(done, 0.0)
        wheel_contact_frames += extras["wheel_contacts"][0].all().long()
        position = env.robot.get_pos()[0]
        x_t = position[0]
        travelled += torch.where(done[0], 0.0, x_t - last_x)
        last_x = x_t.clone()
        if camera_follow is not None and not native_monitor:
            if native_drive is not None:
                now_render = monotonic()
                if now_render >= next_render:
                    next_render = max(now_render, next_render + render_interval)
                    x_cam, y_cam = position[:2].tolist()
                    camera_follow.follow_position(x_cam, y_cam)
                    # Render at viewer_fps (default 30 Hz) independently of the 50 Hz control loop.
                    viewer.update(force=True)
            else:
                x_cam, y_cam = position[:2].tolist()
                camera_follow.follow_position(x_cam, y_cam)
        if is_video_step:
            vx_cmd_val = env.commands[0, 0].item() if locomotion else vx_cmd
            frames.append((qpos[0].cpu().numpy().copy(), seconds + control_dt, vx_cmd_val, lin_vel[0, 0].item()))
        window_steps += 1
        if is_log_step:
            if not finite_ok.item():
                raise RuntimeError(f"Nonfinite simulation state at step {i + 1}")
            now_log = monotonic()
            rtf = (window_steps * control_dt) / max(now_log - window_wall, 1e-6)
            window_wall = now_log
            window_steps = 0
            x, y_pos, z_pos = position.tolist()
            vx = lin_vel[0, 0].item()
            if locomotion:
                vx_cmd_log, yaw_cmd_log, height_ref_log = env.commands[0].tolist()
                body_height = body_height_t.item()
                body_reference = height_ref_log + env.body_height_offset
            else:
                vx_cmd_log, yaw_cmd_log = vx_cmd, yaw_cmd
            yaw = (f" wz={ang_vel[0, 2].item():+.2f}rad/s yaw_target={yaw_cmd_log:+.2f}rad/s"
                   if options.keyboard or options.yaw else "")
            height = (f" body_height={body_height * 100:.1f}cm "
                      f"ref={body_reference * 100:.1f}cm goal={(height_request + env.body_height_offset) * 100:.1f}cm "
                      f"pitch={extras['pitch_deg'][0].item():+.1f}deg"
                      if locomotion else "")
            lateral = f" y={y_pos:.3f}m" if options.free_drive else ""
            chassis_mode = f" mode={self_righting.mode_name()}" if self_righting is not None else ""
            print(f"t={seconds + control_dt:.2f}s x={x:+.3f}m{lateral} z={z_pos:.3f}m "
                  f"vx={vx:+.3f}m/s target={vx_cmd_log:+.2f}m/s{yaw}{height} "
                  f"resets={resets.item()}{chassis_mode} rtf={rtf:.2f}x", flush=True)
        if live_view is not None or native_drive is not None:
            now = monotonic()
            # Allow up to 2 control steps of bounded catch-up so brief render jitter stays 1:1 with wall time.
            next_step = max(now - 2.0 * control_dt, next_step + control_dt)
            if next_step > now:
                sleep(next_step - now)

    if not finite_ok.item():
        raise RuntimeError(f"Nonfinite simulation state at step {steps}")
    print(f"Demo: {steps} control steps x {env.num_envs} envs, {resets.item()} resets, "
          f"env 0 forward displacement {travelled.item():+.3f} m, both wheels in contact "
          f"for {wheel_contact_frames.item()}/{steps} steps.")
    if locomotion:
        print(f"Falls: {falls.item()}; env 0 body-height MAE: {100 * height_error.item() / steps:.2f} cm; "
              f"leg torque saturation: {100 * saturated_steps.item() / steps:.1f}% of steps")
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
    parser.add_argument("--surface", choices=(SCUT_V1_SURFACES + SCUT_GROUND_COMPOUND_SURFACES
                                             + SCUT_GROUND_SINGLE_SIDE_SURFACES + (SCUT_GROUND_SPEED_BUMP, "transitions")),
                        help="Spawn on this lane when replaying a terrain policy")
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
    parser.add_argument("--free-drive", action="store_true",
                        help="Drive across all terrain lanes without a time limit in the native Genesis viewer")
    parser.add_argument("--fall-reset", action="store_true",
                        help="Keyboard demo: reset on fall instead of the firmware RESCUE/GETUP self-righting")
    sim_logs = parser.add_mutually_exclusive_group()
    sim_logs.add_argument("--sim-log", type=Path,
                          help="Demo run log directory (default: outputs/sim_logs/<timestamp>, always on in demo)")
    sim_logs.add_argument("--no-sim-log", action="store_true", help="Disable the automatic demo run log")
    parser.add_argument("--video", type=Path, help="Save a replay of env 0 as an MP4")
    checkpoint_args = parser.add_mutually_exclusive_group()
    checkpoint_args.add_argument("--checkpoint", type=Path, help="Replay a PPO checkpoint in demo mode")
    checkpoint_args.add_argument("--resume", type=Path, help="Continue PPO training from a checkpoint in train mode")
    checkpoint_args.add_argument("--init-checkpoint", type=Path,
                                 help="Start a new run from compatible weights with a fresh optimizer")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--output", type=Path, help="New or empty directory for the PPO run")
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
