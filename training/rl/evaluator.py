"""Interactive demo, replay, keyboard driving, local viewer bridge, and video recording."""

from itertools import count
from math import cos, hypot, sin
from time import monotonic, sleep

import genesis as gs
import torch

from rl.checkpoint import build_inference_policy
from tasks.locomotion_env import WheelLegLocomotionEnv


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
        from viz.sim_logger import SimLogger

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
        policy = build_inference_policy(
            env,
            ppo,
            options.checkpoint,
            actor_true_forward_speed=getattr(env, "actor_true_forward_speed", False),
            actor_history_steps=getattr(env, "actor_history_steps", 1),
        )

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
        from viz.viewer_drive import ViewerDrive

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
        from viz.command_slope import KeyboardCommandSlope

        command_slope = KeyboardCommandSlope(
            vx_accel=options.accel,
            vx_decel=options.brake_accel,
            wz_accel=options.yaw_accel,
            wz_decel=options.yaw_brake_accel,
        )
    elif viewer is not None:
        from viz.viewer_drive import ViewerFollow

        camera_follow = viewer.add_plugin(ViewerFollow())

    self_righting = None
    if use_self_righting:
        from robots.v02.rescue import FallSelfRighting

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
        from viz.render_video import record

        record(options.model, env.robot, frames, options.video, round(1 / (control_dt * control_steps_per_frame)),
               env.label, "PPO policy" if policy is not None else "Scripted controller",
               terrain=getattr(env, "lanes", None))
