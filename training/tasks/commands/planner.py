"""Command sampling, heading hold, friction-circle limiting, and continuous height planning."""

import torch

from configs.experiment import MAX_FORWARD_SPEED, YAW_REFERENCE
from tasks.commands.maneuvers import compute_maneuver_goals
from tasks.commands.terrain_cmd import (
    sample_terrain_command_goals,
    sample_traversal_cruise,
    step_transition_stop,
)


def yaw_from_quat(quat):
    w, x, y, z = quat.unbind(dim=1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def sample_heights(env, count):
    sample = torch.rand((2, count), device=env.device)
    floor = env.height_sampling_floor
    height = floor + sample[0] * (env.height_high - floor)
    height = torch.where(sample[1] >= 1.0 - env.height_max_probability, env.height_high, height)
    height = torch.where(sample[1] < 0.60, max(floor, env.height_nominal), height)
    return torch.where(sample[1] < 0.25, floor, height)


def sample_heights_for_envs(env, ids):
    height = env._sample_heights(ids.numel())
    obs_range = getattr(env, "obstacle_height_range", ())
    hard_range = getattr(env, "hard_obstacle_height_range", ())
    if env.lanes is None or not (obs_range or hard_range):
        return height
    surf = env.surface[ids]
    traverse = surf != 0
    if not traverse.any():
        return height
    u = torch.rand((2, ids.numel()), device=env.device)
    if obs_range:
        olow, ohigh = obs_range
        mid_cap = min(ohigh, 0.24)
        obs_h = torch.where(
            u[1] < 0.35,
            olow + u[0] * (mid_cap - olow),
            olow + u[0] * (ohigh - olow),
        )
        height = torch.where(traverse, obs_h, height)
    if hard_range and getattr(env, "hard_obstacle_surfaces", None) is not None:
        hlow, hhigh = hard_range
        mid_cap = min(hhigh, 0.24)
        hard_h = torch.where(
            u[1] < 0.45,
            hlow + u[0] * (mid_cap - hlow),
            hlow + u[0] * (hhigh - hlow),
        )
        height = torch.where(traverse & env.hard_obstacle_surfaces[surf], hard_h, height)
    return height


def sample_motion_goals(env, ids):
    if ids.numel() == 0:
        return
    sample = torch.rand((ids.numel(), 4), device=env.device)
    forward_cap = None
    if env.special_command_sampler is not None and not env.terrain_traversal:
        motion, duration = env._sample_special_mode_goals(ids)
    else:
        motion = env.command_lower[:2] + sample[:, :2] * (env.command_upper[:2] - env.command_lower[:2])
        motion[:, 0] *= sample[:, 2] >= 0.25
        motion[:, 1] *= (sample[:, 2] >= 0.15) & (sample[:, 2] <= 0.75)
        low, high = env.command_period
        duration = low + sample[:, 3] * (high - low)
        if env.speed_levels.numel():
            forward_cap = env.episode_speed_limit[ids]
            motion[:, 0] = motion[:, 0].clamp(-forward_cap, forward_cap)
        if env.speed_maneuvers and not env.terrain_traversal:
            motion, duration = compute_maneuver_goals(env, ids, motion, duration, sample[:, 3])
    if env.terrain_traversal:
        # Keep moving along the obstacle lane across successive command resamples.
        surface = env.surface[ids]
        traverse = surface != 0
        if env.special_command_sampler is not None:
            flat = ~traverse
            if flat.any():
                motion[flat], duration[flat] = env._sample_special_mode_goals(ids[flat])
            if traverse.any():
                env.special_command_sampler.reset(ids[traverse])
        elif env.speed_maneuvers:
            flat = ~traverse
            motion[flat], duration[flat] = compute_maneuver_goals(
                env, ids[flat], motion[flat], duration[flat], sample[flat, 3]
            )
        if getattr(env, "terrain_commands_enabled", False):
            env.terrain_spinning[ids[~traverse]] = False
            if traverse.any():
                motion[traverse], duration[traverse] = sample_terrain_command_goals(env, ids[traverse])
        else:
            cruise = sample_traversal_cruise(env, ids, sample, forward_cap, surface)
            motion[:, 0] = torch.where(traverse, cruise, motion[:, 0])
            motion[:, 1] = torch.where(traverse, 0.0, motion[:, 1])
    if env.stationary_episode_fraction:
        motion.masked_fill_(env.stationary_episode[ids, None], 0.0)
    env.goals[ids, :2] = motion
    env.command_time[ids] = duration


def sample_height_goals(env, ids):
    if ids.numel() == 0:
        return
    if hasattr(env, "_sample_heights_for_envs"):
        height = env._sample_heights_for_envs(ids)
    else:
        height = env._sample_heights(ids.numel())
    if env.height_maneuvers:
        flat = env.surface[ids] == 0
        height_only = (env.maneuver[ids] == env.height_mode) & flat
        previous = env.goals[ids, 2]
        initial_height = env.episode_steps[ids] == 0
        if env.height_maneuver_sequence.numel():
            index = env.height_sequence_index[ids]
            index = (index + (height_only & ~initial_height)).remainder(env.height_maneuver_sequence.numel())
            env.height_sequence_index[ids] = index
            height = torch.where(height_only, env.height_maneuver_sequence[index], height)
        else:
            midpoint = (env.height_high + env.height_low) / 2
            opposite = torch.where(previous < midpoint, env.height_high, env.height_low)
            height = torch.where(height_only, opposite, height)
        # Start with a fixed height so the cruise cases can reach their speed first.
        hold = flat & (~height_only | initial_height)
        height = torch.where(hold, previous, height)
    elif env.launch_maneuvers:
        launch = (env.maneuver[ids] == env.launch_mode) & (env.surface[ids] == 0)
        height = torch.where(launch, env.goals[ids, 2], height)
    if env.stationary_episode_fraction:
        height = torch.where(env.stationary_episode[ids], env.goals[ids, 2], height)
    if env.low_rough_slope is not None:
        height = torch.where(env.low_rough_slope[ids], height.clamp_max(env.height_nominal), height)
    env.goals[ids, 2] = height
    if env.forward_drop:
        approaching = (env.surface[ids] == env.cliff_surface) & ~env.drop_touched[ids]
        env.goals[ids, 2] = torch.where(
            approaching, env.goals[ids, 2].clamp_min(env.drop_approach_height), env.goals[ids, 2]
        )
    if env.speed_maneuvers or getattr(env, "special_command_sampler", None) is not None:
        low, high = env.command_period
        env.height_time[ids] = low + torch.rand(ids.numel(), device=env.device) * (high - low)


def heading_yaw_target(env, requested_rate, yaw_rate):
    yaw = env._yaw_from_quat(env.robot.get_quat())
    turning = requested_rate.abs() > 1e-6
    # After a turn ends, let the slewed yaw command and measured yaw rate settle before latching heading.
    settling = env.heading_turning & ((env.commands[:, 1].abs() > 0.05) | (yaw_rate.abs() > 0.10))
    on_corridor = (
        getattr(env, "terrain_commands_enabled", False)
        and getattr(env, "auto_commands", False)
        and hasattr(env, "surface")
    )
    latch_mask = (turning | env.heading_turning) & ((env.surface == 0) if on_corridor else True)
    env.heading_target.copy_(torch.where(latch_mask, yaw, env.heading_target))
    env.heading_turning.copy_(turning | settling)
    error = env.heading_target - yaw
    error = torch.atan2(error.sin(), error.cos())
    correction = env.heading_gain * error.sign() * (error.abs() - env.heading_deadband).clamp_min(0.0)
    if on_corridor:
        max_rate = torch.where(
            env.surface != 0,
            requested_rate.new_tensor(max(env.heading_max_yaw_rate, 1.4)),
            requested_rate.new_tensor(env.heading_max_yaw_rate),
        )
        clamped = correction.clamp(-max_rate, max_rate)
    else:
        clamped = correction.clamp(-env.heading_max_yaw_rate, env.heading_max_yaw_rate)
    hold = torch.where(settling, torch.zeros_like(requested_rate), clamped)
    return torch.where(turning, requested_rate, hold)


def advance_commands(env, lin_vel, ang_vel, *, slew=True):
    targets = env.goals.clamp(min=env.command_lower, max=env.command_upper)
    if env.heading_hold_enabled:
        requested_yaw = targets[:, 1]
        held_yaw = env._heading_yaw_target(requested_yaw, ang_vel[:, 2]).clamp(
            min=env.command_lower[1], max=env.command_upper[1]
        )
        bypass_heading = None
        if env.special_command_sampler is not None and env.auto_commands:
            bypass_heading = env.special_command_sampler.is_special_active()
        if getattr(env, "terrain_commands_enabled", False) and env.auto_commands:
            spin_active = env.terrain_spinning
            bypass_heading = spin_active if bypass_heading is None else (bypass_heading | spin_active)
        if bypass_heading is not None and bypass_heading.any():
            yaw = env._yaw_from_quat(env.robot.get_quat())
            on_flat = (env.surface == 0) if hasattr(env, "surface") else torch.ones_like(bypass_heading)
            env.heading_target.copy_(torch.where(bypass_heading & on_flat, yaw, env.heading_target))
            env.heading_turning.copy_(torch.where(bypass_heading, torch.ones_like(bypass_heading), env.heading_turning))
            held_yaw = torch.where(bypass_heading, requested_yaw, held_yaw)
        targets[:, 1] = held_yaw
    if env.recovery_fraction:
        targets[:, :2].masked_fill_(env.recovery_steps[:, None] >= 0, 0.0)
    vx, yaw = targets[:, :2].unbind(1)
    demand = torch.sqrt((vx / MAX_FORWARD_SPEED).square() + (yaw / YAW_REFERENCE).square())
    # The chassis reduces combined demand as turning unloads the inside wheel.
    lateral_accel = (lin_vel[:, 0] * ang_vel[:, 2]).abs()
    current_height = env.measured_heights.mean(dim=1).clamp(0.10, 0.335)
    radius = (1.0 - lateral_accel * current_height / (9.81 * env.half_track)).clamp(0.05, 1.0)
    outer_wheel = vx.abs() + env.half_track * yaw.abs()
    scale = torch.minimum(
        radius / demand.clamp_min(1e-6),
        env.wheel_target_limit * env.wheel_radius / outer_wheel.clamp_min(1e-6),
    )
    desired = targets[:, :2] * scale.clamp_max(1.0)[:, None]
    direct_ep = getattr(env, "direct_episode", None)
    if not slew or (env.auto_commands and env.direct_velocity_commands and (direct_ep is None or direct_ep.all())):
        env.commands[:, :2] = desired
    else:
        motion_delta = desired - env.commands[:, :2]
        step_limit = env.motion_step
        if env.manual_brake_acceleration is not None:
            current_vx = env.commands[:, 0]
            desired_vx = desired[:, 0]
            braking = (current_vx.abs() > 1e-6) & (
                (desired_vx.abs() < current_vx.abs()) | ((desired_vx * current_vx) < 0.0)
            )
            if braking.any():
                step_limit = step_limit.clone()
                step_limit[:, 0] = torch.where(
                    braking, env.manual_brake_acceleration * env.control_dt, step_limit[:, 0]
                )
        if not env.auto_commands:
            # Let manual steering slew independently of the forward ramp while enforcing the wheel-speed bound.
            next_motion = env.commands[:, :2] + motion_delta.clamp(min=-step_limit, max=step_limit)
            next_outer = next_motion[:, 0].abs() + env.half_track * next_motion[:, 1].abs()
            wheel_scale = (env.wheel_target_limit * env.wheel_radius / next_outer.clamp_min(1e-6)).clamp_max(1.0)
            env.commands[:, :2] = next_motion * wheel_scale[:, None]
        else:
            # Move along one segment so the wheel-speed bound holds during reversals too.
            steps_needed = (motion_delta.abs() / step_limit).amax(dim=1).clamp_min(1.0)
            slewed = env.commands[:, :2] + motion_delta / steps_needed[:, None]
            if env.direct_velocity_commands and direct_ep is not None:
                env.commands[:, :2] = torch.where(direct_ep[:, None], desired, slewed)
            else:
                env.commands[:, :2] = slewed
    height_error = targets[:, 2] - env.commands[:, 2]
    height_delta = height_error.clamp(-env.height_step, env.height_step)
    if env.height_acceleration:
        dt = env.control_dt
        max_rate_change = env.height_acceleration * dt
        # Leave one control tick of travel in addition to the continuous braking distance.
        stopping_rate = (max_rate_change ** 2 + 2.0 * env.height_acceleration * height_error.abs()).sqrt()
        desired_rate = height_error.sign() * (stopping_rate - max_rate_change).clamp_max(env.height_step / dt)
        final_rate = height_error / dt
        can_finish = (final_rate.abs() <= max_rate_change) & (
            (final_rate - env.height_rate).abs() <= max_rate_change
        )
        desired_rate = torch.where(can_finish, final_rate, desired_rate)
        height_rate = env.height_rate + (desired_rate - env.height_rate).clamp(-max_rate_change, max_rate_change)
        next_height = (env.commands[:, 2] + height_rate * dt).clamp(env.height_low, env.height_high)
        height_delta = next_height - env.commands[:, 2]
    env.commands[:, 2] += height_delta
    env.height_rate.copy_(height_delta / env.control_dt)


def step_commands(env, done, lin_vel, ang_vel, pitch, extras):
    env.command_time -= env.control_dt
    step_transition_stop(env, done, lin_vel, pitch, extras)
    due_motion = (env.command_time <= 0).nonzero(as_tuple=True)[0]
    env._sample_motion_goals(due_motion)
    due_height = due_motion
    if env.speed_maneuvers or env.special_command_sampler is not None:
        env.height_time -= env.control_dt
        due_height = (env.height_time <= 0).nonzero(as_tuple=True)[0]
    env._sample_height_goals(due_height)
    env._advance_commands(lin_vel, ang_vel)
