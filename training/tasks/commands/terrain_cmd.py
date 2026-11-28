"""Per-terrain command sampling (SCUT obstacle agility) and transition-course stop state machine."""

import torch


def sample_terrain_command_goals(env, t_ids):
    cfg = env.terrain_cmd_cfg
    n = t_ids.numel()
    surf = env.surface[t_ids]
    is_hard = env.hard_obstacle_surfaces[surf]
    can_spin = env.spin_obstacle_surfaces[surf]
    pos = env.robot.get_pos()[t_ids]
    x = pos[:, 0]
    dy = pos[:, 1] - env.spawn_y[t_ids]
    in_zone = x.abs() <= cfg["engage_x"]

    u = torch.rand((n, 5), device=env.device)
    p_spin = cfg["spin_probability"]
    p_stop = cfg["stop_probability"]
    do_spin = in_zone & can_spin & (u[:, 0] < p_spin)
    do_stop = in_zone & ~do_spin & (u[:, 0] < p_spin + p_stop)
    do_cruise = ~(do_spin | do_stop)
    env.terrain_spinning[t_ids] = do_spin

    fast_low, fast_high = cfg["fast_forward_range"]
    hard_low, hard_high = cfg["hard_forward_range"]
    v_fast = fast_low + u[:, 1] * (fast_high - fast_low)
    v_hard = hard_low + u[:, 1] * (hard_high - hard_low)
    v_mag = torch.where(is_hard, v_hard, v_fast)
    sign_x = torch.where(env.traversal_reverse[t_ids], -1.0, 1.0)
    vx = sign_x * v_mag
    vx = torch.where(do_stop, torch.zeros_like(vx), vx)
    vx = torch.where(do_spin, (2.0 * u[:, 1] - 1.0) * 0.12, vx)

    steer = do_cruise & in_zone & (u[:, 2] >= cfg["zero_yaw_probability"])
    yaw_cap = torch.where(
        is_hard,
        vx.new_tensor(cfg["hard_yaw_max"]),
        vx.new_tensor(cfg["fast_yaw_max"]),
    )
    wz_mag = (0.25 + 0.75 * u[:, 3]) * yaw_cap
    raw_sign = torch.where(u[:, 4] < 0.5, -1.0, 1.0)
    yaw_now = env._yaw_from_quat(env.robot.get_quat()[t_ids])
    center_sign = -torch.sign(dy * vx * yaw_now.cos())
    center_sign = torch.where(center_sign == 0.0, raw_sign, center_sign)
    wz_sign = torch.where(dy.abs() > 0.55, center_sign, raw_sign)
    wz_steer = torch.where(yaw_now.abs() > 0.40, torch.zeros_like(wz_mag), wz_sign * wz_mag)
    wz = torch.where(steer, wz_steer, torch.zeros_like(vx))

    spin_low, spin_high = cfg["spin_yaw_range"]
    spin_mag = spin_low + u[:, 3] * (spin_high - spin_low)
    spin_sign = torch.where(u[:, 4] < 0.5, -1.0, 1.0)
    wz = torch.where(do_spin, spin_sign * spin_mag, wz)

    t_low, t_high = cfg["resampling_time_range"]
    dur_cruise = t_low + u[:, 3] * (t_high - t_low)
    dur = torch.where(in_zone, dur_cruise, 0.85 + 0.55 * u[:, 3])
    dur = torch.where(do_stop, 0.85 + 0.75 * u[:, 3], dur)
    dur = torch.where(do_spin, 0.90 + 0.70 * u[:, 3], dur)
    return torch.stack((vx, wz), dim=1), dur


def sample_traversal_cruise(env, ids, sample, forward_cap, surface):
    limit = torch.where(env.traversal_reverse[ids], env.command_lower[0], env.command_upper[0])
    if env.forward_drop:
        cliff = surface == env.cliff_surface
        limit = torch.where(cliff, env.command_upper[0], limit)
    if env.speed_levels.numel():
        terrain_cap = torch.minimum(forward_cap, env.terrain_speed_limit[surface])
        limit = limit.clamp(-terrain_cap, terrain_cap)
    if env.terrain_forward_limit:
        limit = limit.clamp(-env.terrain_forward_limit, env.terrain_forward_limit)
    cruise = limit * (0.65 + 0.35 * sample[:, 0])
    if env.bar_speed is not None:
        bar_cruise = limit.sign() * torch.minimum(limit.abs(), env.bar_speed[ids])
        cruise = torch.where(surface == env.bar_surface, bar_cruise, cruise)
    if env.single_side_surfaces is not None:
        # Both directions vary between two thirds of the cap and full speed.
        side_cruise = limit.clamp(-0.60, 0.60) * ((2.0 + sample[:, 0]) / 3.0)
        cruise = torch.where(env.single_side_surfaces[surface], side_cruise, cruise)
    if env.cliff_approach_speed:
        airborne = cliff & (~env.drop_touched[ids] | (env.landing_elapsed[ids] < env.landing_seconds))
        approach = env.cliff_approach_speed * (0.80 + 0.20 * sample[:, 0])
        cruise = torch.where(airborne, torch.minimum(cruise, approach), cruise)
    if env.transition_course:
        direction = torch.where(env.traversal_reverse[ids], -1.0, 1.0)
        cruise = env.episode_speed_limit[ids] * direction
        phase = env.transition_stop_phase[ids]
        restart = torch.where(cruise.abs() > 0.3, 0.45 * cruise, 0.30 * direction)
        cruise = torch.where(phase == 3, restart, cruise)
        cruise = cruise.masked_fill((phase == 2) | (phase == 4), 0.0)
    return cruise


def step_transition_stop(env, done, lin_vel, pitch, extras):
    if not env.transition_course:
        return
    phase = env.transition_stop_phase
    x = env.robot.get_pos()[:, 0]
    along = torch.where(env.traversal_reverse, -x, x)
    start = (phase == 1) & (along >= env.transition_stop_at)
    stopping = phase == 2
    if env.transition_stop_feedback:
        stable = (stopping & ~done & (env.commands[:, 0].abs() < 0.025)
                  & (lin_vel[:, :2].norm(dim=1) <= 0.08) & extras["wheel_contacts"].all(dim=1)
                  & (pitch <= 8.0) & (env.posture[:, 0] <= 8.0))
        env.transition_still_time.copy_(torch.where(stable, env.transition_still_time + env.control_dt, 0))
        env.transition_stop_elapsed += env.control_dt * stopping
        restart = stopping & (env.transition_still_time >= 0.24 - 1e-6)
        timeout = stopping & (env.transition_stop_elapsed >= 4.5) & ~restart
        # A failed stop stays at zero for the rest of the episode.
        phase.masked_fill_(timeout, 4)
    else:
        env.transition_still_time += env.control_dt * (stopping & (env.commands[:, 0].abs() < 0.025))
        restart = stopping & (env.transition_still_time >= 0.9)
    phase.masked_fill_(start, 2)
    phase.masked_fill_(restart, 3)
    env.command_time.masked_fill_(start | restart, 0.0)
