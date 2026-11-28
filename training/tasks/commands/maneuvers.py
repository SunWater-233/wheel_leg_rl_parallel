"""Legacy multi-phase speed, braking, launch, and height maneuver state machine."""

import torch


def sample_maneuvers(env, ids):
    count = 5 if env.braking_maneuvers else 4
    if env.maneuver_weights.numel():
        env.maneuver[ids] = torch.multinomial(env.maneuver_weights, ids.numel(), replacement=True)
    else:
        env.maneuver[ids] = torch.randint(count, (ids.numel(),), device=env.device)
        if env.launch_maneuvers:
            launch = (torch.rand(ids.numel(), device=env.device) < 0.30) & (env.surface[ids] == 0)
            env.maneuver[ids] = env.maneuver[ids].masked_fill(launch, env.launch_mode)
        if env.height_maneuvers:
            height_only = (torch.rand(ids.numel(), device=env.device) < 0.25) & (env.surface[ids] == 0)
            env.maneuver[ids] = env.maneuver[ids].masked_fill(height_only, env.height_mode)
    env.maneuver_phase[ids] = 0
    env.maneuver_reverse[ids] = torch.rand((ids.numel(), 2), device=env.device) < 0.5


def compute_maneuver_goals(env, ids, motion, duration, hold_sample):
    # Each episode repeats one pattern. Phase zero starts in a randomly chosen direction.
    steady, straight, spin, slalom, corner_brake = range(5)
    mode = env.maneuver[ids]
    phase = env.maneuver_phase[ids]
    reverse, clockwise = env.maneuver_reverse[ids].unbind(dim=1)
    forward_min, yaw_min = env.command_lower[:2].unbind()
    forward_max, yaw_max = env.command_upper[:2].unbind()

    # Run, stop, run, reverse directly, stop, run; the cycle boundary reverses directly again.
    forward = torch.where(reverse ^ (phase >= 3), forward_min, forward_max)
    forward = forward * ((mode == straight) & (phase.remainder(3) != 1))
    turn_speed = 0.65 * torch.where(reverse, forward_min, forward_max)
    forward = torch.where(mode == slalom, turn_speed, forward)
    yaw = torch.where(clockwise ^ (phase.remainder(2) == 1), yaw_min, yaw_max)
    yaw = yaw * ((mode == spin) | (mode == slalom))
    hold_seconds = 0.25 + 0.30 * hold_sample
    if env.speed_levels.numel():
        stop = (mode == straight) & (phase.remainder(3) == 1)
        hold_seconds = torch.where(stop, 0.75 + 0.40 * hold_sample, hold_seconds)
    next_phase = (phase + 1).remainder(6)
    if env.braking_maneuvers:
        braking = mode == corner_brake
        segment = phase.remainder(4)
        # Cruise, enter the turn at speed, brake in the turn, then settle at rest.
        approach = torch.where(reverse, forward_min, forward_max) * (segment < 2)
        turn = torch.where(clockwise ^ (phase >= 4), yaw_min, yaw_max)
        turn = turn * ((segment == 1) | (segment == 2))
        forward = torch.where(braking, approach, forward)
        yaw = torch.where(braking, turn, yaw)
        hold_seconds = torch.where(braking & (segment == 0), 1.45 + 1.0 * hold_sample, hold_seconds)
        hold_seconds = torch.where(braking & (segment == 3), 0.75 + 0.45 * hold_sample, hold_seconds)
        next_phase = torch.where(braking, (phase + 1).remainder(8), next_phase)
    if env.launch_maneuvers:
        launch = mode == env.launch_mode
        segment = phase.remainder(4)
        moving = segment.remainder(2) == 0
        switch_direction = segment >= 2
        launch_hold = 0.90 + 0.30 * hold_sample
        if env.launch_stop_seconds.numel():
            # Split the launch cases between repeated taps in one direction and reversals.
            switch_direction &= clockwise
            slots = (hold_sample * len(env.launch_stop_seconds)).long().clamp_max(len(env.launch_stop_seconds) - 1)
            launch_hold = torch.where(moving, launch_hold, env.launch_stop_seconds[slots])
        launch_speed = torch.where(reverse ^ switch_direction, forward_min, forward_max)
        forward = torch.where(launch, launch_speed * moving, forward)
        hold_seconds = torch.where(launch, launch_hold, hold_seconds)
        next_phase = torch.where(launch, (phase + 1).remainder(4), next_phase)
    if env.height_maneuvers:
        height_only = mode == env.height_mode
        cruise = torch.where(reverse, -0.5, 0.5) * clockwise
        forward = torch.where(height_only, cruise, forward)
        yaw = yaw.masked_fill(height_only, 0.0)

    if env.speed_levels.numel():
        forward_cap = env.episode_speed_limit[ids]
        forward = forward.clamp(-forward_cap, forward_cap)
    target = torch.stack((forward, yaw), dim=1)
    # Reserve time for the slew-limited transition and the dwell.
    ramp_seconds = ((target - env.commands[ids, :2]).abs() / env.motion_step[ids]).amax(dim=1) * env.control_dt
    maneuver_seconds = ramp_seconds + hold_seconds
    if env.launch_maneuvers and env.launch_press_seconds.numel():
        slots = (hold_sample * len(env.launch_press_seconds)).long().clamp_max(len(env.launch_press_seconds) - 1)
        maneuver_seconds = torch.where(launch & moving, env.launch_press_seconds[slots], maneuver_seconds)
    env.maneuver_phase[ids] = next_phase
    return (
        torch.where((mode == steady)[:, None], motion, target),
        torch.where(mode == steady, duration, maneuver_seconds),
    )
