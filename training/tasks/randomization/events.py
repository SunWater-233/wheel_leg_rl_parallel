"""External disturbance events: velocity pushes, persistent wrenches, reset perturbations, and air drops."""

import torch

from tasks.terrains.lanes import ScutGroundLanes


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


def euler_zyx_to_quat(roll, pitch, yaw):
    cr, sr = (0.5 * roll).cos(), (0.5 * roll).sin()
    cp, sp = (0.5 * pitch).cos(), (0.5 * pitch).sin()
    cy, sy = (0.5 * yaw).cos(), (0.5 * yaw).sin()
    return torch.stack((
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ), dim=-1)


def init_events(env, settings):
    env.events_cfg = settings.get("events", {"enabled": False})
    env.events_enabled = bool(env.events_cfg.get("enabled", False))
    env.base_dofs = list(env.robot.get_joint("base_link").dofs_idx_local)
    if env.events_enabled:
        env.push_time = env.commands.new_zeros(env.num_envs)
        env.wrench_time = env.commands.new_zeros(env.num_envs)
        env.ext_wrench = env.commands.new_zeros((env.num_envs, 6))


def sample_event_wrenches(env, ids):
    if ids.numel() == 0:
        return
    cfg = env.events_cfg
    n = ids.numel()
    wlow, whigh = cfg["wrench_interval_seconds"]
    env.wrench_time[ids] = wlow + torch.rand(n, device=env.device) * (whigh - wlow)
    fmax = env.ext_wrench.new_tensor(cfg["external_force_xyz"])
    tmax = env.ext_wrench.new_tensor(cfg["external_torque_xyz"])
    limits = torch.cat((fmax, tmax))
    wrench = (2.0 * torch.rand((n, 6), device=env.device) - 1.0) * limits
    zero_mask = torch.rand(n, device=env.device) < cfg["wrench_zero_probability"]
    if env.recovery_fraction:
        zero_mask = zero_mask | (env.recovery_steps[ids] >= 0)
    wrench.masked_fill_(zero_mask[:, None], 0.0)
    env.ext_wrench[ids] = wrench


def apply_external_wrench(env):
    if getattr(env, "events_enabled", False) and env.auto_commands:
        env.robot.control_dofs_force(env.ext_wrench, env.base_dofs)


def perturb_stance_height(env, ids, height):
    if getattr(env, "events_enabled", False) and env.auto_commands and env.events_cfg["reset_joint_pos_rad"] > 0:
        # Perturb stance height within closed-chain pose_table so all 12 4-bar equalities remain exact.
        h_jitter = (2.0 * torch.rand(ids.numel(), device=env.device) - 1.0) * (
            0.5 * env.events_cfg["reset_joint_pos_rad"]
        )
        return (height + h_jitter).clamp(env.height_low, env.height_high)
    return height


def apply_reset_events(env, ids, qpos):
    """Apply air-drop elevation, roll/pitch/yaw jitter, and initial base velocity perturbations."""
    air_drop_rows = None
    air_drop_vel = None
    init_base_vel = None
    if not (getattr(env, "events_enabled", False) and env.auto_commands):
        return air_drop_rows, air_drop_vel, init_base_vel

    ev = env.events_cfg
    can_perturb = torch.ones(ids.numel(), dtype=torch.bool, device=env.device)
    if env.recovery_fraction:
        can_perturb &= env.recovery_steps[ids] < 0
    on_flat = (env.surface[ids] == 0) if env.lanes is not None else torch.ones_like(can_perturb)
    if ev["air_drop_flat_fraction"] > 0.0:
        air_mask = on_flat & can_perturb & (torch.rand(ids.numel(), device=env.device) < ev["air_drop_flat_fraction"])
        air_drop_rows = air_mask.nonzero(as_tuple=True)[0]
        if air_drop_rows.numel():
            dlow, dhigh = ev["air_drop_height_range"]
            drop_dz = dlow + torch.rand(air_drop_rows.numel(), device=env.device) * (dhigh - dlow)
            qpos[air_drop_rows, env.pose_indices[0]] += drop_dz
            if isinstance(env.lanes, ScutGroundLanes):
                qpos[air_drop_rows, env.base_xy[0]] = (
                    (2.0 * torch.rand(air_drop_rows.numel(), device=env.device) - 1.0) * 2.0
                )
    pose_perturb = can_perturb & (torch.rand(ids.numel(), device=env.device) < 0.75)
    if pose_perturb.any():
        rp_lim = ev["reset_roll_pitch_rad"]
        y_lim = ev["reset_yaw_rad"]
        roll = ((2.0 * torch.rand(ids.numel(), device=env.device) - 1.0) * rp_lim) * pose_perturb
        pitch = ((2.0 * torch.rand(ids.numel(), device=env.device) - 1.0) * rp_lim) * pose_perturb
        dyaw = ((2.0 * torch.rand(ids.numel(), device=env.device) - 1.0) * y_lim) * pose_perturb
        dq = euler_zyx_to_quat(roll, pitch, dyaw)
        qpos[:, env.base_quat] = quat_mul(qpos[:, env.base_quat], dq)
        lift = (env.half_track * roll.abs().sin() + 0.06 * pitch.abs().sin() + 0.006) * pose_perturb
        qpos[:, env.pose_indices[0]] += lift
    init_base_vel = qpos.new_zeros((ids.numel(), 6))
    vel_perturb = can_perturb & (torch.rand(ids.numel(), device=env.device) < 0.50)
    if vel_perturb.any():
        lv = ev["reset_lin_vel"]
        av = ev["reset_ang_vel"]
        v_rand = (2.0 * torch.rand((ids.numel(), 6), device=env.device) - 1.0)
        v_scales = qpos.new_tensor((lv, lv, 0.5 * lv, av, av, av))
        init_base_vel = (v_rand * v_scales) * vel_perturb[:, None]
    if air_drop_rows is not None and air_drop_rows.numel():
        vmax = max(0.55, ev["air_drop_speed_max"])
        vmag = 0.55 + torch.rand(air_drop_rows.numel(), device=env.device) * (vmax - 0.55)
        vsign = torch.where(torch.rand_like(vmag) < 0.5, -1.0, 1.0)
        air_drop_vel = vsign * vmag
        row_yaw = env._yaw_from_quat(qpos[air_drop_rows][:, env.base_quat])
        init_base_vel[air_drop_rows, 0] = air_drop_vel * row_yaw.cos()
        init_base_vel[air_drop_rows, 1] = air_drop_vel * row_yaw.sin()
        init_base_vel[air_drop_rows, 2] = 0.0
    return air_drop_rows, air_drop_vel, init_base_vel


def reset_event_timers(env, ids):
    if not getattr(env, "events_enabled", False):
        return
    plow, phigh = env.events_cfg["push_interval_seconds"]
    env.push_time[ids] = plow + torch.rand(ids.numel(), device=env.device) * (phigh - plow)
    if env.auto_commands:
        sample_event_wrenches(env, ids)
    else:
        env.ext_wrench[ids] = 0.0


def apply_air_drop_braking(env, ids, air_drop_rows, air_drop_vel):
    if air_drop_rows is None or not air_drop_rows.numel():
        return
    brake_prob = env.events_cfg["air_drop_brake_probability"]
    do_brake = torch.rand(air_drop_rows.numel(), device=env.device) < brake_prob
    if not do_brake.any():
        return
    b_rows = air_drop_rows[do_brake]
    b_ids = ids[b_rows]
    b_vel = air_drop_vel[do_brake]
    env.commands[b_ids, 0] = b_vel
    env.goals[b_ids, 0] = 0.0
    env.goals[b_ids, 1] = 0.0
    env.direct_episode[b_ids] = False
    env.command_time[b_ids] = 2.6
    if env.scut_reward.braking_cost is not None:
        env.scut_reward.braking_cost.previous_forward[b_ids] = b_vel
        env.scut_reward.braking_cost.reference_forward[b_ids] = b_vel


def step_events(env, done):
    if not getattr(env, "events_enabled", False):
        return
    env.push_time -= env.control_dt
    env.wrench_time -= env.control_dt
    active_env = ~done
    if env.recovery_fraction:
        active_env = active_env & (env.recovery_steps < 0)
    due_push = ((env.push_time <= 0.0) & active_env).nonzero(as_tuple=True)[0]
    if due_push.numel():
        ev = env.events_cfg
        plow, phigh = ev["push_interval_seconds"]
        env.push_time[due_push] = plow + torch.rand(due_push.numel(), device=env.device) * (phigh - plow)
        bvel = env.robot.get_dofs_velocity(env.base_dofs, envs_idx=due_push)
        dv_xy = (2.0 * torch.rand((due_push.numel(), 2), device=env.device) - 1.0) * ev["push_vel_xy"]
        dw_xy = (2.0 * torch.rand((due_push.numel(), 2), device=env.device) - 1.0) * ev["push_ang_vel_xy"]
        bvel[:, :2] += dv_xy
        bvel[:, 3:5] += dw_xy
        env.robot.set_dofs_velocity(bvel, env.base_dofs, envs_idx=due_push)
    due_wrench = ((env.wrench_time <= 0.0) & ~done).nonzero(as_tuple=True)[0]
    if due_wrench.numel():
        sample_event_wrenches(env, due_wrench)
