"""Low-stance unpowered touchdown recovery training sub-task."""

from math import ceil, cos, radians

import torch


def init_low_recovery(env, settings):
    if not env.recovery_fraction:
        return
    env.recovery_poses = env.home_qpos.new_tensor(env.recovery_poses)
    recovery = settings["low_recovery"]
    env.recovery_unpowered_steps = ceil(recovery["unpowered_seconds"] / env.control_dt)
    env.recovery_deadline_steps = ceil(recovery["deadline_seconds"] / env.control_dt)
    env.recovery_hold_steps = ceil(0.24 / env.control_dt)
    if env.recovery_deadline_steps < env.recovery_unpowered_steps + env.recovery_hold_steps:
        raise ValueError("Low recovery deadline must leave at least 0.24 s after touchdown")
    env.recovery_steps = torch.full_like(env.episode_length_buf, -1)
    env.recovery_stable_steps = torch.zeros_like(env.recovery_steps)
    env.recovery_succeeded = torch.zeros_like(env.recovery_steps, dtype=torch.bool)
    env.recovery_failed = torch.zeros_like(env.recovery_succeeded)
    env.recovery_counts = env.recovery_steps.new_zeros(2)
    env.recovery_zero_force = env.home_qpos.new_zeros((env.num_envs, env.num_actions))


def sample_recovery_starts(env, ids, qpos, stance_height):
    env.recovery_steps[ids] = -1
    env.recovery_stable_steps[ids] = 0
    env.recovery_succeeded[ids] = False
    env.recovery_failed[ids] = False
    if not env.auto_commands:
        return
    selected = (env.surface[ids] == 0) & (torch.rand(ids.numel(), device=env.device) < env.recovery_fraction)
    reset_rows = selected.nonzero(as_tuple=True)[0]
    choices = torch.randint(len(env.recovery_poses), (reset_rows.numel(),), device=env.device)
    poses = env.recovery_poses[choices]
    # Reuse the lane position and terrain elevation. The wheels begin clear of the flat surface.
    height = env.pose_indices[0]
    qpos[reset_rows, height] += poses[:, 0] + 0.07 - stance_height[reset_rows]
    qpos[reset_rows[:, None], env.pose_indices[1:]] = poses[:, 1:]
    env.recovery_steps[ids[reset_rows]] = 0


def mask_unpowered_actions(env, actions):
    if not env.recovery_fraction:
        return actions
    unpowered = (env.recovery_steps >= 0) & (env.recovery_steps < env.recovery_unpowered_steps)
    return actions.masked_fill(unpowered[:, None], 0.0)


def apply_recovery_zero_force(env, actions):
    if not env.recovery_fraction:
        return actions
    unpowered = (env.recovery_steps >= 0) & (env.recovery_steps < env.recovery_unpowered_steps)
    ids = unpowered.nonzero(as_tuple=True)[0]
    if ids.numel():
        env.robot.control_dofs_force(env.recovery_zero_force[ids], env.action_dofs, envs_idx=ids)
    return actions.masked_fill(unpowered[:, None], 0.0)


def check_recovery(env, fallen, out_of_bounds, gravity, contact, height_target):
    recovering = env.recovery_steps >= 0
    env.recovery_steps += recovering
    wheels_loaded = (contact[:, env.wheel_links, 2] > 5.0).all(dim=1)
    stable = (recovering & (env.recovery_steps > env.recovery_unpowered_steps) & ~fallen
              & (gravity[:, 2] < -cos(radians(10.0))) & wheels_loaded
              & ((env.body_clearance - height_target).abs() < 0.04)
              & (env.robot.get_ang().norm(dim=1) < 0.8))
    env.recovery_stable_steps.copy_(torch.where(stable, env.recovery_stable_steps + 1, 0))
    env.recovery_succeeded.copy_(env.recovery_stable_steps >= env.recovery_hold_steps)
    env.recovery_failed.copy_(recovering & ~env.recovery_succeeded
                              & ((env.recovery_steps >= env.recovery_deadline_steps) | out_of_bounds))
    env.recovery_steps.masked_fill_(env.recovery_succeeded | env.recovery_failed, -1)
    return torch.where(recovering, env.recovery_failed, fallen)


def record_recovery_counts(env, extras):
    if not env.recovery_fraction:
        return
    resolved = env.recovery_succeeded | env.recovery_failed
    env.recovery_counts += torch.stack((resolved.sum(), env.recovery_succeeded.sum()))
    extras["recovery_succeeded"] = env.recovery_succeeded.clone()
    extras["recovery_failed"] = env.recovery_failed.clone()
