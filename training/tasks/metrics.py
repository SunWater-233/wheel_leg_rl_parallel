"""Episode tracking metrics, TensorBoard scalars, and adaptive terrain curriculum updates."""

import torch


def init_episode_metrics(env):
    env.measured_heights = env.commands.new_full((env.num_envs, 2), env.height_nominal)
    env.posture = env.commands.new_zeros((env.num_envs, 3))
    env.posture_sums = torch.zeros_like(env.posture)
    env.roll_peaks = torch.zeros_like(env.command_time)
    env.pitch_deg = torch.zeros_like(env.command_time)
    env.pitch_sums = torch.zeros_like(env.pitch_deg)
    env.pitch_peaks = torch.zeros_like(env.pitch_deg)
    env.episode_steps = torch.zeros_like(env.episode_length_buf)
    env.height_errors = torch.zeros_like(env.command_time)
    env.leg_height_errors = torch.zeros_like(env.height_errors)
    env.velocity_errors = torch.zeros_like(env.command_time)
    env.yaw_errors = torch.zeros_like(env.command_time)
    env.leg_saturated_steps = torch.zeros_like(env.episode_steps)


def reset_episode_metrics(env, ids):
    env.episode_steps[ids] = 0
    env.height_errors[ids] = 0.0
    env.leg_height_errors[ids] = 0.0
    env.velocity_errors[ids] = 0.0
    env.yaw_errors[ids] = 0.0
    env.posture_sums[ids] = 0.0
    env.roll_peaks[ids] = 0.0
    env.pitch_sums[ids] = 0.0
    env.pitch_peaks[ids] = 0.0
    env.leg_saturated_steps[ids] = 0


def update_step_metrics_and_logs(env, extras, done, lin_vel, ang_vel):
    env.episode_steps += 1
    if env.transition_course:
        torch.maximum(env.transition_peak_command, env.commands[:, 0].abs(), out=env.transition_peak_command)
    env.velocity_errors += (lin_vel[:, 0] - env.commands[:, 0]).abs()
    env.yaw_errors += (ang_vel[:, 2] - env.commands[:, 1]).abs()
    env.posture_sums += env.posture
    torch.maximum(env.roll_peaks, env.posture[:, 0], out=env.roll_peaks)
    pitch = env.pitch_deg.abs()
    env.pitch_sums += pitch
    torch.maximum(env.pitch_peaks, pitch, out=env.pitch_peaks)
    leg_force = env.robot.get_dofs_control_force(env.leg_dofs)
    saturated = (leg_force.abs() >= 0.99 * env.leg_torque_limit).any(dim=1)
    env.leg_saturated_steps += saturated
    extras["leg_saturated"] = saturated
    extras["leg_heights"] = env.measured_heights.clone()
    extras["body_clearance"] = env.body_clearance.clone()
    extras["posture"] = env.posture.clone()
    extras["pitch_deg"] = env.pitch_deg.clone()

    finished = done.nonzero(as_tuple=True)[0]
    if finished.numel():
        count = env.episode_steps[finished]
        roll, fore_aft, height_gap = (env.posture_sums[finished] / count[:, None]).unbind(dim=1)
        extras["log"] = {
            "body_height_mae_cm": 100.0 * env.height_errors[finished] / count,
            "leg_height_mae_cm": 100.0 * env.leg_height_errors[finished] / count,
            "forward_mae_m_s": env.velocity_errors[finished] / count,
            "yaw_mae_rad_s": env.yaw_errors[finished] / count,
            "fall_rate": extras["fallen"][finished].float(),
            "leg_saturation_rate": env.leg_saturated_steps[finished] / count,
            "roll_abs_deg": roll,
            "roll_peak_deg": env.roll_peaks[finished],
            "pitch_abs_deg": env.pitch_sums[finished] / count,
            "pitch_peak_deg": env.pitch_peaks[finished],
            "leg_fore_aft_gap_cm": fore_aft,
            "leg_height_gap_cm": height_gap,
        }
        if env.terrain_traversal:
            extras["log"]["terrain_progress_m"] = env.traversal_distance[finished]
            if getattr(env, "terrain_curriculum_enabled", False) and env.auto_commands:
                obs_mask = env.surface[finished] != 0
                if obs_mask.any():
                    f_obs = finished[obs_mask]
                    dist = env.traversal_distance[f_obs]
                    fell = extras["fallen"][f_obs]
                    promote = (~fell) & (dist >= env.curriculum_promote_distance)
                    demote = fell & (dist < env.curriculum_demote_distance)
                    cur_lvl = env.terrain_level[f_obs]
                    top_lvl = env.lanes.variants - 1
                    newly_maxed = promote & (cur_lvl >= top_lvl)
                    env.terrain_maxed[f_obs] |= newly_maxed
                    next_lvl = torch.where(promote, (cur_lvl + 1).clamp_max(top_lvl), cur_lvl)
                    next_lvl = torch.where(
                        demote & ~env.terrain_maxed[f_obs], (next_lvl - 1).clamp_min(0), next_lvl
                    )
                    env.terrain_level[f_obs] = next_lvl
        if env.transition_course:
            transitioned = finished[env.surface[finished] != 0]
            if transitioned.numel():
                extras["log"]["Transition/peak_command_m_s"] = env.transition_peak_command[transitioned]
                extras["log"]["Transition/fall_rate"] = extras["fallen"][transitioned].float()
    return pitch
