"""Observation assembly, sensor noise, actor-only IMU delay, history stack, and critic terrain scan."""

from math import tau

from genesis.utils.geom import inv_transform_by_quat
from tensordict import TensorDict
import torch


def leg_projections(env, gravity):
    """Return leg heights and signed wheel offsets ahead of each hip."""
    quat = env.robot.get_quat().contiguous()
    # Match L0*cos(theta) in the firmware sagittal plane. Ignore the lateral anchor offset.
    up = -gravity
    up_x = up[:, 0:1]
    up_z = up[:, 1:].norm(dim=1, keepdim=True)
    vectors = []
    for hip, wheel in zip(env.hips, env.wheels, strict=True):
        offset = (hip.get_anchor_pos() - wheel.get_anchor_pos()).contiguous()
        vectors.append(inv_transform_by_quat(offset, quat))
    offsets = torch.stack(vectors, dim=1)
    x, _, z = offsets.unbind(dim=2)
    return x * up_x + z * up_z, z * up_x - x * up_z


def enable_observation_noise(env):
    settings = env.cfg["observation_noise"]
    stds = [settings["gyro"]] * 3 + [settings["gravity"]] * 3 + [settings["joint_position"]] * 4
    stds += [0.1 * settings["joint_velocity"]] * 6
    if any(stds):
        # Keep some clean observations and vary the sensor noise scale across environments.
        level = env.commands.new_empty((env.num_envs, 1)).uniform_(0.5, 1.5)
        level.masked_fill_(torch.rand_like(level) < 0.25, 0.0)
        env.observation_noise = env.commands.new_tensor(stds) * level


def critic_terrain_observations(env):
    # Scan both travel directions along each wheel track in the horizontal plane.
    yaw = env._yaw_from_quat(env.robot.get_quat())
    heading = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), dim=1)
    wheels = torch.stack([wheel.get_anchor_pos() for wheel in env.wheels], dim=1)
    offsets = heading[:, None, :] * env.critic_scan_offsets[None, :, None]
    points = wheels[:, :, None, :] + offsets[:, None, :, :]
    ground = env.ground_height_at(points)
    relative_height = ground - (wheels[:, :, 2:] - env.wheel_radius)
    relative_height = relative_height.nan_to_num().clamp(-0.20, 0.20)
    return env.critic_wheel_contacts.to(wheels.dtype), 10.0 * relative_height.flatten(1)


def build_locomotion_observations(env, lin_vel, ang_vel, gravity):
    heights = env.leg_heights(gravity)
    nominal = env._poses(env.commands[:, 2])[:, env.leg_columns]
    joint_error = env.robot.get_dofs_position(env.leg_dofs) - nominal
    # The driven leg hinges can accumulate whole turns during recovery.
    joint_pos = (joint_error - tau * torch.round(joint_error / tau)) * env.leg_direction
    joint_vel = env.robot.get_dofs_velocity(env.action_dofs) * env.action_direction
    clean_policy = torch.cat((
        env.commands * env.motion_scales,
        env.height_rate[:, None],
        ang_vel,
        gravity,
        joint_pos,
        joint_vel * 0.1,
        env.last_actions,
        heights * 5.0,
    ), dim=1)
    critic_inputs = (clean_policy, lin_vel)
    if env.critic_scan_offsets is not None:
        critic_inputs += critic_terrain_observations(env)
    critic = torch.cat(critic_inputs, dim=1)
    imu_delay = getattr(env, "imu_obs_delay_steps", None)
    if imu_delay is not None:
        policy_ang_vel = env._imu_ang_buf[env.all_envs, imu_delay]
        policy_gravity = env._imu_grav_buf[env.all_envs, imu_delay]
        policy_heights = env.leg_heights(policy_gravity)
        policy = torch.cat((
            env.commands * env.motion_scales,
            env.height_rate[:, None],
            policy_ang_vel,
            policy_gravity,
            joint_pos,
            joint_vel * 0.1,
            env.last_actions,
            policy_heights * 5.0,
        ), dim=1)
    else:
        policy = clean_policy
    if env.observation_noise is not None:
        # The critic retains the original state; commands, history and inferred leg height stay intact.
        sensors = policy[:, 4:20]
        sensors.add_(torch.randn_like(sensors) * env.observation_noise)
    observations = {"policy": policy, "critic": critic}
    if env.actor_true_forward_speed:
        observations["body_forward_speed"] = lin_vel[:, :1].contiguous()
    if env.policy_history is not None:
        if env.history_needs_fill.any():
            fill = env.history_needs_fill
            env.policy_history[fill] = policy[fill, None, :]
            env.history_needs_fill[fill] = False
        env.last_policy_obs.copy_(policy)
        observations["policy_history"] = env.policy_history.reshape(env.num_envs, -1).clone()
    return TensorDict(observations, batch_size=[env.num_envs])


def shift_policy_history(env):
    if env.policy_history is not None:
        if env.policy_history.shape[1] > 1:
            env.policy_history[:, 1:] = env.policy_history[:, :-1].clone()
        env.policy_history[:, 0] = env.last_policy_obs
