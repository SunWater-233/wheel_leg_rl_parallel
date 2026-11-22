"""Rewards active in SCUT Wheelbipe V14 Rough-v1, adapted to V0.2 actuators and contacts."""

import torch
from genesis.utils.geom import inv_transform_by_quat


class ScutMotionCost:
    """Shared Rough-v1 action, joint motion and wheel alignment penalties."""

    def __init__(self, env):
        self.env = env
        self.weights = env.cfg["locomotion"]["stability"]
        self.previous_actions = torch.zeros_like(env.last_actions)
        self.previous_joint_velocity = torch.zeros_like(env.last_actions)

    def reset(self, ids):
        self.previous_actions[ids] = 0.0
        self.previous_joint_velocity[ids] = 0.0

    def __call__(self, actions, joint_velocity, wheels, quat):
        env = self.env
        weights = self.weights
        action_delta = actions - env.last_actions
        curvature = (action_delta - (env.last_actions - self.previous_actions)).square()
        acceleration = ((joint_velocity - self.previous_joint_velocity) / env.control_dt).square()
        velocity = joint_velocity.square()
        self.previous_actions.copy_(env.last_actions)
        self.previous_joint_velocity.copy_(joint_velocity)

        wheel_separation = (wheels[:, 0] - wheels[:, 1]).contiguous()
        wheel_gap = inv_transform_by_quat(wheel_separation, quat)[:, 0].abs()
        return (
            weights["action_rate"] * action_delta.square().sum(dim=1)
            + weights["action_smoothness_leg"] * curvature[:, :4].sum(dim=1)
            + weights["action_smoothness_wheel"] * curvature[:, 4:].sum(dim=1)
            + weights["leg_joint_acc"] * acceleration[:, :4].sum(dim=1)
            + weights["wheel_acc"] * acceleration[:, 4:].sum(dim=1)
            + weights["leg_joint_vel"] * velocity[:, :4].sum(dim=1)
            + weights["wheel_vel"] * velocity[:, 4:].sum(dim=1)
            + weights["no_fork"] * (wheel_gap > weights["no_fork_distance"])
            + (weights["no_fork_square_scale"] * wheel_gap).square()
        )


class ScutRoughReward:
    def __init__(self, env):
        self.env = env
        self.motion_cost = ScutMotionCost(env)
        self.air_time = env.commands.new_zeros(env.num_envs)
        self.wheel_contact_time = env.commands.new_zeros((env.num_envs, 2))
        self.body_contact_time = torch.zeros_like(self.air_time)
        self.nonwheel_links = [i for i in range(env.robot.n_links) if i not in env.wheel_links]
        self.knee_dofs = env._dofs(("knee_L", "knee_R"))
        lower, upper = env.robot.get_dofs_limit(self.knee_dofs)
        margin = 0.03 * (upper - lower)
        self.knee_lower = lower + margin
        self.knee_upper = upper - margin
        scan_axis = env.commands.new_tensor((-0.01, 0.0, 0.01))
        xy = torch.cartesian_prod(scan_axis, scan_axis)
        self.scanner_offsets = torch.cat((xy, torch.zeros_like(xy[:, :1])), dim=1)
        self.minimum_body_clearance = env.height_low + env.body_height_offset

    def reset(self, ids):
        self.motion_cost.reset(ids)
        self.air_time[ids] = 0.0
        self.wheel_contact_time[ids] = 0.0
        self.body_contact_time[ids] = 0.0

    def measure_clearance(self, fallback):
        env = self.env
        base_pos = env.robot.get_pos()
        ground = env.ground_height_at(base_pos[:, None] + self.scanner_offsets)
        valid = torch.isfinite(ground)
        ground_mean = ground.masked_fill(~valid, 0.0).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return torch.where(valid.any(dim=1), base_pos[:, 2] - ground_mean, fallback)

    def _airborne(self, wheels, contact):
        env = self.env
        dt = env.control_dt
        wheel_force = contact[:, env.wheel_links].norm(dim=-1)
        wheel_clearance = wheels[..., 2] - env.ground_height_at(wheels) - env.wheel_radius
        # Map the body-height gate to the V0.2 stance. Keep SCUT's wheel clearance and contact timings.
        enter = (env.body_clearance > self.minimum_body_clearance + 0.08) & (wheel_clearance > 0.08).all(dim=1)
        enter &= (wheel_force < 1.0).all(dim=1)
        active = self.air_time > 0.0
        wheel_loaded = (wheel_force > 20.0) & (wheel_clearance + env.wheel_radius < 0.15)
        body_loaded = (contact[:, env.body_links].norm(dim=-1) > 5.0).any(dim=1)
        self.wheel_contact_time.copy_(torch.where(active[:, None] & wheel_loaded, self.wheel_contact_time + dt, 0.0))
        self.body_contact_time.copy_(torch.where(active & body_loaded, self.body_contact_time + dt, 0.0))
        wheel_time = self.wheel_contact_time.amax(dim=1)
        finished = (wheel_time >= 0.30 - 1e-6) | (self.body_contact_time >= 0.25 - 1e-6) | (self.air_time >= 1.0 - dt)
        airborne = enter | (active & ~finished)
        self.air_time.copy_(torch.where(airborne, self.air_time + dt, 0.0))
        return airborne, wheel_time

    def _airborne_reward(self, airborne, wheel_time, offsets, quat, forward, lin_vel, wheel_vel, wheel_torque):
        env = self.env
        before_contact = airborne & (wheel_time < 0.02 - 1e-6)
        low_torque = torch.exp(-(wheel_torque.abs().amax(dim=1) / 1.5).square())
        # Wheel speeds use the mirror signs from the drive, so forward is positive on both sides.
        direction = forward.sign()
        moving = (forward.abs() > 1.0) & (lin_vel[:, 0] * direction > 1.0)
        wheel_progress = (wheel_vel * env.wheel_direction * direction[:, None] / 10.0).clamp(0.0, 1.0)
        wheel_progress = wheel_progress.amin(dim=1)
        shortly_after_contact = airborne & (wheel_time >= 0.02 - 1e-6) & (wheel_time < 0.05)

        w, x, y, z = quat.unbind(dim=1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        longitudinal = offsets[..., 0] * yaw.cos()[:, None] + offsets[..., 1] * yaw.sin()[:, None]
        wheels_below = (offsets[..., 2] < env.wheel_radius - self.minimum_body_clearance).all(dim=1)
        centered = torch.exp(-longitudinal.square() / 0.02).mean(dim=1)
        centering_mask = before_contact & (self.body_contact_time < 0.02 - 1e-6) & wheels_below
        knees = env.robot.get_dofs_position(self.knee_dofs)
        knee_overrun = (self.knee_lower - knees).clamp_min(0.0) + (knees - self.knee_upper).clamp_min(0.0)
        return (
            20.0 * before_contact * low_torque
            + 10.0 * (before_contact & moving) * wheel_progress
            - 10.0 * (shortly_after_contact & moving) * (1.0 - wheel_progress)
            + 10.0 * centering_mask * centered
            - 10.0 * airborne * knee_overrun.sum(dim=1)
        )

    def __call__(self, lin_vel, ang_vel, gravity, contact, fallen, actions):
        env = self.env
        joint_velocity = env.robot.get_dofs_velocity(env.action_dofs)
        torque = env.robot.get_dofs_control_force(env.action_dofs)
        wheels = torch.stack([wheel.get_anchor_pos() for wheel in env.wheels], dim=1)
        offsets = wheels - env.robot.get_pos()[:, None]
        quat = env.robot.get_quat().contiguous()
        motion_cost = self.motion_cost(actions, joint_velocity, wheels, quat)
        airborne, wheel_time = self._airborne(wheels, contact)
        in_air = airborne.float()
        bad_contact = (contact[:, self.nonwheel_links].norm(dim=-1) > 3.0).any(dim=1)

        forward, yaw = env.commands[:, :2].unbind(dim=1)
        stand = forward.abs() < 0.1
        forward_target = forward.masked_fill(stand, 0.0)
        yaw_target = yaw.masked_fill(yaw.abs() < 0.1, 0.0)
        gx, gy = gravity[:, :2].unbind(dim=1)
        forward_error = forward_target - lin_vel[:, 0] * gravity[:, 1:].norm(dim=1)
        yaw_error = yaw_target - ang_vel[:, 2]
        tilt_scale = 2.0 + 2.0 * torch.exp(-forward.square() / 3.0)
        height_error = env.body_clearance - (env.commands[:, 2] + env.body_height_offset)

        tracking = (
            1.25 * torch.exp(-forward_error.clamp(-1.0, 1.0).square() / 0.5)
            - (0.25 * forward_error).square()
            + torch.exp(-yaw_error.clamp(-0.8, 0.8).square() / 0.25) - (0.5 * yaw_error).square()
            - stand * lin_vel[:, :2].abs().sum(dim=1)
            + torch.exp(-gx.square() / 0.02) + torch.exp(-gy.square() / 0.01)
            - 2.0 * ((~airborne) * (tilt_scale * gx).square() + (tilt_scale * gy).square())
            + torch.exp(-height_error.clamp(-0.15, 0.15).square() / 0.001)
            - (~airborne) * (10.0 * height_error).square()
        )
        wheel_vel = joint_velocity[:, 4:]
        wheel_torque = torque[:, 4:]
        regularization = (
            motion_cost
            + 1e-5 * torque.square().sum(dim=1) + 1e-5 * (wheel_torque * wheel_vel).clamp_min(0.0).sum(dim=1)
            + 0.50 * lin_vel[:, 2].square() + 0.05 * ang_vel[:, :2].square().sum(dim=1)
            + (2.0 + 48.0 * in_air) * bad_contact
            + (200.0 + 1000.0 * in_air) * fallen
        )
        return tracking - regularization + self._airborne_reward(
            airborne, wheel_time, offsets, quat, forward, lin_vel, wheel_vel, wheel_torque
        )
