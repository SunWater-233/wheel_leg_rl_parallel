"""SCUT Wheelbipe V14 Rough-v1 rewards for the Genesis robots."""

import torch
from genesis.utils.geom import inv_transform_by_quat
from torch.nn.functional import smooth_l1_loss

CONTACT_TIME_TOLERANCE = 1e-6


class BrakingPitchCost:
    """Track commanded stops through the speed ramp and the subsequent recovery."""

    def __init__(self, commands, settings, dt):
        self.settings = settings
        self.dt = dt
        self.previous_forward = torch.zeros_like(commands[:, 0])
        self.direction = torch.zeros_like(self.previous_forward)
        self.recovery_elapsed = torch.zeros_like(self.previous_forward)
        self.distance_error = torch.zeros_like(self.previous_forward)

    def reset(self, ids):
        self.previous_forward[ids] = 0.0
        self.direction[ids] = 0.0
        self.recovery_elapsed[ids] = 0.0
        self.distance_error[ids] = 0.0

    def __call__(self, forward, goal, measured_forward, pitch_projection, pitch_rate, eligible):
        cfg = self.settings
        reference_speed = forward.abs()
        previous_speed = self.previous_forward.abs()
        stop_requested = eligible & (goal.abs() <= cfg["zero_speed"])
        start = (stop_requested & (self.direction == 0.0) & (previous_speed >= cfg["start_speed"])
                 & (reference_speed < previous_speed))
        direction = torch.where(start, self.previous_forward.sign(), self.direction)
        direction = direction.masked_fill(~stop_requested, 0.0)
        self.previous_forward.copy_(forward)

        recovering = (direction != 0.0) & (reference_speed <= cfg["zero_speed"])
        self.recovery_elapsed.copy_(torch.where(recovering, self.recovery_elapsed + self.dt, 0.0))
        active = (direction != 0.0) & (self.recovery_elapsed <= cfg["recovery_seconds"])
        self.direction.copy_(direction.masked_fill(~active, 0.0))

        # Integrate travel along the commanded braking direction against the slew-limited reference.
        measured_speed = (measured_forward * direction).clamp_min(0.0)
        distance_step = (measured_speed - reference_speed) * self.dt
        self.distance_error.copy_(torch.where(active, self.distance_error + distance_step, 0.0))
        excess_distance = (self.distance_error - cfg["distance_tolerance"]).clamp_min(0.0)
        zero = torch.zeros_like(forward)
        rate_cost = smooth_l1_loss(pitch_rate / cfg["pitch_rate_scale"], zero, reduction="none")
        pitch_cost = smooth_l1_loss(pitch_projection / cfg["recovery_pitch_scale"], zero, reduction="none")
        speed_cost = smooth_l1_loss(measured_forward / cfg["recovery_speed_scale"], zero, reduction="none")
        distance_cost = smooth_l1_loss(excess_distance / cfg["distance_scale"], zero, reduction="none")
        recovery_cost = cfg["recovery_pitch_weight"] * pitch_cost + cfg["recovery_speed_weight"] * speed_cost
        return active * (
            cfg["pitch_rate_weight"] * rate_cost
            + recovering * recovery_cost
            + cfg["distance_weight"] * distance_cost
        )


class ScutRoughReward:
    def __init__(self, env, *, airborne_minimum_height=None):
        self.env = env
        self.settings = env.cfg["scut_rough_reward"]
        braking = self.settings["braking"]
        self.braking_cost = BrakingPitchCost(env.commands, braking, env.control_dt) if braking["enabled"] else None
        self.previous_actions = torch.zeros_like(env.last_actions)
        self.previous_joint_velocity = torch.zeros_like(env.last_actions)
        self.nonwheel_links = [i for i in range(env.robot.n_links) if i not in env.wheel_links]
        self.minimum_body_clearance = airborne_minimum_height
        if self.minimum_body_clearance is not None:
            self.air_time = env.commands.new_zeros(env.num_envs)
            self.wheel_contact_time = env.commands.new_zeros((env.num_envs, 2))
            self.body_contact_time = torch.zeros_like(self.air_time)
            self.knee_dofs = env._dofs(("knee_L", "knee_R"))
            lower, upper = env.robot.get_dofs_limit(self.knee_dofs)
            airborne = self.settings["airborne"]
            margin = airborne["knee_margin_fraction"] * (upper - lower)
            self.knee_lower = lower + margin
            self.knee_upper = upper - margin
            radius = airborne["body_clearance_scan_radius"]
            scan_axis = env.commands.new_tensor((-radius, 0.0, radius))
            xy = torch.cartesian_prod(scan_axis, scan_axis)
            self.scanner_offsets = torch.cat((xy, torch.zeros_like(xy[:, :1])), dim=1)

    def reset(self, ids):
        if self.braking_cost is not None:
            self.braking_cost.reset(ids)
        self.previous_actions[ids] = 0.0
        self.previous_joint_velocity[ids] = 0.0
        if self.minimum_body_clearance is not None:
            self.air_time[ids] = 0.0
            self.wheel_contact_time[ids] = 0.0
            self.body_contact_time[ids] = 0.0

    def _motion_cost(self, actions, joint_velocity, wheels, quat):
        env = self.env
        cfg = self.settings["motion"]
        action_delta = actions - env.last_actions
        curvature = (action_delta - (env.last_actions - self.previous_actions)).square()
        acceleration = ((joint_velocity - self.previous_joint_velocity) / env.control_dt).square()
        velocity = joint_velocity.square()
        self.previous_actions.copy_(env.last_actions)
        self.previous_joint_velocity.copy_(joint_velocity)

        wheel_separation = (wheels[:, 0] - wheels[:, 1]).contiguous()
        wheel_gap = inv_transform_by_quat(wheel_separation, quat)[:, 0].abs()
        return (
            cfg["action_delta_weight"] * action_delta.square().sum(dim=1)
            + cfg["leg_action_curvature_weight"] * curvature[:, :4].sum(dim=1)
            + cfg["wheel_action_curvature_weight"] * curvature[:, 4:].sum(dim=1)
            + cfg["leg_acceleration_weight"] * acceleration[:, :4].sum(dim=1)
            + cfg["wheel_acceleration_weight"] * acceleration[:, 4:].sum(dim=1)
            + cfg["leg_velocity_weight"] * velocity[:, :4].sum(dim=1)
            + cfg["wheel_velocity_weight"] * velocity[:, 4:].sum(dim=1)
            + cfg["wheel_gap_threshold_weight"] * (wheel_gap > cfg["wheel_gap_threshold"]).float()
            + (cfg["wheel_gap_scale"] * wheel_gap).square()
        )

    def measure_clearance(self, fallback):
        env = self.env
        base_pos = env.robot.get_pos()
        ground = env.ground_height_at(base_pos[:, None] + self.scanner_offsets)
        valid = torch.isfinite(ground)
        ground_mean = ground.masked_fill(~valid, 0.0).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return torch.where(valid.any(dim=1), base_pos[:, 2] - ground_mean, fallback)

    def _airborne(self, wheels, contact):
        env = self.env
        cfg = self.settings["airborne"]
        dt = env.control_dt
        wheel_force = contact[:, env.wheel_links].norm(dim=-1)
        wheel_clearance = wheels[..., 2] - env.ground_height_at(wheels) - env.wheel_radius
        # Map the body-height gate to the V0.2 stance. Keep SCUT's wheel clearance and contact timings.
        enter = (env.body_clearance > self.minimum_body_clearance + cfg["body_clearance_margin"])
        enter &= (wheel_clearance > cfg["wheel_clearance_threshold"]).all(dim=1)
        enter &= (wheel_force < cfg["liftoff_wheel_force_threshold"]).all(dim=1)
        active = self.air_time > 0.0
        wheel_loaded = (wheel_force > cfg["landing_wheel_force_threshold"]) & (
            wheel_clearance + env.wheel_radius < cfg["landing_wheel_center_height"]
        )
        body_loaded = (contact[:, env.body_links].norm(dim=-1) > cfg["body_contact_force_threshold"]).any(dim=1)
        self.wheel_contact_time.copy_(torch.where(active[:, None] & wheel_loaded, self.wheel_contact_time + dt, 0.0))
        self.body_contact_time.copy_(torch.where(active & body_loaded, self.body_contact_time + dt, 0.0))
        wheel_time = self.wheel_contact_time.amax(dim=1)
        finished = (
            (wheel_time >= cfg["wheel_contact_end_seconds"] - CONTACT_TIME_TOLERANCE)
            | (self.body_contact_time >= cfg["body_contact_end_seconds"] - CONTACT_TIME_TOLERANCE)
            | (self.air_time >= cfg["max_airborne_seconds"] - dt)
        )
        airborne = enter | (active & ~finished)
        self.air_time.copy_(torch.where(airborne, self.air_time + dt, 0.0))
        return airborne, wheel_time

    def _airborne_reward(self, airborne, wheel_time, offsets, quat, forward, lin_vel, wheel_vel, wheel_torque):
        env = self.env
        cfg = self.settings["airborne"]
        contact_transition = cfg["wheel_contact_transition_seconds"] - CONTACT_TIME_TOLERANCE
        before_contact = airborne & (wheel_time < contact_transition)
        low_torque = torch.exp(-(wheel_torque.abs().amax(dim=1) / cfg["wheel_torque_scale"]).square())
        # Wheel speeds use the mirror signs from the drive, so forward is positive on both sides.
        direction = forward.sign()
        speed_threshold = cfg["progress_speed_threshold"]
        moving = (forward.abs() > speed_threshold) & (lin_vel[:, 0] * direction > speed_threshold)
        signed_wheel_speed = wheel_vel * env.wheel_direction * direction[:, None]
        wheel_progress = (signed_wheel_speed / cfg["wheel_speed_scale"]).clamp(0.0, 1.0)
        wheel_progress = wheel_progress.amin(dim=1)
        contact_window = (wheel_time >= contact_transition) & (wheel_time < cfg["post_contact_end_seconds"])
        shortly_after_contact = airborne & contact_window

        w, x, y, z = quat.unbind(dim=1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        longitudinal = offsets[..., 0] * yaw.cos()[:, None] + offsets[..., 1] * yaw.sin()[:, None]
        wheels_below = (offsets[..., 2] < env.wheel_radius - self.minimum_body_clearance).all(dim=1)
        centered = torch.exp(-longitudinal.square() / cfg["centering_width"]).mean(dim=1)
        body_contact_limit = cfg["body_contact_centering_seconds"] - CONTACT_TIME_TOLERANCE
        centering_mask = before_contact & (self.body_contact_time < body_contact_limit) & wheels_below
        knees = env.robot.get_dofs_position(self.knee_dofs)
        knee_overrun = (self.knee_lower - knees).clamp_min(0.0) + (knees - self.knee_upper).clamp_min(0.0)
        return (
            cfg["low_torque_reward_weight"] * before_contact * low_torque
            + cfg["wheel_progress_reward_weight"] * (before_contact & moving) * wheel_progress
            - cfg["landing_wheel_progress_penalty_weight"] * (shortly_after_contact & moving) * (1.0 - wheel_progress)
            + cfg["centering_reward_weight"] * centering_mask * centered
            - cfg["knee_overrun_weight"] * airborne * knee_overrun.sum(dim=1)
        )

    def __call__(self, lin_vel, ang_vel, gravity, contact, fallen, actions):
        env = self.env
        tracking = self.settings["tracking"]
        regularization = self.settings["regularization"]
        joint_velocity = env.robot.get_dofs_velocity(env.action_dofs)
        torque = env.robot.get_dofs_control_force(env.action_dofs)
        wheels = torch.stack([wheel.get_anchor_pos() for wheel in env.wheels], dim=1)
        quat = env.robot.get_quat().contiguous()
        motion_cost = self._motion_cost(actions, joint_velocity, wheels, quat)
        if self.minimum_body_clearance is None:
            airborne = torch.zeros_like(fallen)
        else:
            airborne, wheel_time = self._airborne(wheels, contact)
        in_air = airborne.float()
        nonwheel_force = contact[:, self.nonwheel_links].norm(dim=-1)
        bad_contact = (nonwheel_force > regularization["nonwheel_contact_force_threshold"]).any(dim=1)

        forward, yaw = env.commands[:, :2].unbind(dim=1)
        stand = forward.abs() < tracking["forward_deadband"]
        forward_target = forward.masked_fill(stand, 0.0)
        yaw_target = yaw.masked_fill(yaw.abs() < tracking["yaw_deadband"], 0.0)
        gx, gy = gravity[:, :2].unbind(dim=1)
        measured_forward = lin_vel[:, 0] * gravity[:, 1:].norm(dim=1)
        forward_error = forward_target - measured_forward
        yaw_error = yaw_target - ang_vel[:, 2]
        height_error = env.body_clearance - (env.commands[:, 2] + env.body_height_offset)
        forward_clip = tracking["forward_error_clip"]
        yaw_clip = tracking["yaw_error_clip"]
        height_clip = tracking["height_error_clip"]
        tilt_scale = tracking["tilt_base_scale"] + tracking["tilt_low_speed_scale"] * torch.exp(
            -forward.square() / tracking["tilt_speed_width"]
        )
        pitch_cost = tracking["pitch_penalty_weight"] * (~airborne) * (tilt_scale * gx).square()
        if tracking["ground_pitch_error_weight"] or tracking["ground_pitch_rate_weight"]:
            pitch_rate = ang_vel[:, 1]
            zero = torch.zeros_like(gx)
            # Resolve small errors near level while retaining a gradient through large excursions.
            angle_error = smooth_l1_loss(gx / tracking["ground_pitch_error_scale"], zero, reduction="none")
            rate_error = smooth_l1_loss(pitch_rate / tracking["ground_pitch_rate_scale"], zero, reduction="none")
            pitch_cost += (~airborne) * (tracking["ground_pitch_error_weight"] * angle_error
                                         + tracking["ground_pitch_rate_weight"] * rate_error)
        roll_cost = tracking["roll_penalty_weight"] * (tilt_scale * gy).square()
        if tracking["flat_roll_error_weight"] or tracking["flat_roll_rate_weight"]:
            flat_grounded = (env.surface == 0) & ~airborne
            zero = torch.zeros_like(gy)
            angle_error = smooth_l1_loss(gy / tracking["flat_roll_error_scale"], zero, reduction="none")
            rate_error = smooth_l1_loss(ang_vel[:, 0] / tracking["flat_roll_rate_scale"], zero, reduction="none")
            roll_cost += flat_grounded * (tracking["flat_roll_error_weight"] * angle_error
                                          + tracking["flat_roll_rate_weight"] * rate_error)

        tracking_reward = (
            tracking["forward_reward_weight"]
            * torch.exp(-forward_error.clamp(-forward_clip, forward_clip).square() / tracking["forward_error_width"])
            - (tracking["forward_error_scale"] * forward_error).square()
            + tracking["yaw_reward_weight"]
            * torch.exp(-yaw_error.clamp(-yaw_clip, yaw_clip).square() / tracking["yaw_error_width"])
            - (tracking["yaw_error_scale"] * yaw_error).square()
            - tracking["standing_drift_weight"] * (stand * lin_vel[:, :2].abs().sum(dim=1))
            + tracking["pitch_reward_weight"] * torch.exp(-gx.square() / tracking["pitch_width"])
            + tracking["roll_reward_weight"] * torch.exp(-gy.square() / tracking["roll_width"])
            - (pitch_cost + roll_cost)
            + tracking["height_reward_weight"]
            * torch.exp(-height_error.clamp(-height_clip, height_clip).square() / tracking["height_error_width"])
            - (~airborne) * (tracking["height_error_scale"] * height_error).square()
        )
        wheel_vel = joint_velocity[:, 4:]
        wheel_torque = torque[:, 4:]
        contact_weight = regularization["nonwheel_contact_weight"] + regularization["airborne_nonwheel_contact_extra_weight"] * in_air
        fall_weight = regularization["fall_weight"] + regularization["airborne_fall_extra_weight"] * in_air
        regularization_cost = (
            motion_cost
            + regularization["torque_weight"] * torque.square().sum(dim=1)
            + regularization["wheel_power_weight"] * (wheel_torque * wheel_vel).clamp_min(0.0).sum(dim=1)
            + regularization["vertical_velocity_weight"] * lin_vel[:, 2].square()
            + regularization["body_angular_velocity_weight"] * ang_vel[:, :2].square().sum(dim=1)
            + contact_weight * bad_contact
            + fall_weight * fallen
        )
        reward = tracking_reward - regularization_cost
        if self.braking_cost is not None:
            eligible = (env.surface == 0) & ~airborne & ~fallen
            reward -= self.braking_cost(forward, env.goals[:, 0], measured_forward, gx, ang_vel[:, 1], eligible)
        if self.minimum_body_clearance is None:
            return reward
        offsets = wheels - env.robot.get_pos()[:, None]
        return reward + self._airborne_reward(
            airborne, wheel_time, offsets, quat, forward, lin_vel, wheel_vel, wheel_torque
        )
