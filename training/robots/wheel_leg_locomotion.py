"""V0.2 locomotion with continuous hip-to-wheel height commands."""

import genesis as gs
import torch
from genesis.utils.geom import inv_transform_by_quat
from tensordict import TensorDict
from torch.nn.functional import smooth_l1_loss

from configs.experiment import MAX_FORWARD_ACCELERATION, MAX_FORWARD_SPEED, YAW_REFERENCE
from envs.terrain import ScutV1Lanes, TerrainLanes
from robots.domain_randomization import randomize_dynamics
from robots.scut_rough_reward import ScutMotionCost, ScutRoughReward
from robots.wheel_leg import HOME_JOINTS, WheelLegFlatEnv
from robots.wheel_leg_stance import STANCE_SPAWN_GAP, solve_stances


class WheelLegLocomotionEnv(WheelLegFlatEnv):
    task_name = "locomotion"
    wheel_target_limit = 2.9224 / WheelLegFlatEnv.wheel_radius
    half_track = 0.21865
    forward_acceleration = MAX_FORWARD_ACCELERATION
    yaw_acceleration = 24.0
    braking_recovery_seconds = 0.90
    acceleration_settle_seconds = 0.35
    drop_approach_height = 0.23
    landing_seconds = 0.32

    def _robot_morph(self, model_path, viewer):
        self.pose_table = solve_stances(model_path, self.cfg["locomotion"]["height_range"], self.wheel_radius)
        return super()._robot_morph(model_path, viewer)

    def _add_ground(self, material, viewer):
        self.lanes = None
        self.ground = None
        self.terrain_contacts = []
        settings = self.cfg["locomotion"]
        if settings["terrain"] == "plane":
            super()._add_ground(material, viewer)
            return
        lanes = ScutV1Lanes if settings["terrain"] == "scut_v1" else TerrainLanes
        self.lanes = lanes(settings)
        surface = self.lanes.visual_surface() if viewer else None
        self.ground = self.scene.add_entity(self.lanes.morph(viewer), material=material, surface=surface)
        self.terrain_contacts.append(self.ground)
        if isinstance(self.lanes, ScutV1Lanes):
            deck_surface = gs.surfaces.Default(color=(0.73, 0.47, 0.27), roughness=0.95) if viewer else None
            for center, half_size in self.lanes.deck_boxes():
                box = gs.morphs.Box(pos=center, size=tuple(2 * extent for extent in half_size),
                                    fixed=True, visualization=viewer)
                self.terrain_contacts.append(self.scene.add_entity(box, material=material, surface=deck_surface))

    def terrain_description(self):
        if self.lanes is None:
            return super().terrain_description()
        surface, variant = self.terrain_selection[:, 0].tolist()
        return self.lanes.describe(surface, variant)

    def _set_drives(self):
        self.leg_torque_limit = self.cfg["locomotion"]["leg_torque_limit"]
        super()._set_drives()

    def _initialize_task(self):
        settings = self.cfg["locomotion"]
        randomization = self.cfg.get("domain_randomization", {})
        if randomization.get("enabled"):
            randomize_dynamics(self, randomization)
        self.height_low, self.height_high = settings["height_range"]
        self.height_nominal = settings["height_nominal"]
        self.height_max_probability = settings["height_max_probability"]
        self.height_lowering_schedule = settings["height_lowering_schedule"]
        self.height_sampling_floor = self.height_low
        self.height_curriculum_rollout = 0
        self._set_height_curriculum_iteration(0)
        self.pose_table = self.home_qpos.new_tensor(self.pose_table)
        self.body_height_tracking = settings["body_height_tracking"]
        self.body_height_offset = self.pose_table[0, 0].item() - self.height_low - STANCE_SPAWN_GAP
        stability = settings["stability"]
        common_stability = stability["enabled"] and settings["reward_profile"] == "v02"
        posture_weights = stability if common_stability else self.cfg["reward"]
        self.flat_pitch_weight = posture_weights["flat_pitch"]
        base = self.robot.get_joint("base_link").qs_idx_local
        self.base_xy = base[:2]
        self.base_quat = base[3:7]
        self.pose_indices = [base[2], *(self.robot.get_joint(name).qs_idx_local[0] for name in HOME_JOINTS)]
        self.leg_columns = [list(HOME_JOINTS).index(name) + 1 for name in self.leg_joints]
        self.hips = [self.robot.get_joint(f"hip_{side}") for side in "LR"]
        self.wheels = [self.robot.get_joint(f"wheel_{side}") for side in "LR"]
        self.goals = torch.zeros_like(self.commands)
        self.motion_scales = self.commands.new_tensor((1.0, 1.0, 5.0))
        ranges = self.cfg["forward_command"], self.cfg["yaw_command"], settings["height_range"]
        self.command_lower, self.command_upper = self.commands.new_tensor(ranges).T
        self.motion_step = self.commands.new_tensor((self.forward_acceleration, self.yaw_acceleration))
        self.motion_step = self.motion_step.expand(self.num_envs, -1).clone() * self.control_dt
        self.speed_levels = self.commands.new_tensor(settings["command_speed_levels"])
        self.acceleration_levels = self.commands.new_tensor(settings["command_acceleration_levels"])
        self.acceleration_weights = self.commands.new_tensor(settings["command_acceleration_weights"])
        if self.speed_levels.numel():
            self.episode_speed_limit = self.commands.new_full((self.num_envs,), MAX_FORWARD_SPEED)
            if self.lanes is not None:
                self.terrain_speed_limit = self.commands.new_tensor([
                    MAX_FORWARD_SPEED if surface == "flat" else
                    0.8 if surface in ("stairs_slow", "stairs_fast", "cliff") else 1.2
                    for surface in self.lanes.surfaces
                ])
        self.cliff_approach_speed = settings["cliff_approach_speed"]
        self.observation_noise = None
        self.height_step = settings["height_rate"] * self.control_dt
        self.height_acceleration = settings["height_acceleration"]
        self.command_time = self.commands.new_zeros(self.num_envs)
        self.command_period = settings["command_seconds"]
        self.speed_maneuvers = settings["speed_maneuvers"]
        self.braking_maneuvers = settings["braking_maneuvers"]
        self.launch_maneuvers = settings["launch_maneuvers"]
        self.height_maneuvers = settings["height_maneuvers"]
        self.launch_press_seconds = self.commands.new_tensor(settings["launch_press_seconds"])
        self.launch_stop_seconds = self.commands.new_tensor(settings["launch_stop_seconds"])
        self.maneuver_weights = self.commands.new_tensor(settings["speed_maneuver_weights"])
        self.launch_mode = 5 if self.braking_maneuvers else 4
        self.height_mode = self.launch_mode + int(self.launch_maneuvers)
        if self.speed_maneuvers:
            self.height_time = torch.zeros_like(self.command_time)
            self.maneuver = torch.zeros_like(self.episode_length_buf)
            self.maneuver_phase = torch.zeros_like(self.maneuver)
            self.maneuver_reverse = torch.zeros_like(self.commands[:, :2], dtype=torch.bool)
        self.height_rate = torch.zeros_like(self.command_time)
        self.measured_heights = self.commands.new_full((self.num_envs, 2), self.height_nominal)
        self.leg_alignment_weight = 0.0 if common_stability else self.cfg["reward"]["leg_alignment"]
        self.braking_pitch_weight = self.cfg["reward"]["braking_pitch"]
        self.braking_recovery_pitch_weight = self.cfg["reward"]["braking_recovery_pitch"]
        self.braking_yaw_weight = self.cfg["reward"]["braking_yaw"]
        self.braking_enabled = bool(
            self.braking_pitch_weight or self.braking_yaw_weight or self.braking_recovery_pitch_weight
        )
        self.acceleration_pitch_weight = self.cfg["reward"]["acceleration_pitch"]
        self.height_transition_weight = self.cfg["reward"]["height_transition"]
        if self.height_transition_weight:
            self.height_transition_time = torch.zeros_like(self.command_time)
            self.height_transition_supported = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
            self.leg_height_rate_gap = torch.zeros_like(self.command_time)
            self.last_leg_target = self.commands.new_zeros((self.num_envs, len(self.leg_dofs)))
            self.last_leg_target_step = torch.zeros_like(self.last_leg_target)
            self.leg_target_curvature = torch.zeros_like(self.last_leg_target)
        self.settled_motion_weight = posture_weights["settled_motion"]
        if self.settled_motion_weight:
            self.motion_settled = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
            self.action_target_limit = self.commands.new_tensor(
                [0.35 / self.leg_action_scale] * 4 + [self.wheel_target_limit / 10.0] * 2
            )
        self.steady_speed_weight = self.cfg["reward"]["steady_speed"]
        if self.braking_enabled:
            self.braking_time = torch.zeros_like(self.command_time)
        if self.acceleration_pitch_weight:
            self.acceleration_time = torch.zeros_like(self.command_time)
        if self.braking_enabled or self.acceleration_pitch_weight:
            self.forward_command_accel = torch.zeros_like(self.command_time)
        if self.steady_speed_weight:
            self.steady_forward_time = torch.zeros_like(self.command_time)
        self.posture = self.commands.new_zeros((self.num_envs, 3))
        self.posture_sums = torch.zeros_like(self.posture)
        self.roll_peaks = torch.zeros_like(self.command_time)
        self.pitch_deg = torch.zeros_like(self.command_time)
        self.pitch_sums = torch.zeros_like(self.pitch_deg)
        self.pitch_peaks = torch.zeros_like(self.pitch_deg)
        self.episode_steps = torch.zeros_like(self.episode_length_buf)
        self.height_errors = torch.zeros_like(self.command_time)
        self.body_clearance = torch.zeros_like(self.command_time)
        self.velocity_errors = torch.zeros_like(self.command_time)
        self.yaw_errors = torch.zeros_like(self.command_time)
        self.leg_saturated_steps = torch.zeros_like(self.episode_steps)
        self.spawn_y = torch.zeros_like(self.command_time)
        self.terrain_selection = torch.zeros((2, self.num_envs), dtype=self.episode_length_buf.dtype, device=self.device)
        self.surface, self.terrain_variant = self.terrain_selection.unbind()
        self.terrain_traversal = settings["terrain_traversal"]
        if self.terrain_traversal:
            self.traversal_reverse = torch.zeros_like(self.surface, dtype=torch.bool)
            self.traversal_distance = torch.zeros_like(self.command_time)
        self.auto_commands = True
        self.demo_surface = None
        self.demo_initial_height = self.height_nominal
        self.demo_reverse = False
        self.demo_variant = 0
        self.forward_drop = bool(settings["cliff_drop_heights"])
        if self.lanes is not None:
            self.lane_centers = self.commands.new_tensor(self.lanes.centers)
            self.surface_weights = self.commands.new_tensor(self.lanes.surface_weights)
        if isinstance(self.lanes, ScutV1Lanes):
            self.variant_weights = self.commands.new_tensor(self.lanes.variant_weights)
            self.lane_spawn_x = self.commands.new_tensor(self.lanes.spawn_x)
            self.lane_spawn_z = self.commands.new_tensor(self.lanes.spawn_z)
            self.spawn_x = torch.zeros_like(self.spawn_y)
        if self.forward_drop:
            self.cliff_surface = self.lanes.surfaces.index("cliff")
            self.deck_height = torch.zeros_like(self.command_time)
            self.drop_touched = torch.zeros_like(self.surface, dtype=torch.bool)
            self.landing_elapsed = self.command_time.new_full((self.num_envs,), self.landing_seconds)
            self.body_weight = 9.81 * self.robot.get_mass()
        self.scut_reward = ScutRoughReward(self) if settings["reward_profile"] == "scut_rough_v1" else None
        self.scut_motion_cost = ScutMotionCost(self) if common_stability else None

    def _poses(self, height):
        span = max(self.height_high - self.height_low, 1e-8)
        position = ((height - self.height_low) / span).clamp(0.0, 1.0) * (len(self.pose_table) - 1)
        lower = position.long().clamp_max(len(self.pose_table) - 2)
        return self.pose_table[lower].lerp(self.pose_table[lower + 1], (position - lower)[:, None])

    def _sample_heights(self, count):
        sample = torch.rand((2, count), device=self.device)
        floor = self.height_sampling_floor
        height = floor + sample[0] * (self.height_high - floor)
        height = torch.where(sample[1] >= 1.0 - self.height_max_probability, self.height_high, height)
        height = torch.where(sample[1] < 0.60, max(floor, self.height_nominal), height)
        return torch.where(sample[1] < 0.25, floor, height)

    def start_height_curriculum(self, iteration, rollout_steps):
        if self.height_lowering_schedule:
            self.height_curriculum_rollout = rollout_steps
            self.height_curriculum_step = iteration * rollout_steps
            self._set_height_curriculum_iteration(iteration)
            if iteration:
                self.reset()

    def _set_height_curriculum_iteration(self, iteration):
        if not self.height_lowering_schedule:
            return
        for (start, high), (end, low) in zip(self.height_lowering_schedule, self.height_lowering_schedule[1:]):
            if iteration < end:
                self.height_sampling_floor = high + (low - high) * (iteration - start) / (end - start)
                return
        self.height_sampling_floor = self.height_lowering_schedule[-1][1]

    def _sample_motion_goals(self, ids):
        if ids.numel() == 0:
            return
        sample = torch.rand((ids.numel(), 4), device=self.device)
        motion = self.command_lower[:2] + sample[:, :2] * (self.command_upper[:2] - self.command_lower[:2])
        motion[:, 0] *= sample[:, 2] >= 0.25
        motion[:, 1] *= (sample[:, 2] >= 0.15) & (sample[:, 2] <= 0.75)
        low, high = self.command_period
        duration = low + sample[:, 3] * (high - low)
        if self.speed_levels.numel():
            forward_cap = self.episode_speed_limit[ids]
            motion[:, 0] = motion[:, 0].clamp(-forward_cap, forward_cap)
        if self.speed_maneuvers and not self.terrain_traversal:
            motion, duration = self._maneuver_goals(ids, motion, duration, sample[:, 3])
        if self.terrain_traversal:
            # Keep moving along the obstacle lane across successive command resamples.
            surface = self.surface[ids]
            traverse = surface != 0
            if self.speed_maneuvers:
                flat = ~traverse
                motion[flat], duration[flat] = self._maneuver_goals(
                    ids[flat], motion[flat], duration[flat], sample[flat, 3]
                )
            limit = torch.where(self.traversal_reverse[ids], self.command_lower[0], self.command_upper[0])
            if self.forward_drop:
                cliff = surface == self.cliff_surface
                limit = torch.where(cliff, self.command_upper[0], limit)
            if self.speed_levels.numel():
                terrain_cap = torch.minimum(forward_cap, self.terrain_speed_limit[surface])
                limit = limit.clamp(-terrain_cap, terrain_cap)
            cruise = limit * (0.65 + 0.35 * sample[:, 0])
            if self.cliff_approach_speed:
                airborne = cliff & (~self.drop_touched[ids] | (self.landing_elapsed[ids] < self.landing_seconds))
                approach = self.cliff_approach_speed * (0.80 + 0.20 * sample[:, 0])
                cruise = torch.where(airborne, torch.minimum(cruise, approach), cruise)
            motion[:, 0] = torch.where(traverse, cruise, motion[:, 0])
            motion[:, 1] = torch.where(traverse, 0.0, motion[:, 1])
        self.goals[ids, :2] = motion
        self.command_time[ids] = duration

    def _maneuver_goals(self, ids, motion, duration, hold_sample):
        # Each episode repeats one pattern. Phase zero starts in a randomly chosen direction.
        steady, straight, spin, slalom, corner_brake = range(5)
        mode = self.maneuver[ids]
        phase = self.maneuver_phase[ids]
        reverse, clockwise = self.maneuver_reverse[ids].unbind(dim=1)
        forward_min, yaw_min = self.command_lower[:2].unbind()
        forward_max, yaw_max = self.command_upper[:2].unbind()

        # Run, stop, run, reverse directly, stop, run; the cycle boundary reverses directly again.
        forward = torch.where(reverse ^ (phase >= 3), forward_min, forward_max)
        forward = forward * ((mode == straight) & (phase.remainder(3) != 1))
        turn_speed = 0.65 * torch.where(reverse, forward_min, forward_max)
        forward = torch.where(mode == slalom, turn_speed, forward)
        yaw = torch.where(clockwise ^ (phase.remainder(2) == 1), yaw_min, yaw_max)
        yaw = yaw * ((mode == spin) | (mode == slalom))
        hold_seconds = 0.25 + 0.30 * hold_sample
        if self.speed_levels.numel():
            stop = (mode == straight) & (phase.remainder(3) == 1)
            hold_seconds = torch.where(stop, 0.75 + 0.40 * hold_sample, hold_seconds)
        if self.steady_speed_weight:
            sustained = (mode == straight) & (phase.remainder(3) == 0) & (hold_sample >= 0.5)
            hold_seconds = torch.where(sustained, 1.5 + hold_sample, hold_seconds)
        next_phase = (phase + 1).remainder(6)
        if self.braking_maneuvers:
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
        if self.launch_maneuvers:
            launch = mode == self.launch_mode
            segment = phase.remainder(4)
            moving = segment.remainder(2) == 0
            switch_direction = segment >= 2
            launch_hold = 0.90 + 0.30 * hold_sample
            if self.launch_stop_seconds.numel():
                # Split the launch cases between repeated taps in one direction and reversals.
                switch_direction &= clockwise
                slots = (hold_sample * len(self.launch_stop_seconds)).long().clamp_max(len(self.launch_stop_seconds) - 1)
                launch_hold = torch.where(moving, launch_hold, self.launch_stop_seconds[slots])
            launch_speed = torch.where(reverse ^ switch_direction, forward_min, forward_max)
            forward = torch.where(launch, launch_speed * moving, forward)
            hold_seconds = torch.where(launch, launch_hold, hold_seconds)
            next_phase = torch.where(launch, (phase + 1).remainder(4), next_phase)
        if self.height_maneuvers:
            height_only = mode == self.height_mode
            cruise = torch.where(reverse, -0.5, 0.5) * clockwise
            forward = torch.where(height_only, cruise, forward)
            yaw = yaw.masked_fill(height_only, 0.0)

        if self.speed_levels.numel():
            forward_cap = self.episode_speed_limit[ids]
            forward = forward.clamp(-forward_cap, forward_cap)
        target = torch.stack((forward, yaw), dim=1)
        # Reserve time for the slew-limited transition and the dwell.
        ramp_seconds = ((target - self.commands[ids, :2]).abs() / self.motion_step[ids]).amax(dim=1) * self.control_dt
        maneuver_seconds = ramp_seconds + hold_seconds
        if self.launch_maneuvers and self.launch_press_seconds.numel():
            slots = (hold_sample * len(self.launch_press_seconds)).long().clamp_max(len(self.launch_press_seconds) - 1)
            maneuver_seconds = torch.where(launch & moving, self.launch_press_seconds[slots], maneuver_seconds)
        self.maneuver_phase[ids] = next_phase
        return torch.where((mode == steady)[:, None], motion, target), torch.where(
            mode == steady, duration, maneuver_seconds
        )

    def _sample_height_goals(self, ids):
        if ids.numel() == 0:
            return
        height = self._sample_heights(ids.numel())
        if self.height_maneuvers:
            flat = self.surface[ids] == 0
            height_only = (self.maneuver[ids] == self.height_mode) & flat
            previous = self.goals[ids, 2]
            midpoint = (self.height_high + self.height_low) / 2
            opposite = torch.where(previous < midpoint, self.height_high, self.height_low)
            height = torch.where(height_only, opposite, height)
            # Start with a fixed height so the cruise cases can reach their speed first.
            hold = flat & (~height_only | (self.episode_steps[ids] == 0))
            height = torch.where(hold, previous, height)
        elif self.launch_maneuvers:
            launch = (self.maneuver[ids] == self.launch_mode) & (self.surface[ids] == 0)
            height = torch.where(launch, self.goals[ids, 2], height)
        self.goals[ids, 2] = height
        if self.forward_drop:
            approaching = (self.surface[ids] == self.cliff_surface) & ~self.drop_touched[ids]
            self.goals[ids, 2] = torch.where(approaching, self.goals[ids, 2].clamp_min(self.drop_approach_height),
                                               self.goals[ids, 2])
        if self.speed_maneuvers:
            low, high = self.command_period
            self.height_time[ids] = low + torch.rand(ids.numel(), device=self.device) * (high - low)

    def _reset_envs(self, ids):
        height = (self._sample_heights(ids.numel()) if self.auto_commands else
                  self.commands.new_full((ids.numel(),), self.demo_initial_height))
        qpos = self.home_qpos[ids].clone()
        ground_height = 0.0
        scut_course = isinstance(self.lanes, ScutV1Lanes)
        if self.lanes is not None:
            if self.demo_surface is None:
                self.surface[ids] = torch.multinomial(self.surface_weights, ids.numel(), replacement=True)
                variant = (torch.multinomial(self.variant_weights, ids.numel(), replacement=True) if scut_course else
                           torch.randint(self.lanes.variants, (ids.numel(),), device=self.device))
                jitter = torch.rand(ids.numel(), device=self.device) * 0.12 - 0.06
            else:
                self.surface[ids] = self.demo_surface
                variant = self.demo_variant
                jitter = 0.0
            self.terrain_variant[ids] = variant
            lane = self.surface[ids] * self.lanes.variants + variant
            self.spawn_y[ids] = self.lane_centers[lane]
            qpos[:, self.base_xy[0]] = 0.0
            qpos[:, self.base_xy[1]] = self.spawn_y[ids] + jitter
            if scut_course:
                reverse = (torch.rand(ids.numel(), device=self.device) < 0.5 if self.auto_commands else
                           torch.full((ids.numel(),), self.demo_reverse, device=self.device, dtype=torch.bool))
                self.traversal_reverse[ids] = reverse
                ground_height = self.lane_spawn_z[lane]
                positive_x = reverse
                if self.forward_drop:
                    cliff = self.surface[ids] == self.cliff_surface
                    positive_x = reverse ^ cliff
                    w, x, y, z = qpos[:, self.base_quat].unbind(dim=1)
                    turned = torch.stack((-z, -y, x, w), dim=1)
                    qpos[:, self.base_quat] = torch.where((cliff & reverse)[:, None], turned, qpos[:, self.base_quat])
                    if self.auto_commands:
                        height = torch.where(cliff, height.clamp_min(self.drop_approach_height), height)
                    self.deck_height[ids] = torch.where(cliff, ground_height, 0.0)
                    self.drop_touched[ids] = False
                    self.landing_elapsed[ids] = self.landing_seconds
                self.spawn_x[ids] = torch.where(positive_x, self.lane_spawn_x[lane], -self.lane_spawn_x[lane])
                qpos[:, self.base_xy[0]] = self.spawn_x[ids]
        pose = self._poses(height)
        qpos[:, self.pose_indices] = pose
        qpos[:, self.pose_indices[0]] += ground_height
        self.robot.set_qpos(qpos, envs_idx=ids, zero_velocity=True)
        self.episode_length_buf[ids] = 0
        self.episode_steps[ids] = 0
        self.height_errors[ids] = 0.0
        self.velocity_errors[ids] = 0.0
        self.yaw_errors[ids] = 0.0
        self.posture_sums[ids] = 0.0
        self.roll_peaks[ids] = 0.0
        self.pitch_sums[ids] = 0.0
        self.pitch_peaks[ids] = 0.0
        if self.terrain_traversal:
            self.traversal_distance[ids] = 0.0
            if self.auto_commands and not scut_course:
                self.traversal_reverse[ids] = torch.rand(ids.numel(), device=self.device) < 0.5
        if self.braking_enabled:
            self.braking_time[ids] = 0.0
        if self.acceleration_pitch_weight:
            self.acceleration_time[ids] = 0.0
        if self.braking_enabled or self.acceleration_pitch_weight:
            self.forward_command_accel[ids] = 0.0
        if self.height_transition_weight:
            self.height_transition_time[ids] = 0.0
            self.height_transition_supported[ids] = False
            self.leg_height_rate_gap[ids] = 0.0
            self.last_leg_target[ids] = pose[:, self.leg_columns] + self.leg_bias
            self.last_leg_target_step[ids] = 0.0
            self.leg_target_curvature[ids] = 0.0
        if self.steady_speed_weight:
            self.steady_forward_time[ids] = 0.0
        if self.settled_motion_weight:
            self.motion_settled[ids] = False
        self.leg_saturated_steps[ids] = 0
        self.last_actions[ids] = 0.0
        if self.scut_reward is not None:
            self.scut_reward.reset(ids)
        if self.scut_motion_cost is not None:
            self.scut_motion_cost.reset(ids)
        self.height_rate[ids] = 0.0
        self.commands[ids] = 0.0
        self.commands[ids, 2] = height
        self.measured_heights[ids] = height[:, None]
        self.goals[ids] = self.commands[ids]
        if self.speed_levels.numel() and self.auto_commands:
            levels = torch.randint(self.speed_levels.numel(), (ids.numel(),), device=self.device)
            self.episode_speed_limit[ids] = self.speed_levels[levels]
        if self.acceleration_levels.numel():
            forward_accel = self.forward_acceleration
            if self.auto_commands:
                if self.acceleration_weights.numel():
                    levels = torch.multinomial(self.acceleration_weights, ids.numel(), replacement=True)
                else:
                    levels = torch.randint(self.acceleration_levels.numel(), (ids.numel(),), device=self.device)
                forward_accel = self.acceleration_levels[levels]
            self.motion_step[ids, 0] = forward_accel * self.control_dt
        if self.auto_commands:
            if self.speed_maneuvers:
                count = 5 if self.braking_maneuvers else 4
                if self.maneuver_weights.numel():
                    self.maneuver[ids] = torch.multinomial(self.maneuver_weights, ids.numel(), replacement=True)
                else:
                    self.maneuver[ids] = torch.randint(count, (ids.numel(),), device=self.device)
                    if self.launch_maneuvers:
                        launch = (torch.rand(ids.numel(), device=self.device) < 0.30) & (self.surface[ids] == 0)
                        self.maneuver[ids] = self.maneuver[ids].masked_fill(launch, self.launch_mode)
                    if self.height_maneuvers:
                        height_only = (torch.rand(ids.numel(), device=self.device) < 0.25) & (self.surface[ids] == 0)
                        self.maneuver[ids] = self.maneuver[ids].masked_fill(height_only, self.height_mode)
                self.maneuver_phase[ids] = 0
                self.maneuver_reverse[ids] = torch.rand((ids.numel(), 2), device=self.device) < 0.5
            self._sample_motion_goals(ids)
            self._sample_height_goals(ids)

    def begin_demo(self, height, surface, reverse=False):
        self.auto_commands = False
        self.demo_initial_height = height
        self.demo_surface = self.lanes.surfaces.index(surface) if self.lanes is not None else None
        self.demo_reverse = reverse
        self.demo_variant = 0
        self.reset()

    def set_commands(self, forward, yaw, height):
        self.goals[:] = self.commands.new_tensor((forward, yaw, height))
        lin_vel, ang_vel, _, _ = super()._base_state()
        self._advance_commands(lin_vel, ang_vel)

    def _advance_commands(self, lin_vel, ang_vel):
        targets = self.goals.clamp(min=self.command_lower, max=self.command_upper)
        vx, yaw = targets[:, :2].unbind(1)
        demand = torch.sqrt((vx / MAX_FORWARD_SPEED).square() + (yaw / YAW_REFERENCE).square())
        # The chassis reduces combined demand as turning unloads the inside wheel.
        lateral_accel = (lin_vel[:, 0] * ang_vel[:, 2]).abs()
        current_height = self.measured_heights.mean(dim=1).clamp(0.10, 0.335)
        radius = (1.0 - lateral_accel * current_height / (9.81 * self.half_track)).clamp(0.05, 1.0)
        outer_wheel = vx.abs() + self.half_track * yaw.abs()
        scale = torch.minimum(radius / demand.clamp_min(1e-6),
                              self.wheel_target_limit * self.wheel_radius / outer_wheel.clamp_min(1e-6))
        motion_delta = targets[:, :2] * scale.clamp_max(1.0)[:, None] - self.commands[:, :2]
        # Move along one segment so the wheel-speed bound holds during reversals too.
        steps_needed = (motion_delta.abs() / self.motion_step).amax(dim=1).clamp_min(1.0)
        motion_step = motion_delta / steps_needed[:, None]
        if self.braking_enabled or self.acceleration_pitch_weight:
            self.forward_command_accel.copy_(motion_step[:, 0] / self.control_dt)
        if self.braking_enabled:
            forward = self.commands[:, 0]
            # Distinguish a brake request from feedback changes in the combined motion limiter.
            requested_drop = forward.sign() * (forward - vx)
            deceleration = -forward.sign() * self.forward_command_accel
            minimum_speed = 0.10 if self.braking_recovery_pitch_weight else 0.20
            slowing = (forward.abs() > minimum_speed) & (requested_drop > 0.02) & (deceleration > 0.75)
            self.braking_time.copy_(torch.where(slowing, self.braking_recovery_seconds,
                                                (self.braking_time - self.control_dt).clamp_min(0.0)))
        if self.acceleration_pitch_weight:
            forward = self.commands[:, 0]
            next_forward = forward + motion_step[:, 0]
            speeding_up = ((next_forward.abs() > forward.abs()) & (vx.abs() > forward.abs() + 0.02)
                           & (next_forward * vx > 0.0) & (self.forward_command_accel.abs() > 0.65))
            speeding_up &= self.surface == 0
            self.acceleration_time.copy_(torch.where(speeding_up, self.acceleration_settle_seconds,
                                                     (self.acceleration_time - self.control_dt).clamp_min(0.0)))
        self.commands[:, :2] += motion_step
        if self.steady_speed_weight:
            steady = ((motion_step[:, 0].abs() <= 0.50 * self.control_dt) & (self.commands[:, 1].abs() <= 0.20)
                      & (self.commands[:, 0].abs() >= 0.40))
            self.steady_forward_time.copy_(torch.where(
                steady, (self.steady_forward_time + self.control_dt).clamp_max(0.50), 0.0
            ))
        height_error = targets[:, 2] - self.commands[:, 2]
        height_delta = height_error.clamp(-self.height_step, self.height_step)
        if self.height_acceleration:
            dt = self.control_dt
            max_rate_change = self.height_acceleration * dt
            # Leave one control tick of travel in addition to the continuous braking distance.
            stopping_rate = (max_rate_change ** 2 + 2.0 * self.height_acceleration * height_error.abs()).sqrt()
            desired_rate = height_error.sign() * (stopping_rate - max_rate_change).clamp_max(self.height_step / dt)
            final_rate = height_error / dt
            can_finish = (final_rate.abs() <= max_rate_change) & (
                (final_rate - self.height_rate).abs() <= max_rate_change
            )
            desired_rate = torch.where(can_finish, final_rate, desired_rate)
            height_rate = self.height_rate + (desired_rate - self.height_rate).clamp(-max_rate_change, max_rate_change)
            next_height = (self.commands[:, 2] + height_rate * dt).clamp(self.height_low, self.height_high)
            height_delta = next_height - self.commands[:, 2]
        self.commands[:, 2] += height_delta
        self.height_rate.copy_(height_delta / self.control_dt)
        if self.height_transition_weight:
            self.height_transition_time.copy_(torch.where(
                height_delta.abs() > 1e-5, 0.35, (self.height_transition_time - self.control_dt).clamp_min(0.0)
            ))
        if self.settled_motion_weight:
            self.motion_settled.copy_(
                (self.surface == 0) & (motion_delta[:, 0].abs() < 0.03) & (motion_delta[:, 1].abs() < 0.08)
                & (self.commands[:, 1].abs() <= 0.25) & (height_delta.abs() <= 0.001)
            )

    def _leg_projections(self, gravity):
        """Return leg heights and signed wheel offsets ahead of each hip."""
        quat = self.robot.get_quat().contiguous()
        # Match L0*cos(theta) in the firmware sagittal plane. Ignore the lateral anchor offset.
        up = -gravity
        up_x = up[:, 0:1]
        up_z = up[:, 1:].norm(dim=1, keepdim=True)
        vectors = []
        for hip, wheel in zip(self.hips, self.wheels, strict=True):
            offset = (hip.get_anchor_pos() - wheel.get_anchor_pos()).contiguous()
            vectors.append(inv_transform_by_quat(offset, quat))
        offsets = torch.stack(vectors, dim=1)
        x, _, z = offsets.unbind(dim=2)
        return x * up_x + z * up_z, z * up_x - x * up_z

    def leg_heights(self, gravity):
        heights, _ = self._leg_projections(gravity)
        return heights

    def enable_observation_noise(self):
        settings = self.cfg["observation_noise"]
        stds = [settings["gyro"]] * 3 + [settings["gravity"]] * 3 + [settings["joint_position"]] * 4
        stds += [0.1 * settings["joint_velocity"]] * 6
        if any(stds):
            # Keep some clean observations and vary the sensor noise scale across environments.
            level = self.commands.new_empty((self.num_envs, 1)).uniform_(0.5, 1.5)
            level.masked_fill_(torch.rand_like(level) < 0.25, 0.0)
            self.observation_noise = self.commands.new_tensor(stds) * level

    def get_observations(self):
        lin_vel, ang_vel, gravity, _ = super()._base_state()
        heights = self.leg_heights(gravity)
        nominal = self._poses(self.commands[:, 2])[:, self.leg_columns]
        joint_pos = (self.robot.get_dofs_position(self.leg_dofs) - nominal) * self.leg_direction
        joint_vel = self.robot.get_dofs_velocity(self.action_dofs) * self.action_direction
        policy = torch.cat((self.commands * self.motion_scales, self.height_rate[:, None], ang_vel, gravity,
                            joint_pos, joint_vel * 0.1, self.last_actions, heights * 5.0), dim=1)
        critic = torch.cat((policy, lin_vel), dim=1)
        if self.observation_noise is not None:
            # The critic retains the original state; commands, history and inferred leg height stay intact.
            sensors = policy[:, 4:20]
            sensors.add_(torch.randn_like(sensors) * self.observation_noise)
        return TensorDict({"policy": policy, "critic": critic}, batch_size=[self.num_envs])

    def _drive(self, actions):
        nominal = self._poses(self.commands[:, 2])[:, self.leg_columns]
        offsets = (actions[:, :4] * self.leg_action_scale).clamp(-0.35, 0.35)
        targets = nominal + offsets * self.leg_direction + self.leg_bias
        if self.height_transition_weight:
            target_step = targets - self.last_leg_target
            self.leg_target_curvature.copy_(target_step - self.last_leg_target_step)
            self.last_leg_target_step.copy_(target_step)
            self.last_leg_target.copy_(targets)
        self.robot.control_dofs_position(targets, self.leg_dofs)
        self.robot.control_dofs_position(self.gimbal_target, self.gimbal_dofs)

    def ground_height_at(self, pos):
        if self.ground is None:
            return torch.zeros_like(pos[..., 2])
        points = pos.reshape(self.num_envs, -1, 3)
        surface_z = self.ground.get_terrain_height(points[..., :2].contiguous())
        if self.forward_drop:
            on_deck = ((points[..., 0].abs() <= self.lanes.deck_half_length)
                       & ((points[..., 1] - self.spawn_y[:, None]).abs() <= self.lanes.deck_half_width))
            surface_z = surface_z + self.deck_height[:, None] * on_deck
        return surface_z.reshape(pos.shape[:-1])

    def _height_and_termination(self, height, gravity, contact):
        heights, fore_aft = self._leg_projections(gravity)
        if self.height_transition_weight:
            height_change = heights - self.measured_heights
            self.leg_height_rate_gap.copy_((height_change[:, 0] - height_change[:, 1]) / self.control_dt)
            loaded = (contact[:, self.wheel_links, 2] > 5.0).all(dim=1)
            self.height_transition_supported.copy_(
                loaded & (self.surface == 0) & (self.commands[:, 1].abs() <= 0.25)
            )
        self.measured_heights.copy_(heights)
        self.posture[:, 0] = torch.rad2deg(torch.atan2(gravity[:, 1], -gravity[:, 2])).abs()
        self.pitch_deg.copy_(torch.rad2deg(torch.atan2(gravity[:, 0], gravity[:, 1:].norm(dim=1))))
        self.posture[:, 1] = 100.0 * (fore_aft[:, 0] - fore_aft[:, 1]).abs()
        self.posture[:, 2] = 100.0 * (heights[:, 0] - heights[:, 1]).abs()
        body_contact = contact[:, self.body_links].norm(dim=-1).amax(dim=1) > 10.0
        clearance = height
        out_of_bounds = False
        if self.ground is not None:
            pos = self.robot.get_pos()
            if self.terrain_traversal:
                x = pos[:, 0]
                if isinstance(self.lanes, ScutV1Lanes):
                    x = x - self.spawn_x
                torch.maximum(self.traversal_distance, x.abs(), out=self.traversal_distance)
            surface_z = self.ground_height_at(pos)
            clearance = height - surface_z.nan_to_num()
            out_of_bounds = ((pos[:, 0].abs() >= self.lanes.travel_limit)
                             | ((pos[:, 1] - self.spawn_y).abs() >= self.lanes.lateral_limit)
                             | ~torch.isfinite(surface_z))
        measured_clearance = clearance if self.scut_reward is None else self.scut_reward.measure_clearance(clearance)
        self.body_clearance.copy_(measured_clearance)
        if self.scut_reward is not None:
            error = self.body_clearance - (self.commands[:, 2] + self.body_height_offset)
            height_reward = 0.0
        elif self.body_height_tracking:
            # Cover larger errors on obstacles while rewarding precise tracking on flat ground.
            error = clearance - (self.commands[:, 2] + self.body_height_offset)
            flat = self.surface == 0
            height_reward = 1.5 * torch.exp(-(error / 0.030).square())
            height_cost = smooth_l1_loss(error / 0.018, torch.zeros_like(error), reduction="none")
            vertical_error = (self.robot.get_vel()[:, 2] - self.height_rate) / 0.12
            vertical_cost = smooth_l1_loss(vertical_error, torch.zeros_like(vertical_error), reduction="none")
            flat_reward = 1.5 * torch.exp(-(error / 0.010).square()) - 0.85 * height_cost - 0.35 * vertical_cost
            height_reward += flat_reward.masked_fill(~flat, 0.0)
        else:
            error = self.measured_heights.mean(dim=1) - self.commands[:, 2]
            height_reward = 1.5 * torch.exp(-0.5 * (error / 0.012).square()) - 0.5 * (error / 0.035).square()
        self.height_errors += error.abs()
        if self.forward_drop:
            if self.scut_reward is None:
                height_reward = self._landing_reward(pos, contact, error, height_reward)
            else:
                self._update_drop_landing(pos, contact)
        fallen = body_contact | (clearance < 0.045) | (gravity[:, 2] > -0.65)
        return height_reward, fallen, out_of_bounds

    def _update_drop_landing(self, pos, contact):
        forces = contact[:, self.wheel_links, 2].clamp_min(0.0)
        left, right = forces.unbind(dim=1)
        crossed_edge = pos[:, 0].abs() > self.lanes.deck_half_length + 0.04
        touchdown = (self.deck_height > 0.0) & crossed_edge & ~self.drop_touched & ((left + right) > 5.0)
        self.drop_touched |= touchdown
        self.landing_elapsed.add_(self.control_dt).clamp_max_(self.landing_seconds)
        self.landing_elapsed.masked_fill_(touchdown, 0.0)
        return left, right

    def _landing_reward(self, pos, contact, height_error, height_reward):
        left, right = self._update_drop_landing(pos, contact)
        t = self.landing_elapsed
        compression = (t / 0.06).clamp_max(1.0) * (1.0 - t / self.landing_seconds)
        stroke = (self.commands[:, 2] - self.height_low).clamp(0.0, 0.06)
        error = height_error + stroke * compression
        overload = (torch.maximum(left, right) / self.body_weight - 2.0).clamp_min(0.0)
        imbalance = ((left - right).abs() / self.body_weight - 0.35).clamp_min(0.0)
        both_loaded = (left > 5.0) & (right > 5.0)
        cushioned = (1.5 * torch.exp(-0.5 * (error / 0.020).square()) - 0.5 * (error / 0.035).square()
                     + 0.5 * both_loaded - overload - 0.4 * imbalance)
        if self.body_height_tracking:
            # Once past the deck, suspend ground-height tracking until the wheels touch down.
            falling_from_deck = ((self.deck_height > 0.0) & (pos[:, 0].abs() > self.lanes.deck_half_length)
                                 & ~self.drop_touched)
            height_reward = height_reward.masked_fill(falling_from_deck, 0.0)
        return torch.where(t < self.landing_seconds, cushioned, height_reward)

    def _reward(self, lin_vel, ang_vel, gravity, contact, height_reward, fallen, actions):
        if self.scut_reward is not None:
            return self.scut_reward(lin_vel, ang_vel, gravity, contact, fallen, actions)
        return super()._reward(lin_vel, ang_vel, gravity, contact, height_reward, fallen, actions)

    def _action_cost(self, actions):
        if self.scut_motion_cost is None:
            return super()._action_cost(actions)
        wheels = torch.stack([wheel.get_anchor_pos() for wheel in self.wheels], dim=1)
        quat = self.robot.get_quat().contiguous()
        joint_velocity = self.robot.get_dofs_velocity(self.action_dofs)
        return self.scut_motion_cost(actions, joint_velocity, wheels, quat)

    def _task_cost(self, lin_vel, ang_vel, actions):
        cost = 0.0
        if self.leg_alignment_weight:
            # Allow 2 cm of relative fore-aft travel and grow linearly for large deviations.
            excess = (self.posture[:, 1] - 2.0).clamp_min(0.0) / 5.0
            cost = self.leg_alignment_weight * smooth_l1_loss(excess, torch.zeros_like(excess), reduction="none")
        if self.flat_pitch_weight:
            pitch_error = (self.pitch_deg.abs() - 1.0).clamp_min(0.0) / 5.0
            pitch_rate = ang_vel[:, 1] / 0.45
            pitch_cost = smooth_l1_loss(pitch_error, torch.zeros_like(pitch_error), reduction="none")
            rate_cost = smooth_l1_loss(pitch_rate, torch.zeros_like(pitch_rate), reduction="none")
            flat_cost = self.flat_pitch_weight * (pitch_cost + 0.35 * rate_cost)
            cost = cost + flat_cost.masked_fill(self.surface != 0, 0.0)
        if self.braking_enabled:
            recovery = (self.braking_time / 0.30).clamp_max(1.0)
        if self.braking_pitch_weight:
            # Permit a larger transient lean during the ramp, then tighten as the reference settles.
            effort = (self.forward_command_accel.abs() / self.forward_acceleration).clamp_max(1.0)
            angle_excess = (self.pitch_deg.abs() - (6.0 + 6.0 * effort)).clamp_min(0.0) / 8.0
            rate_excess = (ang_vel[:, 1].abs() - (0.65 + 0.50 * effort)).clamp_min(0.0) / 1.5
            angle_cost = smooth_l1_loss(angle_excess, torch.zeros_like(angle_excess), reduction="none")
            rate_cost = smooth_l1_loss(rate_excess, torch.zeros_like(rate_excess), reduction="none")
            cost = cost + self.braking_pitch_weight * recovery * (angle_cost + 0.5 * rate_cost)
        if self.braking_recovery_pitch_weight:
            stopping = (self.goals[:, 0].abs() <= 0.05) & (self.commands[:, 0].abs() <= 0.05)
            stopping &= self.surface == 0
            pitch_error = (self.pitch_deg.abs() - 1.0).clamp_min(0.0) / 3.0
            pitch_rate = ang_vel[:, 1] / 0.40
            angle_cost = smooth_l1_loss(pitch_error, torch.zeros_like(pitch_error), reduction="none")
            rate_cost = smooth_l1_loss(pitch_rate, torch.zeros_like(pitch_rate), reduction="none")
            cost = cost + self.braking_recovery_pitch_weight * recovery * stopping * (angle_cost + 0.35 * rate_cost)
        if self.braking_yaw_weight:
            straight = ((self.surface == 0) & (self.goals[:, 1].abs() <= 0.05)
                        & (self.commands[:, 1].abs() <= 0.05))
            yaw_error = (ang_vel[:, 2] - self.commands[:, 1]) / 0.35
            yaw_cost = smooth_l1_loss(yaw_error, torch.zeros_like(yaw_error), reduction="none")
            cost = cost + self.braking_yaw_weight * recovery * straight * yaw_cost
        if self.acceleration_pitch_weight:
            effort = (self.forward_command_accel.abs() / self.forward_acceleration).clamp_max(1.0)
            angle_excess = (self.pitch_deg.abs() - (2.5 + 2.5 * effort)).clamp_min(0.0) / 5.0
            angle_cost = smooth_l1_loss(angle_excess, torch.zeros_like(angle_excess), reduction="none")
            rate_cost = 0.18 * (ang_vel[:, 1] / 0.70).square().clamp_max(4.0)
            speed_error = (lin_vel[:, 0] - self.commands[:, 0]) / 0.25
            tracking_cost = 0.25 * smooth_l1_loss(speed_error, torch.zeros_like(speed_error), reduction="none")
            accelerating = (self.acceleration_time / 0.12).clamp_max(1.0)
            cost = cost + self.acceleration_pitch_weight * accelerating * (angle_cost + rate_cost + tracking_cost)
        if self.settled_motion_weight:
            limit = self.action_target_limit
            target_step = actions.clamp(-limit, limit) - self.last_actions.clamp(-limit, limit)
            action_cost = 0.08 * target_step.square().sum(dim=1)
            body_cost = 0.14 * (ang_vel[:, :2] / 0.30).square().clamp_max(4.0).mean(dim=1)
            if self.braking_pitch_weight:
                body_cost = body_cost + 0.18 * recovery * (ang_vel[:, 1] / 0.30).square().clamp_max(4.0)
            cost = cost + self.settled_motion_weight * self.motion_settled * (action_cost + body_cost)
        if self.height_transition_weight:
            transition = (self.height_transition_time / 0.15).clamp_max(1.0) * self.height_transition_supported
            roll_error = (self.posture[:, 0] - 0.50).clamp_min(0.0) / 2.0
            roll_rate = ang_vel[:, 0] / 0.35
            extension_gap = self.leg_height_rate_gap / 0.15
            target_curvature = self.leg_target_curvature / 0.02
            cost = cost + self.height_transition_weight * transition * (
                smooth_l1_loss(roll_error, torch.zeros_like(roll_error), reduction="none")
                + 0.35 * smooth_l1_loss(roll_rate, torch.zeros_like(roll_rate), reduction="none")
                + 0.25 * smooth_l1_loss(extension_gap, torch.zeros_like(extension_gap), reduction="none")
                + 0.15 * smooth_l1_loss(target_curvature, torch.zeros_like(target_curvature), reduction="none").mean(dim=1)
            )
        if self.steady_speed_weight:
            # Gradually emphasize speed error after the straight-line reference has settled.
            steady = ((self.steady_forward_time - 0.25) / 0.25).clamp(0.0, 1.0)
            speed_error = (lin_vel[:, 0] - self.commands[:, 0]) / 0.25
            cost = cost + self.steady_speed_weight * steady * smooth_l1_loss(
                speed_error, torch.zeros_like(speed_error), reduction="none"
            )
        return cost

    def _after_step(self, extras, done, lin_vel, ang_vel):
        if self.height_curriculum_rollout:
            self.height_curriculum_step += 1
            iteration, remainder = divmod(self.height_curriculum_step, self.height_curriculum_rollout)
            if not remainder:
                self._set_height_curriculum_iteration(iteration)
        self.episode_steps += 1
        self.velocity_errors += (lin_vel[:, 0] - self.commands[:, 0]).abs()
        self.yaw_errors += (ang_vel[:, 2] - self.commands[:, 1]).abs()
        self.posture_sums += self.posture
        torch.maximum(self.roll_peaks, self.posture[:, 0], out=self.roll_peaks)
        pitch = self.pitch_deg.abs()
        self.pitch_sums += pitch
        torch.maximum(self.pitch_peaks, pitch, out=self.pitch_peaks)
        leg_force = self.robot.get_dofs_control_force(self.leg_dofs)
        saturated = (leg_force.abs() >= 0.99 * self.leg_torque_limit).any(dim=1)
        self.leg_saturated_steps += saturated
        extras["leg_saturated"] = saturated
        extras["leg_heights"] = self.measured_heights.clone()
        extras["body_clearance"] = self.body_clearance.clone()
        extras["posture"] = self.posture.clone()
        extras["pitch_deg"] = self.pitch_deg.clone()
        finished = done.nonzero(as_tuple=True)[0]
        if finished.numel():
            count = self.episode_steps[finished]
            roll, fore_aft, height_gap = (self.posture_sums[finished] / count[:, None]).unbind(dim=1)
            height_metric = "body_height_mae_cm" if self.body_height_tracking else "height_mae_cm"
            extras["log"] = {
                height_metric: 100.0 * self.height_errors[finished] / count,
                "forward_mae_m_s": self.velocity_errors[finished] / count,
                "yaw_mae_rad_s": self.yaw_errors[finished] / count,
                "fall_rate": extras["fallen"][finished].float(),
                "leg_saturation_rate": self.leg_saturated_steps[finished] / count,
                "roll_abs_deg": roll,
                "roll_peak_deg": self.roll_peaks[finished],
                "pitch_abs_deg": self.pitch_sums[finished] / count,
                "pitch_peak_deg": self.pitch_peaks[finished],
                "leg_fore_aft_gap_cm": fore_aft,
                "leg_height_gap_cm": height_gap,
            }
            if self.terrain_traversal:
                extras["log"]["terrain_progress_m"] = self.traversal_distance[finished]
        if self.auto_commands:
            self.command_time -= self.control_dt
            due_motion = (self.command_time <= 0).nonzero(as_tuple=True)[0]
            self._sample_motion_goals(due_motion)
            due_height = due_motion
            if self.speed_maneuvers:
                self.height_time -= self.control_dt
                due_height = (self.height_time <= 0).nonzero(as_tuple=True)[0]
            self._sample_height_goals(due_height)
            self._advance_commands(lin_vel, ang_vel)
