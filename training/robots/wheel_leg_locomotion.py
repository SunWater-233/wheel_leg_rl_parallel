"""V0.2 locomotion with continuous hip-to-wheel height commands."""

from math import radians

import genesis as gs
import torch
from genesis.utils.geom import inv_transform_by_quat
from tensordict import TensorDict

from configs.experiment import MAX_FORWARD_ACCELERATION, MAX_FORWARD_SPEED, YAW_REFERENCE
from envs.terrain import SCUT_GROUND_SINGLE_SIDE_SURFACES, TERRAIN_LANES, ScutGroundLanes, ScutV1Lanes
from robots.domain_randomization import randomize_dynamics
from robots.scut_rough_reward import ScutRoughReward
from robots.wheel_leg import HOME_JOINTS, WheelLegFlatEnv
from robots.wheel_leg_stance import STANCE_SPAWN_GAP, solve_stances


class WheelLegLocomotionEnv(WheelLegFlatEnv):
    task_name = "locomotion"
    wheel_target_limit = 2.9224 / WheelLegFlatEnv.wheel_radius
    half_track = 0.21865
    forward_acceleration = MAX_FORWARD_ACCELERATION
    yaw_acceleration = 24.0
    drop_approach_height = 0.23
    landing_seconds = 0.32
    critic_scan_distances = (-0.30, -0.18, -0.09, 0.0, 0.09, 0.18, 0.30)
    critic_terrain_dims = 2 * (1 + len(critic_scan_distances))

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
        self.lanes = TERRAIN_LANES[settings["terrain"]](settings)
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
        self.body_height_offset = self.pose_table[0, 0].item() - self.height_low - STANCE_SPAWN_GAP
        base = self.robot.get_joint("base_link").qs_idx_local
        self.base_xy = base[:2]
        self.base_quat = base[3:7]
        self.pose_indices = [base[2], *(self.robot.get_joint(name).qs_idx_local[0] for name in HOME_JOINTS)]
        self.leg_columns = [list(HOME_JOINTS).index(name) + 1 for name in self.leg_joints]
        self.hips = [self.robot.get_joint(f"hip_{side}") for side in "LR"]
        self.goals = torch.zeros_like(self.commands)
        heading = settings["heading_hold"]
        self.heading_hold_enabled = heading["enabled"]
        if self.heading_hold_enabled:
            self.heading_target = self.commands.new_zeros(self.num_envs)
            self.heading_turning = torch.zeros_like(self.heading_target, dtype=torch.bool)
            self.heading_gain = heading["gain"]
            self.heading_deadband = radians(heading["deadband_deg"])
            self.heading_max_yaw_rate = heading["max_yaw_rate"]
        self.motion_scales = self.commands.new_tensor((1.0, 1.0, 5.0))
        ranges = self.cfg["forward_command"], self.cfg["yaw_command"], settings["height_range"]
        self.command_lower, self.command_upper = self.commands.new_tensor(ranges).T
        self.motion_step = self.commands.new_tensor((self.forward_acceleration, self.yaw_acceleration))
        self.motion_step = self.motion_step.expand(self.num_envs, -1).clone() * self.control_dt
        self.manual_forward_acceleration = settings["manual_forward_acceleration"]
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
        self.critic_scan_offsets = None
        if settings["critic_terrain_observations"]:
            self.critic_scan_offsets = self.commands.new_tensor(self.critic_scan_distances)
            self.critic_wheel_contacts = torch.zeros_like(self.commands[:, :2], dtype=torch.bool)
        self.height_step = settings["height_rate"] * self.control_dt
        self.height_acceleration = settings["height_acceleration"]
        self.command_time = self.commands.new_zeros(self.num_envs)
        self.command_period = settings["command_seconds"]
        self.stationary_episode_fraction = settings["stationary_episode_fraction"]
        if self.stationary_episode_fraction:
            self.stationary_episode = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
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
        self.posture = self.commands.new_zeros((self.num_envs, 3))
        self.posture_sums = torch.zeros_like(self.posture)
        self.roll_peaks = torch.zeros_like(self.command_time)
        self.pitch_deg = torch.zeros_like(self.command_time)
        self.pitch_sums = torch.zeros_like(self.pitch_deg)
        self.pitch_peaks = torch.zeros_like(self.pitch_deg)
        self.episode_steps = torch.zeros_like(self.episode_length_buf)
        self.height_errors = torch.zeros_like(self.command_time)
        self.velocity_errors = torch.zeros_like(self.command_time)
        self.yaw_errors = torch.zeros_like(self.command_time)
        self.leg_saturated_steps = torch.zeros_like(self.episode_steps)
        self.spawn_y = torch.zeros_like(self.command_time)
        self.terrain_selection = torch.zeros((2, self.num_envs), dtype=self.episode_length_buf.dtype, device=self.device)
        self.surface, self.terrain_variant = self.terrain_selection.unbind()
        self.terrain_traversal = settings["terrain_traversal"]
        self.terrain_forward_limit = settings["terrain_forward_limit"]
        if self.terrain_traversal:
            self.traversal_reverse = torch.zeros_like(self.surface, dtype=torch.bool)
            self.traversal_distance = torch.zeros_like(self.command_time)
        self.auto_commands = True
        self.demo_surface = None
        self.demo_initial_height = self.height_nominal
        self.demo_reverse = False
        self.demo_variant = 0
        self.forward_drop = bool(settings["cliff_drop_heights"])
        self.surface_variant_weights = None
        self.single_side_surfaces = None
        self.low_rough_slope = None
        self.bar_speed = None
        if self.lanes is not None:
            self.lane_centers = self.commands.new_tensor(self.lanes.centers)
            self.surface_weights = self.commands.new_tensor(self.lanes.surface_weights)
        if isinstance(self.lanes, ScutV1Lanes):
            self.variant_weights = self.commands.new_tensor(self.lanes.variant_weights)
            self.lane_spawn_x = self.commands.new_tensor(self.lanes.spawn_x)
            self.lane_spawn_z = self.commands.new_tensor(self.lanes.spawn_z)
            self.spawn_x = torch.zeros_like(self.spawn_y)
            self.reverse_probabilities = (self.commands.new_tensor(self.lanes.reverse_probabilities)
                                          if isinstance(self.lanes, ScutGroundLanes) else None)
        if isinstance(self.lanes, ScutGroundLanes):
            if self.lanes.surface_variant_weights is not None:
                self.surface_variant_weights = self.commands.new_tensor(self.lanes.surface_variant_weights)
            if settings["scut_ground_single_side_fraction"]:
                self.single_side_surfaces = self.surface.new_tensor(
                    [surface in SCUT_GROUND_SINGLE_SIDE_SURFACES for surface in self.lanes.surfaces],
                    dtype=torch.bool,
                )
            self.low_rough_slope_probability = settings["scut_ground_rough_slope_low_height_probability"]
            if self.low_rough_slope_probability:
                self.rough_slope_surface = self.lanes.surfaces.index("rough_slope")
                self.low_rough_slope = torch.zeros_like(self.surface, dtype=torch.bool)
            if settings["scut_ground_bars_speed_levels"]:
                self.bar_surface = self.lanes.surfaces.index("bars")
                self.bar_speed_levels = self.commands.new_tensor(settings["scut_ground_bars_speed_levels"])
                self.bar_speed = torch.zeros_like(self.command_time)
        if self.forward_drop:
            self.cliff_surface = self.lanes.surfaces.index("cliff")
            self.deck_height = torch.zeros_like(self.command_time)
            self.drop_touched = torch.zeros_like(self.surface, dtype=torch.bool)
            self.landing_elapsed = self.command_time.new_full((self.num_envs,), self.landing_seconds)

        self.scut_reward = ScutRoughReward(self, airborne_minimum_height=self.height_low + self.body_height_offset)

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
            if self.terrain_forward_limit:
                limit = limit.clamp(-self.terrain_forward_limit, self.terrain_forward_limit)
            cruise = limit * (0.65 + 0.35 * sample[:, 0])
            if self.bar_speed is not None:
                bar_cruise = limit.sign() * torch.minimum(limit.abs(), self.bar_speed[ids])
                cruise = torch.where(surface == self.bar_surface, bar_cruise, cruise)
            if self.single_side_surfaces is not None:
                # Both directions vary between two thirds of the cap and full speed.
                side_cruise = limit.clamp(-0.60, 0.60) * ((2.0 + sample[:, 0]) / 3.0)
                cruise = torch.where(self.single_side_surfaces[surface], side_cruise, cruise)
            if self.cliff_approach_speed:
                airborne = cliff & (~self.drop_touched[ids] | (self.landing_elapsed[ids] < self.landing_seconds))
                approach = self.cliff_approach_speed * (0.80 + 0.20 * sample[:, 0])
                cruise = torch.where(airborne, torch.minimum(cruise, approach), cruise)
            motion[:, 0] = torch.where(traverse, cruise, motion[:, 0])
            motion[:, 1] = torch.where(traverse, 0.0, motion[:, 1])
        if self.stationary_episode_fraction:
            motion.masked_fill_(self.stationary_episode[ids, None], 0.0)
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
        if self.stationary_episode_fraction:
            height = torch.where(self.stationary_episode[ids], self.goals[ids, 2], height)
        if self.low_rough_slope is not None:
            height = torch.where(self.low_rough_slope[ids], height.clamp_max(self.height_nominal), height)
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
                if self.surface_variant_weights is not None:
                    variant = torch.multinomial(self.surface_variant_weights[self.surface[ids]], 1).squeeze(1)
                elif scut_course:
                    variant = torch.multinomial(self.variant_weights, ids.numel(), replacement=True)
                else:
                    variant = torch.randint(self.lanes.variants, (ids.numel(),), device=self.device)
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
                if self.auto_commands:
                    reverse_probability = (0.5 if self.reverse_probabilities is None else
                                           self.reverse_probabilities[self.surface[ids]])
                    reverse = torch.rand(ids.numel(), device=self.device) < reverse_probability
                else:
                    reverse = torch.full((ids.numel(),), self.demo_reverse, device=self.device, dtype=torch.bool)
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
            if self.low_rough_slope is not None and self.auto_commands:
                focus = ((self.surface[ids] == self.rough_slope_surface)
                         & (torch.rand(ids.numel(), device=self.device) < self.low_rough_slope_probability))
                self.low_rough_slope[ids] = focus
                height = torch.where(focus, height.clamp_max(self.height_nominal), height)
        qpos[:, self.pose_indices] = self._poses(height)
        qpos[:, self.pose_indices[0]] += ground_height
        self.robot.set_qpos(qpos, envs_idx=ids, zero_velocity=True)
        if self.heading_hold_enabled:
            self.heading_target[ids] = self._yaw_from_quat(qpos[:, self.base_quat])
            self.heading_turning[ids] = False
        if self.critic_scan_offsets is not None:
            self.critic_wheel_contacts[ids] = False
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
        self.leg_saturated_steps[ids] = 0
        self.last_actions[ids] = 0.0
        self.scut_reward.reset(ids)
        self.height_rate[ids] = 0.0
        self.commands[ids] = 0.0
        self.commands[ids, 2] = height
        self.measured_heights[ids] = height[:, None]
        self.goals[ids] = self.commands[ids]
        if self.speed_levels.numel() and self.auto_commands:
            levels = torch.randint(self.speed_levels.numel(), (ids.numel(),), device=self.device)
            self.episode_speed_limit[ids] = self.speed_levels[levels]
        if not self.auto_commands:
            self.motion_step[ids, 0] = self.manual_forward_acceleration * self.control_dt
        elif self.acceleration_levels.numel():
            if self.acceleration_weights.numel():
                levels = torch.multinomial(self.acceleration_weights, ids.numel(), replacement=True)
            else:
                levels = torch.randint(self.acceleration_levels.numel(), (ids.numel(),), device=self.device)
            self.motion_step[ids, 0] = self.acceleration_levels[levels] * self.control_dt
        if self.auto_commands:
            if self.bar_speed is not None:
                level = torch.randint(self.bar_speed_levels.numel(), (ids.numel(),), device=self.device)
                self.bar_speed[ids] = self.bar_speed_levels[level]
            if self.stationary_episode_fraction:
                stationary = torch.rand(ids.numel(), device=self.device) < self.stationary_episode_fraction
                self.stationary_episode[ids] = stationary & (self.surface[ids] == 0)
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

    def begin_demo(self, height, surface, reverse=False, variant=0):
        self.auto_commands = False
        self.demo_initial_height = height
        self.demo_surface = self.lanes.surfaces.index(surface) if self.lanes is not None else None
        self.demo_reverse = reverse
        self.demo_variant = variant
        self.reset()

    def set_commands(self, forward, yaw, height):
        self.goals[:] = self.commands.new_tensor((forward, yaw, height))
        lin_vel, ang_vel, _, _ = super()._base_state()
        self._advance_commands(lin_vel, ang_vel)

    def _advance_commands(self, lin_vel, ang_vel):
        targets = self.goals.clamp(min=self.command_lower, max=self.command_upper)
        if self.heading_hold_enabled:
            targets[:, 1] = self._heading_yaw_target(targets[:, 1]).clamp(
                min=self.command_lower[1], max=self.command_upper[1]
            )
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
        self.commands[:, :2] += motion_step
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

    @staticmethod
    def _yaw_from_quat(quat):
        w, x, y, z = quat.unbind(dim=1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))

    def _heading_yaw_target(self, requested_rate):
        yaw = self._yaw_from_quat(self.robot.get_quat())
        turning = requested_rate.abs() > 1e-6
        # Latch the measured heading on release, including after a turn at zero forward speed.
        self.heading_target.copy_(torch.where(turning | self.heading_turning, yaw, self.heading_target))
        self.heading_turning.copy_(turning)
        error = self.heading_target - yaw
        error = torch.atan2(error.sin(), error.cos())
        correction = self.heading_gain * error.sign() * (error.abs() - self.heading_deadband).clamp_min(0.0)
        return torch.where(turning, requested_rate, correction.clamp(-self.heading_max_yaw_rate, self.heading_max_yaw_rate))

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
        critic_inputs = (policy, lin_vel)
        if self.critic_scan_offsets is not None:
            critic_inputs += self._critic_terrain_observations()
        critic = torch.cat(critic_inputs, dim=1)
        if self.observation_noise is not None:
            # The critic retains the original state; commands, history and inferred leg height stay intact.
            sensors = policy[:, 4:20]
            sensors.add_(torch.randn_like(sensors) * self.observation_noise)
        return TensorDict({"policy": policy, "critic": critic}, batch_size=[self.num_envs])

    def _critic_terrain_observations(self):
        # Scan both travel directions along each wheel track in the horizontal plane.
        yaw = self._yaw_from_quat(self.robot.get_quat())
        heading = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), dim=1)
        wheels = torch.stack([wheel.get_anchor_pos() for wheel in self.wheels], dim=1)
        offsets = heading[:, None, :] * self.critic_scan_offsets[None, :, None]
        points = wheels[:, :, None, :] + offsets[:, None, :, :]
        ground = self.ground_height_at(points)
        relative_height = ground - (wheels[:, :, 2:] - self.wheel_radius)
        relative_height = relative_height.nan_to_num().clamp(-0.20, 0.20)
        return self.critic_wheel_contacts.to(wheels.dtype), 10.0 * relative_height.flatten(1)

    def _drive(self, actions):
        nominal = self._poses(self.commands[:, 2])[:, self.leg_columns]
        offsets = (actions[:, :4] * self.leg_action_scale).clamp(-0.35, 0.35)
        targets = nominal + offsets * self.leg_direction + self.leg_bias
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

    def _termination(self, height, gravity, contact):
        heights, fore_aft = self._leg_projections(gravity)
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
            travel_limit = self.lanes.travel_limit
            if isinstance(self.lanes, ScutGroundLanes):
                travel_limit = torch.where(self.surface == 0, self.lanes.flat_travel_limit, travel_limit)
            out_of_bounds = ((pos[:, 0].abs() >= travel_limit)
                             | ((pos[:, 1] - self.spawn_y).abs() >= self.lanes.lateral_limit)
                             | ~torch.isfinite(surface_z))

        if self.ground is not None:
            clearance_for_reward = self.scut_reward.measure_clearance(clearance)
        else:
            clearance_for_reward = clearance
        self.body_clearance.copy_(clearance_for_reward)
        height_target = self.commands[:, 2] + self.body_height_offset
        self.height_errors += (self.body_clearance - height_target).abs()
        if self.forward_drop:
            self._update_drop_landing(pos, contact)
        fallen = body_contact | (clearance < 0.045) | (gravity[:, 2] > -0.65)
        return fallen, out_of_bounds

    def _update_drop_landing(self, pos, contact):
        forces = contact[:, self.wheel_links, 2].clamp_min(0.0)
        left, right = forces.unbind(dim=1)
        crossed_edge = pos[:, 0].abs() > self.lanes.deck_half_length + 0.04
        touchdown = (self.deck_height > 0.0) & crossed_edge & ~self.drop_touched & ((left + right) > 5.0)
        self.drop_touched |= touchdown
        self.landing_elapsed.add_(self.control_dt).clamp_max_(self.landing_seconds)
        self.landing_elapsed.masked_fill_(touchdown, 0.0)

    def _after_step(self, extras, done, lin_vel, ang_vel):
        if self.critic_scan_offsets is not None:
            self.critic_wheel_contacts.copy_(extras["wheel_contacts"])
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
            extras["log"] = {
                "body_height_mae_cm": 100.0 * self.height_errors[finished] / count,
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
