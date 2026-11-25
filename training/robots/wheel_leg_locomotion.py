"""V0.2 locomotion with continuous hip-to-wheel height commands."""

from itertools import product
from math import ceil, cos, radians, tau

import genesis as gs
import torch
from genesis.utils.geom import inv_transform_by_quat
from tensordict import TensorDict

from configs.experiment import MAX_FORWARD_ACCELERATION, MAX_FORWARD_SPEED, YAW_REFERENCE
from envs.terrain import SCUT_GROUND_SINGLE_SIDE_SURFACES, TERRAIN_LANES, ScutGroundLanes, ScutV1Lanes, TransitionLanes
from robots.domain_randomization import randomize_dynamics
from robots.scut_rough_reward import ScutRoughReward
from robots.special_mode_commands import SpecialModeUniformVelocityCommand
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

    def __init__(self, *args, preview_lane=None, **kwargs):
        self.preview_lane = preview_lane
        super().__init__(*args, **kwargs)

    def _robot_morph(self, model_path, viewer):
        settings = self.cfg["locomotion"]
        height_limits = settings["height_range"]
        self.pose_table = solve_stances(model_path, height_limits, self.wheel_radius)
        self.recovery_fraction = settings["low_recovery"]["flat_episode_fraction"]
        if self.recovery_fraction:
            offsets = tuple(product((-0.11, -0.075, 0.075, 0.11), repeat=2))
            self.recovery_poses = solve_stances(model_path, height_limits, self.wheel_radius, wheel_offsets=offsets)
        return super()._robot_morph(model_path, viewer)

    def _add_ground(self, material, viewer):
        self.lanes = None
        self.ground = None
        self.terrain_contacts = []
        settings = self.cfg["locomotion"]
        if settings["terrain"] == "plane":
            super()._add_ground(material, viewer)
            self.terrain_contacts.append(self.plane_ground)
            return
        self.lanes = self._make_lanes(settings)
        lane = None
        if self.preview_lane is not None:
            name, variant = self.preview_lane
            lane = self.lanes.surfaces.index(name) * self.lanes.variants + variant
        surface = self.lanes.visual_surface(lane) if viewer else None
        self.ground = self.scene.add_entity(self.lanes.morph(viewer, lane), material=material, surface=surface)
        self.terrain_contacts.append(self.ground)
        if isinstance(self.lanes, ScutV1Lanes):
            deck_surface = gs.surfaces.Default(color=(0.73, 0.47, 0.27), roughness=0.95) if viewer else None
            for center, half_size in self.lanes.deck_boxes():
                box = gs.morphs.Box(pos=center, size=tuple(2 * extent for extent in half_size),
                                    fixed=True, visualization=viewer)
                self.terrain_contacts.append(self.scene.add_entity(box, material=material, surface=deck_surface))

    def _make_lanes(self, settings):
        return TERRAIN_LANES[settings["terrain"]](settings)

    def terrain_description(self):
        if self.lanes is None:
            return super().terrain_description()
        surface, variant = self.terrain_selection[:, 0].tolist()
        return self.lanes.describe(surface, variant)

    def _set_drives(self):
        settings = self.cfg["locomotion"]
        self.leg_torque_limit = settings["leg_torque_limit"]
        self.wheel_torque_limit = settings["wheel_torque_limit"]
        self.wheel_target_delay_steps = settings["wheel_target_delay_steps"]
        super()._set_drives()

    def _initialize_task(self):
        settings = self.cfg["locomotion"]
        randomization = self.cfg.get("domain_randomization", {})
        if randomization.get("enabled"):
            randomize_dynamics(self, randomization)
        self.height_low, self.height_high = settings["height_range"]
        self.obstacle_height_range = tuple(settings.get("obstacle_height_range", ()))
        self.hard_obstacle_height_range = tuple(settings.get("hard_obstacle_height_range", ()))
        self.height_nominal = settings["height_nominal"]
        self.height_max_probability = settings["height_max_probability"]
        self.height_lowering_schedule = settings["height_lowering_schedule"]
        self.height_sampling_floor = self.height_low
        self.training_rollout_steps = 0
        self.training_step_counter = 0
        self.training_iteration = 0
        self.height_curriculum_rollout = 0
        self._set_height_curriculum_iteration(0)
        self.pose_table = self.home_qpos.new_tensor(self.pose_table)
        if self.recovery_fraction:
            self.recovery_poses = self.home_qpos.new_tensor(self.recovery_poses)
            recovery = settings["low_recovery"]
            self.recovery_unpowered_steps = ceil(recovery["unpowered_seconds"] / self.control_dt)
            self.recovery_deadline_steps = ceil(recovery["deadline_seconds"] / self.control_dt)
            self.recovery_hold_steps = ceil(0.24 / self.control_dt)
            if self.recovery_deadline_steps < self.recovery_unpowered_steps + self.recovery_hold_steps:
                raise ValueError("Low recovery deadline must leave at least 0.24 s after touchdown")
            self.recovery_steps = torch.full_like(self.episode_length_buf, -1)
            self.recovery_stable_steps = torch.zeros_like(self.recovery_steps)
            self.recovery_succeeded = torch.zeros_like(self.recovery_steps, dtype=torch.bool)
            self.recovery_failed = torch.zeros_like(self.recovery_succeeded)
            self.recovery_counts = self.recovery_steps.new_zeros(2)
            self.recovery_zero_force = self.home_qpos.new_zeros((self.num_envs, self.num_actions))
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
        self.manual_brake_acceleration = None
        self.direct_velocity_commands = settings["direct_velocity_commands"]
        self.direct_velocity_probability = settings.get("direct_velocity_probability", 1.0)
        self.direct_episode = torch.full(
            (self.num_envs,), self.direct_velocity_commands, dtype=torch.bool, device=self.device
        )
        special_cfg = settings["special_mode_commands"]
        self.special_command_sampler = (
            SpecialModeUniformVelocityCommand(
                special_cfg,
                num_envs=self.num_envs,
                device=self.device,
                default_forward_range=self.cfg["forward_command"],
                default_yaw_range=self.cfg["yaw_command"],
            )
            if special_cfg["enabled"] else None
        )
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
        self.actor_true_forward_speed = settings["actor_true_forward_speed"]
        self.actor_history_steps = settings["actor_history_steps"]
        self.policy_obs_dim = 28
        if self.actor_history_steps > 1:
            self.policy_history = self.commands.new_zeros((self.num_envs, self.actor_history_steps - 1,
                                                           self.policy_obs_dim))
            self.last_policy_obs = self.commands.new_zeros((self.num_envs, self.policy_obs_dim))
            self.history_needs_fill = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            self.policy_history = None
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
        self.height_maneuver_sequence = self.commands.new_tensor(settings["height_maneuver_sequence"])
        if self.height_maneuver_sequence.numel():
            self.height_sequence_index = torch.zeros_like(self.episode_length_buf)
        self.launch_press_seconds = self.commands.new_tensor(settings["launch_press_seconds"])
        self.launch_stop_seconds = self.commands.new_tensor(settings["launch_stop_seconds"])
        self.maneuver_weights = self.commands.new_tensor(settings["speed_maneuver_weights"])
        self.launch_mode = 5 if self.braking_maneuvers else 4
        self.height_mode = self.launch_mode + int(self.launch_maneuvers)
        if self.speed_maneuvers or self.special_command_sampler is not None:
            self.height_time = torch.zeros_like(self.command_time)
        if self.speed_maneuvers:
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
        self.leg_height_errors = torch.zeros_like(self.height_errors)
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
        self.auto_commands = self.preview_lane is None
        self.free_drive = False
        self.demo_surface = None
        self.demo_initial_height = self.height_nominal
        self.demo_reverse = False
        self.demo_variant = 0
        if self.preview_lane is not None:
            name, self.demo_variant = self.preview_lane
            self.demo_surface = self.lanes.surfaces.index(name)
        self.forward_drop = bool(settings["cliff_drop_heights"])
        self.surface_variant_weights = None
        self.single_side_surfaces = None
        self.low_rough_slope = None
        self.bar_speed = None
        self.transition_course = isinstance(self.lanes, TransitionLanes)
        self.transition_stop_feedback = settings["transition_stop_feedback"]
        if self.transition_course:
            self.transition_speed_levels = self.commands.new_tensor(settings["transition_speed_levels"])
            self.transition_reverse_spawn_z = self.commands.new_tensor(self.lanes.reverse_spawn_z)
            self.transition_forward_only = self.surface.new_tensor(self.lanes.forward_only, dtype=torch.bool)
            self.transition_stop_at = torch.zeros_like(self.command_time)
            self.transition_stop_phase = torch.zeros_like(self.surface)
            self.transition_still_time = torch.zeros_like(self.command_time)
            self.transition_stop_elapsed = torch.zeros_like(self.command_time)
            self.transition_peak_command = torch.zeros_like(self.command_time)
        self.high_stair_surfaces = None
        self.hard_obstacle_surfaces = None
        self.spin_obstacle_surfaces = None
        if self.lanes is not None:
            self.lane_centers = self.commands.new_tensor(self.lanes.centers)
            self.surface_weights = self.commands.new_tensor(self.lanes.surface_weights)
            self.high_stair_surfaces = self.surface.new_tensor(
                [surface in ("stairs_slow", "stairs_fast") for surface in self.lanes.surfaces],
                dtype=torch.bool,
            )
            self.hard_obstacle_surfaces = self.surface.new_tensor(
                [
                    surface in (
                        "bars", "stairs", "inv_stairs_high",
                        "single_side_bars_left", "single_side_bars_right",
                        "stairs_slow", "stairs_fast",
                    )
                    for surface in self.lanes.surfaces
                ],
                dtype=torch.bool,
            )
            self.spin_obstacle_surfaces = self.surface.new_tensor(
                [
                    surface in ("slope", "inv_slope", "rough", "rough_slope", "inv_stairs_low")
                    for surface in self.lanes.surfaces
                ],
                dtype=torch.bool,
            )
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

        self.terrain_cmd_cfg = settings.get("terrain_commands", {"enabled": False})
        self.terrain_commands_enabled = bool(self.terrain_cmd_cfg.get("enabled", False))
        self.terrain_spinning = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        curriculum_cfg = settings.get("terrain_curriculum", {"enabled": False})
        self.terrain_curriculum_enabled = (
            bool(curriculum_cfg.get("enabled", False)) and self.lanes is not None and self.lanes.variants >= 2
        )
        if self.terrain_curriculum_enabled:
            self.curriculum_promote_distance = curriculum_cfg["promote_distance"]
            self.curriculum_demote_distance = curriculum_cfg["demote_distance"]
            self.curriculum_max_weights = self.commands.new_tensor(curriculum_cfg["max_level_weights"])
            self.terrain_level = (
                (torch.rand(self.num_envs, device=self.device) < 0.30).long().clamp_max(self.lanes.variants - 1)
            )
            self.terrain_maxed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.events_cfg = settings.get("events", {"enabled": False})
        self.events_enabled = bool(self.events_cfg.get("enabled", False))
        self.base_dofs = list(self.robot.get_joint("base_link").dofs_idx_local)
        if self.events_enabled:
            self.push_time = self.commands.new_zeros(self.num_envs)
            self.wrench_time = self.commands.new_zeros(self.num_envs)
            self.ext_wrench = self.commands.new_zeros((self.num_envs, 6))

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

    def _sample_heights_for_envs(self, ids):
        height = self._sample_heights(ids.numel())
        obs_range = getattr(self, "obstacle_height_range", ())
        hard_range = getattr(self, "hard_obstacle_height_range", ())
        if self.lanes is None or not (obs_range or hard_range):
            return height
        surf = self.surface[ids]
        traverse = surf != 0
        if not traverse.any():
            return height
        u = torch.rand((2, ids.numel()), device=self.device)
        if obs_range:
            olow, ohigh = obs_range
            mid_cap = min(ohigh, 0.24)
            obs_h = torch.where(
                u[1] < 0.35,
                olow + u[0] * (mid_cap - olow),
                olow + u[0] * (ohigh - olow),
            )
            height = torch.where(traverse, obs_h, height)
        if hard_range and getattr(self, "hard_obstacle_surfaces", None) is not None:
            hlow, hhigh = hard_range
            mid_cap = min(hhigh, 0.24)
            hard_h = torch.where(
                u[1] < 0.45,
                hlow + u[0] * (mid_cap - hlow),
                hlow + u[0] * (hhigh - hlow),
            )
            height = torch.where(traverse & self.hard_obstacle_surfaces[surf], hard_h, height)
        return height

    def start_height_curriculum(self, iteration, rollout_steps):
        self.training_rollout_steps = rollout_steps
        self.training_step_counter = iteration * rollout_steps
        self.training_iteration = iteration
        if self.special_command_sampler is not None:
            self.special_command_sampler.set_training_iteration(iteration)
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

    def _sample_special_mode_goals(self, ids):
        episode_time = self.episode_length_buf.float() * self.control_dt
        lin_vel = ang_vel = gravity = None
        if self.special_command_sampler.require_stable:
            lin_vel, ang_vel, gravity, _ = super()._base_state()
        return self.special_command_sampler.sample(
            ids,
            episode_time=episode_time,
            gravity=gravity,
            lin_vel=lin_vel,
            ang_vel=ang_vel,
        )

    def _sample_terrain_command_goals(self, t_ids):
        cfg = self.terrain_cmd_cfg
        n = t_ids.numel()
        surf = self.surface[t_ids]
        is_hard = self.hard_obstacle_surfaces[surf]
        can_spin = self.spin_obstacle_surfaces[surf]
        pos = self.robot.get_pos()[t_ids]
        x = pos[:, 0]
        dy = pos[:, 1] - self.spawn_y[t_ids]
        in_zone = x.abs() <= cfg["engage_x"]

        u = torch.rand((n, 5), device=self.device)
        p_spin = cfg["spin_probability"]
        p_stop = cfg["stop_probability"]
        do_spin = in_zone & can_spin & (u[:, 0] < p_spin)
        do_stop = in_zone & ~do_spin & (u[:, 0] < p_spin + p_stop)
        do_cruise = ~(do_spin | do_stop)
        self.terrain_spinning[t_ids] = do_spin

        fast_low, fast_high = cfg["fast_forward_range"]
        hard_low, hard_high = cfg["hard_forward_range"]
        v_fast = fast_low + u[:, 1] * (fast_high - fast_low)
        v_hard = hard_low + u[:, 1] * (hard_high - hard_low)
        v_mag = torch.where(is_hard, v_hard, v_fast)
        sign_x = torch.where(self.traversal_reverse[t_ids], -1.0, 1.0)
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
        yaw_now = self._yaw_from_quat(self.robot.get_quat()[t_ids])
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

    def _sample_motion_goals(self, ids):
        if ids.numel() == 0:
            return
        sample = torch.rand((ids.numel(), 4), device=self.device)
        if self.special_command_sampler is not None and not self.terrain_traversal:
            motion, duration = self._sample_special_mode_goals(ids)
        else:
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
            if self.special_command_sampler is not None:
                flat = ~traverse
                if flat.any():
                    motion[flat], duration[flat] = self._sample_special_mode_goals(ids[flat])
                if traverse.any():
                    self.special_command_sampler.reset(ids[traverse])
            elif self.speed_maneuvers:
                flat = ~traverse
                motion[flat], duration[flat] = self._maneuver_goals(
                    ids[flat], motion[flat], duration[flat], sample[flat, 3]
                )
            if getattr(self, "terrain_commands_enabled", False):
                self.terrain_spinning[ids[~traverse]] = False
                if traverse.any():
                    motion[traverse], duration[traverse] = self._sample_terrain_command_goals(ids[traverse])
            else:
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
                if self.transition_course:
                    direction = torch.where(self.traversal_reverse[ids], -1.0, 1.0)
                    cruise = self.episode_speed_limit[ids] * direction
                    phase = self.transition_stop_phase[ids]
                    restart = torch.where(cruise.abs() > 0.3, 0.45 * cruise, 0.30 * direction)
                    cruise = torch.where(phase == 3, restart, cruise)
                    cruise = cruise.masked_fill((phase == 2) | (phase == 4), 0.0)
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
        if hasattr(self, "_sample_heights_for_envs"):
            height = self._sample_heights_for_envs(ids)
        else:
            height = self._sample_heights(ids.numel())
        if self.height_maneuvers:
            flat = self.surface[ids] == 0
            height_only = (self.maneuver[ids] == self.height_mode) & flat
            previous = self.goals[ids, 2]
            initial_height = self.episode_steps[ids] == 0
            if self.height_maneuver_sequence.numel():
                index = self.height_sequence_index[ids]
                index = (index + (height_only & ~initial_height)).remainder(self.height_maneuver_sequence.numel())
                self.height_sequence_index[ids] = index
                height = torch.where(height_only, self.height_maneuver_sequence[index], height)
            else:
                midpoint = (self.height_high + self.height_low) / 2
                opposite = torch.where(previous < midpoint, self.height_high, self.height_low)
                height = torch.where(height_only, opposite, height)
            # Start with a fixed height so the cruise cases can reach their speed first.
            hold = flat & (~height_only | initial_height)
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
        if self.speed_maneuvers or getattr(self, "special_command_sampler", None) is not None:
            low, high = self.command_period
            self.height_time[ids] = low + torch.rand(ids.numel(), device=self.device) * (high - low)

    @staticmethod
    def _quat_mul(q1, q2):
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        return torch.stack((
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ), dim=-1)

    @staticmethod
    def _euler_zyx_to_quat(roll, pitch, yaw):
        cr, sr = (0.5 * roll).cos(), (0.5 * roll).sin()
        cp, sp = (0.5 * pitch).cos(), (0.5 * pitch).sin()
        cy, sy = (0.5 * yaw).cos(), (0.5 * yaw).sin()
        return torch.stack((
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ), dim=-1)

    def _sample_event_wrenches(self, ids):
        if ids.numel() == 0:
            return
        cfg = self.events_cfg
        n = ids.numel()
        wlow, whigh = cfg["wrench_interval_seconds"]
        self.wrench_time[ids] = wlow + torch.rand(n, device=self.device) * (whigh - wlow)
        fmax = self.ext_wrench.new_tensor(cfg["external_force_xyz"])
        tmax = self.ext_wrench.new_tensor(cfg["external_torque_xyz"])
        limits = torch.cat((fmax, tmax))
        wrench = (2.0 * torch.rand((n, 6), device=self.device) - 1.0) * limits
        zero_mask = torch.rand(n, device=self.device) < cfg["wrench_zero_probability"]
        if self.recovery_fraction:
            zero_mask = zero_mask | (self.recovery_steps[ids] >= 0)
        wrench.masked_fill_(zero_mask[:, None], 0.0)
        self.ext_wrench[ids] = wrench

    def _apply_external_wrench(self):
        if getattr(self, "events_enabled", False) and self.auto_commands:
            self.robot.control_dofs_force(self.ext_wrench, self.base_dofs)

    def _reset_envs(self, ids):
        qpos = self.home_qpos[ids].clone()
        ground_height = 0.0
        scut_course = isinstance(self.lanes, ScutV1Lanes)
        if self.transition_course and self.speed_levels.numel() and self.auto_commands:
            levels = torch.randint(self.speed_levels.numel(), (ids.numel(),), device=self.device)
            self.episode_speed_limit[ids] = self.speed_levels[levels]
        if self.lanes is not None:
            if self.demo_surface is None:
                self.surface[ids] = torch.multinomial(self.surface_weights, ids.numel(), replacement=True)
                if getattr(self, "terrain_curriculum_enabled", False):
                    maxed = self.terrain_maxed[ids]
                    sampled_max = torch.multinomial(
                        self.curriculum_max_weights, ids.numel(), replacement=True
                    )
                    variant = torch.where(maxed, sampled_max, self.terrain_level[ids])
                elif self.surface_variant_weights is not None:
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
        height = (self._sample_heights_for_envs(ids) if self.auto_commands else
                  self.commands.new_full((ids.numel(),), self.demo_initial_height))
        if self.lanes is not None:
            if self.transition_course and self.auto_commands and self.transition_speed_levels.numel():
                levels = torch.randint(self.transition_speed_levels.numel(), (ids.numel(),), device=self.device)
                self.episode_speed_limit[ids] = torch.where(self.surface[ids] != 0, self.transition_speed_levels[levels],
                                                            self.episode_speed_limit[ids])
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
                if self.transition_course:
                    if self.auto_commands:
                        reverse = reverse & ~self.transition_forward_only[lane]
                    self.traversal_reverse[ids] = reverse
                    ground_height = torch.where(reverse, self.transition_reverse_spawn_z[lane], ground_height)
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
                approach = self.lane_spawn_x[lane]
                if self.transition_course and self.auto_commands:
                    long_runup = (self.surface[ids] != 0) & (self.episode_speed_limit[ids] >= 1.6)
                    approach = torch.where(long_runup, 9.0, approach)
                if self.free_drive and self.demo_surface == 0 and isinstance(self.lanes, ScutGroundLanes):
                    # Start on the clear apron shared by the obstacle lanes.
                    approach = torch.full_like(approach, self.lanes.approach_x)
                self.spawn_x[ids] = torch.where(positive_x, approach, -approach)
                qpos[:, self.base_xy[0]] = self.spawn_x[ids]
            if self.low_rough_slope is not None and self.auto_commands:
                focus = ((self.surface[ids] == self.rough_slope_surface)
                         & (torch.rand(ids.numel(), device=self.device) < self.low_rough_slope_probability))
                self.low_rough_slope[ids] = focus
                height = torch.where(focus, height.clamp_max(self.height_nominal), height)
        if self.auto_commands and self.height_maneuver_sequence.numel():
            self._sample_maneuvers(ids)
            self.height_sequence_index[ids] = 0
            height_episode = (self.maneuver[ids] == self.height_mode) & (self.surface[ids] == 0)
            height = torch.where(height_episode, self.height_maneuver_sequence[0], height)
        stance_height = height
        if getattr(self, "events_enabled", False) and self.auto_commands and self.events_cfg["reset_joint_pos_rad"] > 0:
            # Perturb stance height within closed-chain pose_table so all 12 4-bar equalities remain exact.
            h_jitter = (2.0 * torch.rand(ids.numel(), device=self.device) - 1.0) * (
                0.5 * self.events_cfg["reset_joint_pos_rad"]
            )
            stance_height = (height + h_jitter).clamp(self.height_low, self.height_high)
        stance = self._poses(stance_height)
        qpos[:, self.pose_indices] = stance
        qpos[:, self.pose_indices[0]] += ground_height
        nominal_yaw = self._yaw_from_quat(qpos[:, self.base_quat])
        if self.recovery_fraction:
            self._sample_recovery_starts(ids, qpos, stance[:, 0])
        air_drop_rows = None
        air_drop_vel = None
        init_base_vel = None
        if getattr(self, "events_enabled", False) and self.auto_commands:
            ev = self.events_cfg
            can_perturb = torch.ones(ids.numel(), dtype=torch.bool, device=self.device)
            if self.recovery_fraction:
                can_perturb &= self.recovery_steps[ids] < 0
            on_flat = (self.surface[ids] == 0) if self.lanes is not None else torch.ones_like(can_perturb)
            if ev["air_drop_flat_fraction"] > 0.0:
                air_mask = on_flat & can_perturb & (torch.rand(ids.numel(), device=self.device) < ev["air_drop_flat_fraction"])
                air_drop_rows = air_mask.nonzero(as_tuple=True)[0]
                if air_drop_rows.numel():
                    dlow, dhigh = ev["air_drop_height_range"]
                    drop_dz = dlow + torch.rand(air_drop_rows.numel(), device=self.device) * (dhigh - dlow)
                    qpos[air_drop_rows, self.pose_indices[0]] += drop_dz
                    if isinstance(self.lanes, ScutGroundLanes):
                        qpos[air_drop_rows, self.base_xy[0]] = (
                            (2.0 * torch.rand(air_drop_rows.numel(), device=self.device) - 1.0) * 2.0
                        )
            pose_perturb = can_perturb & (torch.rand(ids.numel(), device=self.device) < 0.75)
            if pose_perturb.any():
                rp_lim = ev["reset_roll_pitch_rad"]
                y_lim = ev["reset_yaw_rad"]
                roll = ((2.0 * torch.rand(ids.numel(), device=self.device) - 1.0) * rp_lim) * pose_perturb
                pitch = ((2.0 * torch.rand(ids.numel(), device=self.device) - 1.0) * rp_lim) * pose_perturb
                dyaw = ((2.0 * torch.rand(ids.numel(), device=self.device) - 1.0) * y_lim) * pose_perturb
                dq = self._euler_zyx_to_quat(roll, pitch, dyaw)
                qpos[:, self.base_quat] = self._quat_mul(qpos[:, self.base_quat], dq)
                lift = (self.half_track * roll.abs().sin() + 0.06 * pitch.abs().sin() + 0.006) * pose_perturb
                qpos[:, self.pose_indices[0]] += lift
            init_base_vel = qpos.new_zeros((ids.numel(), 6))
            vel_perturb = can_perturb & (torch.rand(ids.numel(), device=self.device) < 0.50)
            if vel_perturb.any():
                lv = ev["reset_lin_vel"]
                av = ev["reset_ang_vel"]
                v_rand = (2.0 * torch.rand((ids.numel(), 6), device=self.device) - 1.0)
                v_scales = qpos.new_tensor((lv, lv, 0.5 * lv, av, av, av))
                init_base_vel = (v_rand * v_scales) * vel_perturb[:, None]
            if air_drop_rows is not None and air_drop_rows.numel():
                vmax = max(0.55, ev["air_drop_speed_max"])
                vmag = 0.55 + torch.rand(air_drop_rows.numel(), device=self.device) * (vmax - 0.55)
                vsign = torch.where(torch.rand_like(vmag) < 0.5, -1.0, 1.0)
                air_drop_vel = vsign * vmag
                row_yaw = self._yaw_from_quat(qpos[air_drop_rows][:, self.base_quat])
                init_base_vel[air_drop_rows, 0] = air_drop_vel * row_yaw.cos()
                init_base_vel[air_drop_rows, 1] = air_drop_vel * row_yaw.sin()
                init_base_vel[air_drop_rows, 2] = 0.0
        self.robot.set_qpos(qpos, envs_idx=ids, zero_velocity=True)
        if init_base_vel is not None and (init_base_vel != 0.0).any():
            self.robot.set_dofs_velocity(init_base_vel, self.base_dofs, envs_idx=ids)
        if self.heading_hold_enabled:
            actual_yaw = self._yaw_from_quat(qpos[:, self.base_quat])
            if self.lanes is not None and self.auto_commands:
                self.heading_target[ids] = torch.where(self.surface[ids] == 0, actual_yaw, nominal_yaw)
            else:
                self.heading_target[ids] = actual_yaw
            self.heading_turning[ids] = False
        if getattr(self, "terrain_commands_enabled", False):
            self.terrain_spinning[ids] = False
        if getattr(self, "events_enabled", False):
            plow, phigh = self.events_cfg["push_interval_seconds"]
            self.push_time[ids] = plow + torch.rand(ids.numel(), device=self.device) * (phigh - plow)
            if self.auto_commands:
                self._sample_event_wrenches(ids)
            else:
                self.ext_wrench[ids] = 0.0
        if self.special_command_sampler is not None:
            self.special_command_sampler.reset(ids)
        if self.critic_scan_offsets is not None:
            self.critic_wheel_contacts[ids] = False
        if self.policy_history is not None:
            self.policy_history[ids] = 0.0
            self.last_policy_obs[ids] = 0.0
            self.history_needs_fill[ids] = True
        self.episode_length_buf[ids] = 0
        self.episode_steps[ids] = 0
        self.height_errors[ids] = 0.0
        self.leg_height_errors[ids] = 0.0
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
        self.pitch_exceed_time[ids] = 0.0
        self.scut_reward.reset(ids)
        self.height_rate[ids] = 0.0
        self.commands[ids] = 0.0
        self.commands[ids, 2] = height
        self.measured_heights[ids] = height[:, None]
        self.goals[ids] = self.commands[ids]
        if not self.transition_course and self.speed_levels.numel() and self.auto_commands:
            levels = torch.randint(self.speed_levels.numel(), (ids.numel(),), device=self.device)
            self.episode_speed_limit[ids] = self.speed_levels[levels]
        if self.transition_course:
            # Brake after reaching the first or second join; some episodes hold cruise throughout.
            sample = torch.rand((ids.numel(), 3), device=self.device)
            self.transition_stop_at[ids] = torch.where(sample[:, 0] < 0.70, -2.50, 0.35) + 0.2 * sample[:, 1]
            if self.transition_stop_feedback:
                self.transition_stop_at[ids] = torch.where(sample[:, 0] < 0.70, -2.8, 0.0) + 0.5 * sample[:, 1] - 0.15
            braking = (self.surface[ids] != 0) & (sample[:, 2] < 0.40) & self.auto_commands
            self.transition_stop_phase[ids] = braking.long()
            self.transition_still_time[ids] = 0.0
            self.transition_stop_elapsed[ids] = 0.0
            self.transition_peak_command[ids] = 0.0
        if not self.auto_commands:
            self.motion_step[ids, 0] = self.manual_forward_acceleration * self.control_dt
        elif self.acceleration_levels.numel():
            if self.acceleration_weights.numel():
                levels = torch.multinomial(self.acceleration_weights, ids.numel(), replacement=True)
            else:
                levels = torch.randint(self.acceleration_levels.numel(), (ids.numel(),), device=self.device)
            self.motion_step[ids, 0] = self.acceleration_levels[levels] * self.control_dt
        if self.auto_commands:
            if self.direct_velocity_commands:
                self.direct_episode[ids] = (
                    torch.rand(ids.numel(), device=self.device) < self.direct_velocity_probability
                )
            else:
                self.direct_episode[ids] = False
            if self.bar_speed is not None:
                level = torch.randint(self.bar_speed_levels.numel(), (ids.numel(),), device=self.device)
                self.bar_speed[ids] = self.bar_speed_levels[level]
            if self.stationary_episode_fraction:
                stationary = torch.rand(ids.numel(), device=self.device) < self.stationary_episode_fraction
                self.stationary_episode[ids] = stationary & (self.surface[ids] == 0)
            if self.speed_maneuvers and not self.height_maneuver_sequence.numel():
                self._sample_maneuvers(ids)
            self._sample_motion_goals(ids)
            self._sample_height_goals(ids)
            if air_drop_rows is not None and air_drop_rows.numel():
                brake_prob = self.events_cfg["air_drop_brake_probability"]
                do_brake = torch.rand(air_drop_rows.numel(), device=self.device) < brake_prob
                if do_brake.any():
                    b_rows = air_drop_rows[do_brake]
                    b_ids = ids[b_rows]
                    b_vel = air_drop_vel[do_brake]
                    self.commands[b_ids, 0] = b_vel
                    self.goals[b_ids, 0] = 0.0
                    self.goals[b_ids, 1] = 0.0
                    self.direct_episode[b_ids] = False
                    self.command_time[b_ids] = 2.6
                    if self.scut_reward.braking_cost is not None:
                        self.scut_reward.braking_cost.previous_forward[b_ids] = b_vel
                        self.scut_reward.braking_cost.reference_forward[b_ids] = b_vel

    def _sample_maneuvers(self, ids):
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

    def _sample_recovery_starts(self, ids, qpos, stance_height):
        self.recovery_steps[ids] = -1
        self.recovery_stable_steps[ids] = 0
        self.recovery_succeeded[ids] = False
        self.recovery_failed[ids] = False
        if not self.auto_commands:
            return
        selected = (self.surface[ids] == 0) & (torch.rand(ids.numel(), device=self.device) < self.recovery_fraction)
        reset_rows = selected.nonzero(as_tuple=True)[0]
        choices = torch.randint(len(self.recovery_poses), (reset_rows.numel(),), device=self.device)
        poses = self.recovery_poses[choices]
        # Reuse the lane position and terrain elevation. The wheels begin clear of the flat surface.
        height = self.pose_indices[0]
        qpos[reset_rows, height] += poses[:, 0] + 0.07 - stance_height[reset_rows]
        qpos[reset_rows[:, None], self.pose_indices[1:]] = poses[:, 1:]
        self.recovery_steps[ids[reset_rows]] = 0

    def begin_demo(self, height, surface, reverse=False, variant=0, *, free_drive=False):
        self.auto_commands = False
        self.free_drive = free_drive
        self.demo_initial_height = height
        self.demo_surface = self.lanes.surfaces.index(surface) if self.lanes is not None else None
        self.demo_reverse = reverse
        self.demo_variant = variant
        self.reset()

    def set_commands(self, forward, yaw, height, *, slew=True):
        self.goals[:] = self.commands.new_tensor((forward, yaw, height))
        lin_vel, ang_vel, _, _ = super()._base_state()
        self._advance_commands(lin_vel, ang_vel, slew=slew)

    def _advance_commands(self, lin_vel, ang_vel, *, slew=True):
        targets = self.goals.clamp(min=self.command_lower, max=self.command_upper)
        if self.heading_hold_enabled:
            requested_yaw = targets[:, 1]
            held_yaw = self._heading_yaw_target(requested_yaw, ang_vel[:, 2]).clamp(
                min=self.command_lower[1], max=self.command_upper[1]
            )
            bypass_heading = None
            if self.special_command_sampler is not None and self.auto_commands:
                bypass_heading = self.special_command_sampler.is_special_active()
            if getattr(self, "terrain_commands_enabled", False) and self.auto_commands:
                spin_active = self.terrain_spinning
                bypass_heading = spin_active if bypass_heading is None else (bypass_heading | spin_active)
            if bypass_heading is not None and bypass_heading.any():
                yaw = self._yaw_from_quat(self.robot.get_quat())
                on_flat = (self.surface == 0) if hasattr(self, "surface") else torch.ones_like(bypass_heading)
                self.heading_target.copy_(torch.where(bypass_heading & on_flat, yaw, self.heading_target))
                self.heading_turning.copy_(torch.where(bypass_heading, torch.ones_like(bypass_heading), self.heading_turning))
                held_yaw = torch.where(bypass_heading, requested_yaw, held_yaw)
            targets[:, 1] = held_yaw
        if self.recovery_fraction:
            targets[:, :2].masked_fill_(self.recovery_steps[:, None] >= 0, 0.0)
        vx, yaw = targets[:, :2].unbind(1)
        demand = torch.sqrt((vx / MAX_FORWARD_SPEED).square() + (yaw / YAW_REFERENCE).square())
        # The chassis reduces combined demand as turning unloads the inside wheel.
        lateral_accel = (lin_vel[:, 0] * ang_vel[:, 2]).abs()
        current_height = self.measured_heights.mean(dim=1).clamp(0.10, 0.335)
        radius = (1.0 - lateral_accel * current_height / (9.81 * self.half_track)).clamp(0.05, 1.0)
        outer_wheel = vx.abs() + self.half_track * yaw.abs()
        scale = torch.minimum(radius / demand.clamp_min(1e-6),
                              self.wheel_target_limit * self.wheel_radius / outer_wheel.clamp_min(1e-6))
        desired = targets[:, :2] * scale.clamp_max(1.0)[:, None]
        direct_ep = getattr(self, "direct_episode", None)
        if not slew or (self.auto_commands and self.direct_velocity_commands and (direct_ep is None or direct_ep.all())):
            self.commands[:, :2] = desired
        else:
            motion_delta = desired - self.commands[:, :2]
            step_limit = self.motion_step
            if self.manual_brake_acceleration is not None:
                current_vx = self.commands[:, 0]
                desired_vx = desired[:, 0]
                braking = (current_vx.abs() > 1e-6) & (
                    (desired_vx.abs() < current_vx.abs()) | ((desired_vx * current_vx) < 0.0)
                )
                if braking.any():
                    step_limit = step_limit.clone()
                    step_limit[:, 0] = torch.where(
                        braking, self.manual_brake_acceleration * self.control_dt, step_limit[:, 0]
                    )
            if not self.auto_commands:
                # Let manual steering slew independently of the forward ramp while enforcing the wheel-speed bound.
                next_motion = self.commands[:, :2] + motion_delta.clamp(min=-step_limit, max=step_limit)
                next_outer = next_motion[:, 0].abs() + self.half_track * next_motion[:, 1].abs()
                wheel_scale = (self.wheel_target_limit * self.wheel_radius / next_outer.clamp_min(1e-6)).clamp_max(1.0)
                self.commands[:, :2] = next_motion * wheel_scale[:, None]
            else:
                # Move along one segment so the wheel-speed bound holds during reversals too.
                steps_needed = (motion_delta.abs() / step_limit).amax(dim=1).clamp_min(1.0)
                slewed = self.commands[:, :2] + motion_delta / steps_needed[:, None]
                if self.direct_velocity_commands and direct_ep is not None:
                    self.commands[:, :2] = torch.where(direct_ep[:, None], desired, slewed)
                else:
                    self.commands[:, :2] = slewed
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

    def _heading_yaw_target(self, requested_rate, yaw_rate):
        yaw = self._yaw_from_quat(self.robot.get_quat())
        turning = requested_rate.abs() > 1e-6
        # After a turn ends, let the slewed yaw command and measured yaw rate settle before latching heading.
        settling = self.heading_turning & ((self.commands[:, 1].abs() > 0.05) | (yaw_rate.abs() > 0.10))
        on_corridor = (
            getattr(self, "terrain_commands_enabled", False)
            and getattr(self, "auto_commands", False)
            and hasattr(self, "surface")
        )
        latch_mask = (turning | self.heading_turning) & ((self.surface == 0) if on_corridor else True)
        self.heading_target.copy_(torch.where(latch_mask, yaw, self.heading_target))
        self.heading_turning.copy_(turning | settling)
        error = self.heading_target - yaw
        error = torch.atan2(error.sin(), error.cos())
        correction = self.heading_gain * error.sign() * (error.abs() - self.heading_deadband).clamp_min(0.0)
        if on_corridor:
            max_rate = torch.where(
                self.surface != 0,
                requested_rate.new_tensor(max(self.heading_max_yaw_rate, 1.4)),
                requested_rate.new_tensor(self.heading_max_yaw_rate),
            )
            clamped = correction.clamp(-max_rate, max_rate)
        else:
            clamped = correction.clamp(-self.heading_max_yaw_rate, self.heading_max_yaw_rate)
        hold = torch.where(settling, torch.zeros_like(requested_rate), clamped)
        return torch.where(turning, requested_rate, hold)

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
        joint_error = self.robot.get_dofs_position(self.leg_dofs) - nominal
        # The driven leg hinges can accumulate whole turns during recovery.
        joint_pos = (joint_error - tau * torch.round(joint_error / tau)) * self.leg_direction
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
        observations = {"policy": policy, "critic": critic}
        if self.actor_true_forward_speed:
            observations["body_forward_speed"] = lin_vel[:, :1].contiguous()
        if self.policy_history is not None:
            if self.history_needs_fill.any():
                fill = self.history_needs_fill
                self.policy_history[fill] = policy[fill, None, :]
                self.history_needs_fill[fill] = False
            self.last_policy_obs.copy_(policy)
            observations["policy_history"] = self.policy_history.reshape(self.num_envs, -1).clone()
        return TensorDict(observations, batch_size=[self.num_envs])

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
        # The position servo needs the equivalent target nearest the current joint angle.
        current = self.robot.get_dofs_position(self.leg_dofs)
        targets += tau * torch.round((current - targets) / tau)
        self.robot.control_dofs_position(targets, self.leg_dofs)
        self.robot.control_dofs_position(self.gimbal_target, self.gimbal_dofs)

    def _apply_controls(self, actions):
        super()._apply_controls(actions)
        if not self.recovery_fraction:
            return actions
        unpowered = (self.recovery_steps >= 0) & (self.recovery_steps < self.recovery_unpowered_steps)
        ids = unpowered.nonzero(as_tuple=True)[0]
        if ids.numel():
            self.robot.control_dofs_force(self.recovery_zero_force[ids], self.action_dofs, envs_idx=ids)
        return actions.masked_fill(unpowered[:, None], 0.0)

    def _episode_timeout(self):
        if self.free_drive:
            return torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        timed_out = super()._episode_timeout()
        if self.recovery_fraction:
            timed_out &= self.recovery_steps < 0
        return timed_out

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

    def _terrain_out_of_bounds(self, pos, surface_z):
        x, y = pos[:, 0], pos[:, 1]
        if self.free_drive:
            margin = 0.30
            outside = (x.abs() >= self.lanes.half_length - margin) | (y <= margin) | (y >= self.lanes.width - margin)
        else:
            travel_limit = self.lanes.travel_limit
            if isinstance(self.lanes, ScutGroundLanes):
                travel_limit = torch.where(self.surface == 0, self.lanes.flat_travel_limit, travel_limit)
            outside = (x.abs() >= travel_limit) | ((y - self.spawn_y).abs() >= self.lanes.lateral_limit)
        return outside | ~torch.isfinite(surface_z)

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
            out_of_bounds = self._terrain_out_of_bounds(pos, surface_z)

        if self.ground is not None:
            clearance_for_reward = self.scut_reward.measure_clearance(clearance)
        else:
            clearance_for_reward = clearance
        self.body_clearance.copy_(clearance_for_reward)
        height_target = self.commands[:, 2] + self.body_height_offset
        self.height_errors += (self.body_clearance - height_target).abs()
        self.leg_height_errors += (heights.mean(dim=1) - self.commands[:, 2]).abs()
        if self.forward_drop:
            self._update_drop_landing(pos, contact)
        not_recovering = None if not self.recovery_fraction else (self.recovery_steps < 0)
        pitch_fallen = self._pitch_fallen(gravity, active=not_recovering)
        fallen = body_contact | (clearance < 0.045) | (gravity[:, 2] > -0.65) | pitch_fallen
        if self.recovery_fraction:
            fallen = self._check_recovery(fallen, out_of_bounds, gravity, contact, height_target)
        return fallen, out_of_bounds

    def _check_recovery(self, fallen, out_of_bounds, gravity, contact, height_target):
        recovering = self.recovery_steps >= 0
        self.recovery_steps += recovering
        wheels_loaded = (contact[:, self.wheel_links, 2] > 5.0).all(dim=1)
        stable = (recovering & (self.recovery_steps > self.recovery_unpowered_steps) & ~fallen
                  & (gravity[:, 2] < -cos(radians(10.0))) & wheels_loaded
                  & ((self.body_clearance - height_target).abs() < 0.04)
                  & (self.robot.get_ang().norm(dim=1) < 0.8))
        self.recovery_stable_steps.copy_(torch.where(stable, self.recovery_stable_steps + 1, 0))
        self.recovery_succeeded.copy_(self.recovery_stable_steps >= self.recovery_hold_steps)
        self.recovery_failed.copy_(recovering & ~self.recovery_succeeded
                                   & ((self.recovery_steps >= self.recovery_deadline_steps) | out_of_bounds))
        self.recovery_steps.masked_fill_(self.recovery_succeeded | self.recovery_failed, -1)
        return torch.where(recovering, self.recovery_failed, fallen)

    def _update_drop_landing(self, pos, contact):
        forces = contact[:, self.wheel_links, 2].clamp_min(0.0)
        left, right = forces.unbind(dim=1)
        crossed_edge = pos[:, 0].abs() > self.lanes.deck_half_length + 0.04
        touchdown = (self.deck_height > 0.0) & crossed_edge & ~self.drop_touched & ((left + right) > 5.0)
        self.drop_touched |= touchdown
        self.landing_elapsed.add_(self.control_dt).clamp_max_(self.landing_seconds)
        self.landing_elapsed.masked_fill_(touchdown, 0.0)

    def _after_step(self, extras, done, lin_vel, ang_vel):
        if self.policy_history is not None:
            if self.policy_history.shape[1] > 1:
                self.policy_history[:, 1:] = self.policy_history[:, :-1].clone()
            self.policy_history[:, 0] = self.last_policy_obs
        if self.critic_scan_offsets is not None:
            self.critic_wheel_contacts.copy_(extras["wheel_contacts"])
        if self.training_rollout_steps:
            self.training_step_counter += 1
            iteration, remainder = divmod(self.training_step_counter, self.training_rollout_steps)
            if not remainder:
                self.training_iteration = iteration
                if self.height_lowering_schedule:
                    self._set_height_curriculum_iteration(iteration)
                if self.special_command_sampler is not None:
                    self.special_command_sampler.set_training_iteration(iteration)
        elif self.height_curriculum_rollout:
            self.height_curriculum_step += 1
            iteration, remainder = divmod(self.height_curriculum_step, self.height_curriculum_rollout)
            if not remainder:
                self._set_height_curriculum_iteration(iteration)
        self.episode_steps += 1
        if self.transition_course:
            torch.maximum(self.transition_peak_command, self.commands[:, 0].abs(), out=self.transition_peak_command)
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
                "leg_height_mae_cm": 100.0 * self.leg_height_errors[finished] / count,
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
                if getattr(self, "terrain_curriculum_enabled", False) and self.auto_commands:
                    obs_mask = self.surface[finished] != 0
                    if obs_mask.any():
                        f_obs = finished[obs_mask]
                        dist = self.traversal_distance[f_obs]
                        fell = extras["fallen"][f_obs]
                        promote = (~fell) & (dist >= self.curriculum_promote_distance)
                        demote = fell & (dist < self.curriculum_demote_distance)
                        cur_lvl = self.terrain_level[f_obs]
                        top_lvl = self.lanes.variants - 1
                        newly_maxed = promote & (cur_lvl >= top_lvl)
                        self.terrain_maxed[f_obs] |= newly_maxed
                        next_lvl = torch.where(promote, (cur_lvl + 1).clamp_max(top_lvl), cur_lvl)
                        next_lvl = torch.where(
                            demote & ~self.terrain_maxed[f_obs], (next_lvl - 1).clamp_min(0), next_lvl
                        )
                        self.terrain_level[f_obs] = next_lvl
            if self.transition_course:
                transitioned = finished[self.surface[finished] != 0]
                if transitioned.numel():
                    extras["log"]["Transition/peak_command_m_s"] = self.transition_peak_command[transitioned]
                    extras["log"]["Transition/fall_rate"] = extras["fallen"][transitioned].float()
        if self.recovery_fraction:
            resolved = self.recovery_succeeded | self.recovery_failed
            self.recovery_counts += torch.stack((resolved.sum(), self.recovery_succeeded.sum()))
            extras["recovery_succeeded"] = self.recovery_succeeded.clone()
            extras["recovery_failed"] = self.recovery_failed.clone()
        if self.auto_commands:
            if getattr(self, "events_enabled", False):
                self.push_time -= self.control_dt
                self.wrench_time -= self.control_dt
                active_env = ~done
                if self.recovery_fraction:
                    active_env = active_env & (self.recovery_steps < 0)
                due_push = ((self.push_time <= 0.0) & active_env).nonzero(as_tuple=True)[0]
                if due_push.numel():
                    ev = self.events_cfg
                    plow, phigh = ev["push_interval_seconds"]
                    self.push_time[due_push] = plow + torch.rand(due_push.numel(), device=self.device) * (phigh - plow)
                    bvel = self.robot.get_dofs_velocity(self.base_dofs, envs_idx=due_push)
                    dv_xy = (2.0 * torch.rand((due_push.numel(), 2), device=self.device) - 1.0) * ev["push_vel_xy"]
                    dw_xy = (2.0 * torch.rand((due_push.numel(), 2), device=self.device) - 1.0) * ev["push_ang_vel_xy"]
                    bvel[:, :2] += dv_xy
                    bvel[:, 3:5] += dw_xy
                    self.robot.set_dofs_velocity(bvel, self.base_dofs, envs_idx=due_push)
                due_wrench = ((self.wrench_time <= 0.0) & ~done).nonzero(as_tuple=True)[0]
                if due_wrench.numel():
                    self._sample_event_wrenches(due_wrench)
            self.command_time -= self.control_dt
            if self.transition_course:
                phase = self.transition_stop_phase
                x = self.robot.get_pos()[:, 0]
                along = torch.where(self.traversal_reverse, -x, x)
                start = (phase == 1) & (along >= self.transition_stop_at)
                stopping = phase == 2
                if self.transition_stop_feedback:
                    stable = (stopping & ~done & (self.commands[:, 0].abs() < 0.025)
                              & (lin_vel[:, :2].norm(dim=1) <= 0.08) & extras["wheel_contacts"].all(dim=1)
                              & (pitch <= 8.0) & (self.posture[:, 0] <= 8.0))
                    self.transition_still_time.copy_(torch.where(stable, self.transition_still_time + self.control_dt, 0))
                    self.transition_stop_elapsed += self.control_dt * stopping
                    restart = stopping & (self.transition_still_time >= 0.24 - 1e-6)
                    timeout = stopping & (self.transition_stop_elapsed >= 4.5) & ~restart
                    # A failed stop stays at zero for the rest of the episode.
                    phase.masked_fill_(timeout, 4)
                else:
                    self.transition_still_time += self.control_dt * (stopping & (self.commands[:, 0].abs() < 0.025))
                    restart = stopping & (self.transition_still_time >= 0.9)
                phase.masked_fill_(start, 2)
                phase.masked_fill_(restart, 3)
                self.command_time.masked_fill_(start | restart, 0.0)
            due_motion = (self.command_time <= 0).nonzero(as_tuple=True)[0]
            self._sample_motion_goals(due_motion)
            due_height = due_motion
            if self.speed_maneuvers or self.special_command_sampler is not None:
                self.height_time -= self.control_dt
                due_height = (self.height_time <= 0).nonzero(as_tuple=True)[0]
            self._sample_height_goals(due_height)
            self._advance_commands(lin_vel, ang_vel)
