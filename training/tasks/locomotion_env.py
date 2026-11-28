"""V0.2 locomotion environment with continuous hip-to-wheel height commands.

Orchestrates modular task components:
- `tasks.terrains`: lane height-fields and drop decks
- `tasks.commands`: velocity/height sampling, heading hold, friction circle, maneuvers
- `tasks.randomization`: dynamics DR and disturbance events (push / wrench / reset jitter / air drop)
- `tasks.observations`: actor/critic observation assembly, noise, IMU delay, history stack
- `tasks.recovery`: low-stance touchdown recovery sub-task
- `tasks.metrics`: episode tracking scalars and terrain curriculum updates
"""

from itertools import product
from math import radians, tau

import genesis as gs
import torch

from configs.schema import MAX_FORWARD_ACCELERATION, MAX_FORWARD_SPEED
from robots.v02.adapter import WheelLegFlatEnv
from robots.v02.spec import HALF_TRACK, HOME_JOINTS, STANCE_SPAWN_GAP, WHEEL_TARGET_LIMIT
from robots.v02.stance import solve_stances
from tasks.commands.maneuvers import compute_maneuver_goals, sample_maneuvers
from tasks.commands.planner import (
    advance_commands,
    heading_yaw_target,
    sample_height_goals,
    sample_heights,
    sample_heights_for_envs,
    sample_motion_goals,
    step_commands,
    yaw_from_quat,
)
from tasks.commands.special_mode import SpecialModeUniformVelocityCommand
from tasks.commands.terrain_cmd import sample_terrain_command_goals
from tasks.metrics import init_episode_metrics, reset_episode_metrics, update_step_metrics_and_logs
from tasks.observations import (
    build_locomotion_observations,
    critic_terrain_observations,
    enable_observation_noise,
    leg_projections,
    shift_policy_history,
)
from tasks.randomization.dynamics import randomize_dynamics
from tasks.randomization.events import (
    apply_air_drop_braking,
    apply_external_wrench,
    apply_reset_events,
    euler_zyx_to_quat,
    init_events,
    perturb_stance_height,
    quat_mul,
    reset_event_timers,
    sample_event_wrenches,
    step_events,
)
from tasks.recovery import (
    apply_recovery_zero_force,
    check_recovery,
    init_low_recovery,
    mask_unpowered_actions,
    record_recovery_counts,
    sample_recovery_starts,
)
from tasks.rewards.scut_rough import ScutRoughReward
from tasks.terrains import (
    SCUT_GROUND_SINGLE_SIDE_SURFACES,
    TERRAIN_LANES,
    ScutGroundLanes,
    ScutV1Lanes,
    TransitionLanes,
)


class WheelLegLocomotionEnv(WheelLegFlatEnv):
    task_name = "locomotion"
    wheel_target_limit = WHEEL_TARGET_LIMIT
    half_track = HALF_TRACK
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
                box = gs.morphs.Box(
                    pos=center,
                    size=tuple(2 * extent for extent in half_size),
                    fixed=True,
                    visualization=viewer,
                )
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
        leg_delay_range = settings["leg_target_delay_steps"]
        self.leg_target_delay_range = tuple(leg_delay_range) if any(leg_delay_range) else None
        wheel_delay_range = settings["wheel_target_delay_steps_range"]
        self.wheel_target_delay_range = tuple(wheel_delay_range) if wheel_delay_range else None
        imu_delay_range = settings["imu_obs_delay_steps"]
        self.imu_obs_delay_range = tuple(imu_delay_range) if any(imu_delay_range) else None
        self.wheel_armature = settings["wheel_armature"]
        super()._set_drives()

    def _initialize_task(self):
        settings = self.cfg["locomotion"]
        randomization = self.cfg.get("domain_randomization", {})
        if randomization.get("enabled"):
            randomize_dynamics(self, randomization)

        self._init_height_and_poses(settings)
        self._init_command_state(settings)
        self._init_observation_state(settings)
        init_episode_metrics(self)
        self._init_terrain_state(settings)
        init_events(self, settings)
        self.scut_reward = ScutRoughReward(self, airborne_minimum_height=self.height_low + self.body_height_offset)

    def _init_height_and_poses(self, settings):
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
        init_low_recovery(self, settings)
        self.body_height_offset = self.pose_table[0, 0].item() - self.height_low - STANCE_SPAWN_GAP
        base = self.robot.get_joint("base_link").qs_idx_local
        self.base_xy = base[:2]
        self.base_quat = base[3:7]
        self.pose_indices = [base[2], *(self.robot.get_joint(name).qs_idx_local[0] for name in HOME_JOINTS)]
        self.leg_columns = [list(HOME_JOINTS).index(name) + 1 for name in self.leg_joints]
        self.hips = [self.robot.get_joint(f"hip_{side}") for side in "LR"]

    def _init_command_state(self, settings):
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

    def _init_observation_state(self, settings):
        self.observation_noise = None
        self.actor_true_forward_speed = settings["actor_true_forward_speed"]
        self.actor_history_steps = settings["actor_history_steps"]
        self.policy_obs_dim = 28
        if self.actor_history_steps > 1:
            self.policy_history = self.commands.new_zeros((
                self.num_envs, self.actor_history_steps - 1, self.policy_obs_dim,
            ))
            self.last_policy_obs = self.commands.new_zeros((self.num_envs, self.policy_obs_dim))
            self.history_needs_fill = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            self.policy_history = None
        self.critic_scan_offsets = None
        if settings["critic_terrain_observations"]:
            self.critic_scan_offsets = self.commands.new_tensor(self.critic_scan_distances)
            self.critic_wheel_contacts = torch.zeros_like(self.commands[:, :2], dtype=torch.bool)

    def _init_terrain_state(self, settings):
        self.spawn_y = torch.zeros_like(self.command_time)
        self.terrain_selection = torch.zeros(
            (2, self.num_envs), dtype=self.episode_length_buf.dtype, device=self.device
        )
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
            self.reverse_probabilities = (
                self.commands.new_tensor(self.lanes.reverse_probabilities)
                if isinstance(self.lanes, ScutGroundLanes) else None
            )
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

    # ---- Stance & curriculum ----

    def _poses(self, height):
        span = max(self.height_high - self.height_low, 1e-8)
        position = ((height - self.height_low) / span).clamp(0.0, 1.0) * (len(self.pose_table) - 1)
        lower = position.long().clamp_max(len(self.pose_table) - 2)
        return self.pose_table[lower].lerp(self.pose_table[lower + 1], (position - lower)[:, None])

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

    # ---- Command & math delegation (preserves method hooks used by tests and eval harnesses) ----

    _quat_mul = staticmethod(quat_mul)
    _euler_zyx_to_quat = staticmethod(euler_zyx_to_quat)
    _yaw_from_quat = staticmethod(yaw_from_quat)

    def _sample_heights(self, count):
        return sample_heights(self, count)

    def _sample_heights_for_envs(self, ids):
        return sample_heights_for_envs(self, ids)

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
        return sample_terrain_command_goals(self, t_ids)

    def _sample_motion_goals(self, ids):
        sample_motion_goals(self, ids)

    def _maneuver_goals(self, ids, motion, duration, hold_sample):
        return compute_maneuver_goals(self, ids, motion, duration, hold_sample)

    def _sample_maneuvers(self, ids):
        sample_maneuvers(self, ids)

    def _sample_height_goals(self, ids):
        sample_height_goals(self, ids)

    def _heading_yaw_target(self, requested_rate, yaw_rate):
        return heading_yaw_target(self, requested_rate, yaw_rate)

    def _advance_commands(self, lin_vel, ang_vel, *, slew=True):
        advance_commands(self, lin_vel, ang_vel, slew=slew)

    def _sample_event_wrenches(self, ids):
        sample_event_wrenches(self, ids)

    def _apply_external_wrench(self):
        apply_external_wrench(self)

    def _sample_recovery_starts(self, ids, qpos, stance_height):
        sample_recovery_starts(self, ids, qpos, stance_height)

    # ---- Reset & demo setup ----

    def _reset_envs(self, ids):
        qpos = self.home_qpos[ids].clone()
        ground_height = 0.0
        scut_course = isinstance(self.lanes, ScutV1Lanes)
        if self.transition_course and self.speed_levels.numel() and self.auto_commands:
            levels = torch.randint(self.speed_levels.numel(), (ids.numel(),), device=self.device)
            self.episode_speed_limit[ids] = self.speed_levels[levels]
        if self.lanes is not None:
            variant, jitter = self._select_reset_lanes(ids, scut_course)
            self.terrain_variant[ids] = variant
        height = (
            self._sample_heights_for_envs(ids) if self.auto_commands
            else self.commands.new_full((ids.numel(),), self.demo_initial_height)
        )
        if self.lanes is not None:
            ground_height, height = self._place_on_lanes(ids, qpos, variant, jitter, scut_course, height)
        if self.auto_commands and self.height_maneuver_sequence.numel():
            self._sample_maneuvers(ids)
            self.height_sequence_index[ids] = 0
            height_episode = (self.maneuver[ids] == self.height_mode) & (self.surface[ids] == 0)
            height = torch.where(height_episode, self.height_maneuver_sequence[0], height)

        stance = self._poses(perturb_stance_height(self, ids, height))
        qpos[:, self.pose_indices] = stance
        qpos[:, self.pose_indices[0]] += ground_height
        nominal_yaw = self._yaw_from_quat(qpos[:, self.base_quat])
        if self.recovery_fraction:
            self._sample_recovery_starts(ids, qpos, stance[:, 0])
        air_drop_rows, air_drop_vel, init_base_vel = apply_reset_events(self, ids, qpos)
        self.robot.set_qpos(qpos, envs_idx=ids, zero_velocity=True)
        if init_base_vel is not None and (init_base_vel != 0.0).any():
            self.robot.set_dofs_velocity(init_base_vel, self.base_dofs, envs_idx=ids)

        self._reset_delay_buffers(ids)
        if self.heading_hold_enabled:
            actual_yaw = self._yaw_from_quat(qpos[:, self.base_quat])
            if self.lanes is not None and self.auto_commands:
                self.heading_target[ids] = torch.where(self.surface[ids] == 0, actual_yaw, nominal_yaw)
            else:
                self.heading_target[ids] = actual_yaw
            self.heading_turning[ids] = False
        if getattr(self, "terrain_commands_enabled", False):
            self.terrain_spinning[ids] = False
        reset_event_timers(self, ids)
        if self.special_command_sampler is not None:
            self.special_command_sampler.reset(ids)
        if self.critic_scan_offsets is not None:
            self.critic_wheel_contacts[ids] = False
        if self.policy_history is not None:
            self.policy_history[ids] = 0.0
            self.last_policy_obs[ids] = 0.0
            self.history_needs_fill[ids] = True
        self.episode_length_buf[ids] = 0
        reset_episode_metrics(self, ids)
        if self.terrain_traversal:
            self.traversal_distance[ids] = 0.0
            if self.auto_commands and not scut_course:
                self.traversal_reverse[ids] = torch.rand(ids.numel(), device=self.device) < 0.5
        self.last_actions[ids] = 0.0
        self.pitch_exceed_time[ids] = 0.0
        self.scut_reward.reset(ids)
        self.height_rate[ids] = 0.0
        self.commands[ids] = 0.0
        self.commands[ids, 2] = height
        self.measured_heights[ids] = height[:, None]
        self.goals[ids] = self.commands[ids]
        self._reset_episode_commands(ids, air_drop_rows, air_drop_vel)

    def _select_reset_lanes(self, ids, scut_course):
        if self.demo_surface is not None:
            self.surface[ids] = self.demo_surface
            return self.demo_variant, 0.0
        self.surface[ids] = torch.multinomial(self.surface_weights, ids.numel(), replacement=True)
        if getattr(self, "terrain_curriculum_enabled", False):
            maxed = self.terrain_maxed[ids]
            sampled_max = torch.multinomial(self.curriculum_max_weights, ids.numel(), replacement=True)
            variant = torch.where(maxed, sampled_max, self.terrain_level[ids])
        elif self.surface_variant_weights is not None:
            variant = torch.multinomial(self.surface_variant_weights[self.surface[ids]], 1).squeeze(1)
        elif scut_course:
            variant = torch.multinomial(self.variant_weights, ids.numel(), replacement=True)
        else:
            variant = torch.randint(self.lanes.variants, (ids.numel(),), device=self.device)
        jitter = torch.rand(ids.numel(), device=self.device) * 0.12 - 0.06
        return variant, jitter

    def _place_on_lanes(self, ids, qpos, variant, jitter, scut_course, height):
        ground_height = 0.0
        if self.transition_course and self.auto_commands and self.transition_speed_levels.numel():
            levels = torch.randint(self.transition_speed_levels.numel(), (ids.numel(),), device=self.device)
            self.episode_speed_limit[ids] = torch.where(
                self.surface[ids] != 0, self.transition_speed_levels[levels], self.episode_speed_limit[ids]
            )
        lane = self.surface[ids] * self.lanes.variants + variant
        self.spawn_y[ids] = self.lane_centers[lane]
        qpos[:, self.base_xy[0]] = 0.0
        qpos[:, self.base_xy[1]] = self.spawn_y[ids] + jitter
        if scut_course:
            if self.auto_commands:
                reverse_probability = (
                    0.5 if self.reverse_probabilities is None else self.reverse_probabilities[self.surface[ids]]
                )
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
            focus = (
                (self.surface[ids] == self.rough_slope_surface)
                & (torch.rand(ids.numel(), device=self.device) < self.low_rough_slope_probability)
            )
            self.low_rough_slope[ids] = focus
            height = torch.where(focus, height.clamp_max(self.height_nominal), height)
        return ground_height, height

    def _reset_episode_commands(self, ids, air_drop_rows, air_drop_vel):
        if not self.transition_course and self.speed_levels.numel() and self.auto_commands:
            levels = torch.randint(self.speed_levels.numel(), (ids.numel(),), device=self.device)
            self.episode_speed_limit[ids] = self.speed_levels[levels]
        if self.transition_course:
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
            apply_air_drop_braking(self, ids, air_drop_rows, air_drop_vel)

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

    # ---- Observations, actuation, ground queries, termination, step hook ----

    def _leg_projections(self, gravity):
        return leg_projections(self, gravity)

    def leg_heights(self, gravity):
        heights, _ = self._leg_projections(gravity)
        return heights

    def enable_observation_noise(self):
        enable_observation_noise(self)

    def get_observations(self):
        lin_vel, ang_vel, gravity, _ = super()._base_state()
        return build_locomotion_observations(self, lin_vel, ang_vel, gravity)

    def _critic_terrain_observations(self):
        return critic_terrain_observations(self)

    def _drive(self, actions):
        nominal = self._poses(self.commands[:, 2])[:, self.leg_columns]
        offsets = (actions[:, :4] * self.leg_action_scale).clamp(-0.35, 0.35)
        targets = nominal + offsets * self.leg_direction + self.leg_bias
        # The position servo needs the equivalent target nearest the current joint angle.
        current = self.robot.get_dofs_position(self.leg_dofs)
        targets += tau * torch.round((current - targets) / tau)
        self.robot.control_dofs_position(targets, self.leg_dofs)
        self.robot.control_dofs_position(self.gimbal_target, self.gimbal_dofs)

    def _effective_recorded_actions(self, actions):
        return mask_unpowered_actions(self, actions)

    def _apply_controls(self, actions):
        super()._apply_controls(actions)
        return apply_recovery_zero_force(self, actions)

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
            on_deck = (
                (points[..., 0].abs() <= self.lanes.deck_half_length)
                & ((points[..., 1] - self.spawn_y[:, None]).abs() <= self.lanes.deck_half_width)
            )
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

        clearance_for_reward = self.scut_reward.measure_clearance(clearance) if self.ground is not None else clearance
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
            fallen = check_recovery(self, fallen, out_of_bounds, gravity, contact, height_target)
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
        shift_policy_history(self)
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
        pitch = update_step_metrics_and_logs(self, extras, done, lin_vel, ang_vel)
        record_recovery_counts(self, extras)
        if self.auto_commands:
            step_events(self, done)
            step_commands(self, done, lin_vel, ang_vel, pitch, extras)
