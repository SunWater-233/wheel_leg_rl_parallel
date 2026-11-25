"""Batched Genesis stepping and flat-ground locomotion task."""

from abc import abstractmethod
from math import radians

import genesis as gs
from genesis.utils.geom import inv_transform_by_quat
from rsl_rl.env import VecEnv
from tensordict import TensorDict
import torch

from configs.experiment import load_config
from robots.scut_rough_reward import ScutRoughReward


class FlatEnv(VecEnv):
    """Four leg position targets and two wheel velocity targets per robot."""

    num_actions = 6
    task_name = "flat"
    wheel_target_limit = 100.0
    wheel_target_delay_steps = 0
    leg_signs = (1.0, 1.0, 1.0, 1.0)
    wheel_signs = (1.0, 1.0)

    def __init__(self, model_path, num_envs, episode_seconds=None, viewer=False, *, task_config=None,
                 training_view_fps=None, keyboard_view=False, viewer_fps=None):
        self.on_step = None
        self.step_visualizer = training_view_fps is None and not keyboard_view
        self.device = gs.device
        self.num_envs = num_envs
        self.control_dt = self.physics_dt * self.decimation
        task = load_config()["task"] if task_config is None else task_config
        if episode_seconds is not None:
            task = {**task, "episode_seconds": episode_seconds}
        self.max_episode_length = max(1, round(task["episode_seconds"] / self.control_dt))
        self.cfg = {**task, "control_dt": self.control_dt}
        batched_dynamics = task.get("domain_randomization", {}).get("enabled", False)
        self.scene = gs.Scene(
            show_viewer=viewer,
            sim_options=gs.options.SimOptions(dt=self.physics_dt, substeps=1),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False, batch_links_info=batched_dynamics,
                                                  batch_dofs_info=batched_dynamics),
            vis_options=gs.options.VisOptions(rendered_envs_idx=None if self.step_visualizer else [0],
                                              shadow=not keyboard_view),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.3, -1.5, 0.8), camera_lookat=(0, 0, self.height_target),
                res=(800, 600) if keyboard_view else None,
                refresh_rate=viewer_fps or training_view_fps or (30 if keyboard_view else 60),
            ),
        )
        material = gs.materials.Rigid(friction=task["friction"])
        self._add_ground(material, viewer)
        self.robot = self.scene.add_entity(self._robot_morph(model_path, viewer), material=material)
        if self.robot.n_equalities != self.expected_equalities:
            raise ValueError(f"Expected {self.expected_equalities} equality constraints, found {self.robot.n_equalities}")
        self.scene.build(n_envs=num_envs, env_spacing=(1.5, 1.5), center_envs_at_origin=self.step_visualizer)
        self.leg_dofs = self._dofs(self.leg_joints)
        self.wheel_dofs = self._dofs(self.wheel_joints)
        self.wheels = [self.robot.get_joint(name) for name in self.wheel_joints]
        self.spring_dofs = self._dofs(self.spring_joints)
        self.action_dofs = self.leg_dofs + self.wheel_dofs
        self.body_links = [self.robot.get_link(name).idx_local for name in self.body_contacts]
        self.wheel_links = [self.robot.get_link(name).idx_local for name in self.wheel_bodies]
        self._set_drives()

        self.home_qpos = self.robot.get_qpos().clone()
        self._set_home_pose()
        self.home_leg_pos = self.home_qpos[:, [self.robot.get_joint(name).qs_idx_local[0] for name in self.leg_joints]]
        self.leg_direction = self.home_qpos.new_tensor(self.leg_signs)
        self.wheel_direction = self.home_qpos.new_tensor(self.wheel_signs)
        self.action_direction = torch.cat((self.leg_direction, self.wheel_direction))
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.commands = torch.zeros((num_envs, 3), device=self.device)
        self.body_clearance = self.commands.new_zeros(num_envs)
        self.body_height_offset = 0.0
        self.last_actions = torch.zeros((num_envs, self.num_actions), device=self.device)
        # Demo drivers may clear this to keep fallen robots in the scene and stand them back up.
        self.reset_on_fall = True
        termination = self.cfg["termination"]
        self.instant_pitch_rad = radians(termination["instant_pitch_deg"])
        self.sustained_pitch_rad = radians(termination["sustained_pitch_deg"])
        self.sustained_pitch_seconds = termination["sustained_pitch_seconds"]
        self.pitch_exceed_time = self.commands.new_zeros(num_envs)
        self.world_gravity = torch.tensor((0.0, 0.0, -1.0), device=self.device).expand(num_envs, -1).contiguous()
        self.all_envs = torch.arange(num_envs, device=self.device)
        self._initialize_task()
        self.reset()

    def _add_ground(self, material, viewer):
        surface = gs.surfaces.Default(color=(0.45, 0.55, 0.63), roughness=0.95) if viewer else None
        self.plane_ground = self.scene.add_entity(gs.morphs.Plane(visualization=viewer), material=material, surface=surface)

    def terrain_description(self):
        return "平地"

    def _initialize_task(self):
        self.scut_reward = ScutRoughReward(self)

    @abstractmethod
    def _robot_morph(self, model_path, viewer):
        """Return the Genesis robot description for this environment."""

    def _set_home_pose(self):
        pass

    @abstractmethod
    def scripted_wheel_speed(self, velocity, angular_velocity, gravity, speed_integral):
        """Return the baseline controller's wheel speed in radians per second."""

    def _dofs(self, names):
        return [self.robot.get_joint(name).dofs_idx_local[0] for name in names]

    @abstractmethod
    def _set_drives(self):
        """Configure the robot's actuated and passive joints."""

    @abstractmethod
    def _drive(self, actions):
        """Apply the leg targets and hold auxiliary joints in place."""

    @abstractmethod
    def _apply_springs(self):
        """Update spring forces before a physics step."""

    def _apply_external_wrench(self):
        """Optional external generalized wrench hook before each physics substep."""

    def _reset_envs(self, ids):
        self.robot.set_qpos(self.home_qpos[ids], envs_idx=ids, zero_velocity=True)
        self.episode_length_buf[ids] = 0
        self.last_actions[ids] = 0.0
        self.pitch_exceed_time[ids] = 0.0
        self.scut_reward.reset(ids)
        for axis, key in enumerate(("forward_command", "yaw_command")):
            low, high = self.cfg[key]
            self.commands[ids, axis] = torch.rand(ids.numel(), device=self.device) * (high - low) + low
        self.commands[ids, 2] = self.height_target

    def reset(self, env_ids=None):
        self._reset_envs(self.all_envs if env_ids is None else env_ids)
        return self.get_observations()

    def _base_state(self):
        quat = self.robot.get_quat().contiguous()
        lin_vel = inv_transform_by_quat(self.robot.get_vel(), quat)
        ang_vel = inv_transform_by_quat(self.robot.get_ang(), quat)
        gravity = inv_transform_by_quat(self.world_gravity, quat)
        height = self.robot.get_pos()[:, 2]
        return lin_vel, ang_vel, gravity, height

    def get_observations(self):
        lin_vel, ang_vel, gravity, height = self._base_state()
        joint_pos = (self.robot.get_dofs_position(self.leg_dofs) - self.home_leg_pos) * self.leg_direction
        joint_vel = self.robot.get_dofs_velocity(self.action_dofs) * self.action_direction
        command_scale = self.commands.new_tensor((1.0, 1.0, 5.0))
        policy = torch.cat((self.commands * command_scale, ang_vel, gravity, joint_pos,
                            joint_vel * 0.1, self.last_actions, height[:, None] * 5.0), dim=-1)
        critic = torch.cat((policy, lin_vel), dim=-1)
        return TensorDict({"policy": policy, "critic": critic}, batch_size=[self.num_envs])

    def _pitch_fallen(self, gravity, *, active=None):
        pitch_abs = torch.atan2(gravity[:, 0], gravity[:, 1:].norm(dim=1)).abs()
        exceeded = pitch_abs > self.sustained_pitch_rad
        if active is not None:
            exceeded = exceeded & active
        self.pitch_exceed_time.copy_(torch.where(exceeded, self.pitch_exceed_time + self.control_dt, 0.0))
        sustained = (self.sustained_pitch_seconds > 0.0) & (
            self.pitch_exceed_time >= self.sustained_pitch_seconds - 1e-6
        )
        instant = pitch_abs > self.instant_pitch_rad
        if active is not None:
            instant = instant & active
        return instant | sustained

    def _termination(self, height, gravity, contact):
        self.body_clearance.copy_(height)
        body_contact = contact[:, self.body_links].norm(dim=-1).amax(dim=1) > 10.0
        pitch_fallen = self._pitch_fallen(gravity)
        fallen = body_contact | (height < self.min_height) | (gravity[:, 2] > -0.65) | pitch_fallen
        return fallen, False

    def _after_step(self, extras, done, lin_vel, ang_vel):
        pass

    def _apply_controls(self, actions):
        self._drive(actions)
        wheel_target = (actions[:, 4:] * 10.0).clamp(-self.wheel_target_limit, self.wheel_target_limit)
        self.robot.control_dofs_velocity(wheel_target * self.wheel_direction, self.wheel_dofs)
        return actions

    def _episode_timeout(self):
        return self.episode_length_buf >= self.max_episode_length

    def step(self, actions):
        if actions.shape != self.last_actions.shape:
            raise ValueError(f"Expected actions shaped {tuple(self.last_actions.shape)}, got {tuple(actions.shape)}")
        actions = actions.clamp(-10.0, 10.0)
        delay = self.wheel_target_delay_steps
        if delay:
            # Update the legs immediately and hold the previous wheel targets until the delayed step.
            previous_wheels = torch.cat((actions[:, :4], self.last_actions[:, 4:]), dim=1)
            self._apply_controls(previous_wheels)
        for substep in range(self.decimation):
            if substep == delay:
                actions = self._apply_controls(actions)
            self._apply_springs()
            self._apply_external_wrench()
            self.scene.step(update_visualizer=self.step_visualizer)

        self.episode_length_buf += 1
        lin_vel, ang_vel, gravity, height = self._base_state()
        contact = self.robot.get_links_net_contact_force()
        fallen, out_of_bounds = self._termination(height, gravity, contact)
        time_outs = self._episode_timeout() | out_of_bounds
        if self.reset_on_fall:
            time_outs = time_outs & ~fallen
            done = fallen | time_outs
        else:
            # Falls are still reported in extras, but the episode keeps running (e.g. for self-righting).
            done = time_outs
        reward = self.control_dt * self.scut_reward(lin_vel, ang_vel, gravity, contact, fallen, actions)
        extras = {
            "time_outs": time_outs,
            "fallen": fallen,
            "wheel_contacts": contact[:, self.wheel_links, 2] > 1.0,
            "terminal_height": height,
        }
        self.last_actions.copy_(actions)
        self._after_step(extras, done, lin_vel, ang_vel)
        reset_ids = done.nonzero(as_tuple=True)[0]
        if reset_ids.numel():
            self._reset_envs(reset_ids)
        if self.on_step is not None:
            self.on_step(done[0])
        return self.get_observations(), reward, done, extras
