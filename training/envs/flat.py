"""Batched Genesis stepping and flat-ground locomotion task."""

from abc import abstractmethod

import genesis as gs
from genesis.utils.geom import inv_transform_by_quat
from rsl_rl.env import VecEnv
from tensordict import TensorDict
import torch

from configs.experiment import load_config


class FlatEnv(VecEnv):
    """Four leg position targets and two wheel velocity targets per robot."""

    num_actions = 6
    task_name = "flat"
    wheel_target_limit = 100.0
    leg_signs = (1.0, 1.0, 1.0, 1.0)
    wheel_signs = (1.0, 1.0)

    def __init__(self, model_path, num_envs, episode_seconds=None, viewer=False, *, task_config=None,
                 training_view_fps=None):
        self.on_step = None
        self.step_visualizer = training_view_fps is None
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
            vis_options=gs.options.VisOptions(rendered_envs_idx=None if self.step_visualizer else [0]),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.3, -1.5, 0.8), camera_lookat=(0, 0, self.height_target),
                refresh_rate=training_view_fps or 60,
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
        self.last_actions = torch.zeros((num_envs, self.num_actions), device=self.device)
        self.world_gravity = torch.tensor((0.0, 0.0, -1.0), device=self.device).expand(num_envs, -1).contiguous()
        self.all_envs = torch.arange(num_envs, device=self.device)
        self._initialize_task()
        self.reset()

    def _add_ground(self, material, viewer):
        surface = gs.surfaces.Default(color=(0.45, 0.55, 0.63), roughness=0.95) if viewer else None
        self.scene.add_entity(gs.morphs.Plane(visualization=viewer), material=material, surface=surface)

    def terrain_description(self):
        return "平地"

    def _initialize_task(self):
        pass

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

    def _reset_envs(self, ids):
        self.robot.set_qpos(self.home_qpos[ids], envs_idx=ids, zero_velocity=True)
        self.episode_length_buf[ids] = 0
        self.last_actions[ids] = 0.0
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

    def _height_and_termination(self, height, gravity, contact):
        height_error = (height - self.commands[:, 2]).square()
        body_contact = contact[:, self.body_links].norm(dim=-1).amax(dim=1) > 10.0
        fallen = body_contact | (height < self.min_height) | (gravity[:, 2] > -0.65)
        height_reward = torch.exp(-height_error / 0.025) - 100.0 * height_error
        return height_reward, fallen, False

    def _task_cost(self, lin_vel, ang_vel, actions):
        return 0.0

    def _after_step(self, extras, done, lin_vel, ang_vel):
        pass

    def step(self, actions):
        if actions.shape != self.last_actions.shape:
            raise ValueError(f"Expected actions shaped {tuple(self.last_actions.shape)}, got {tuple(actions.shape)}")
        actions = actions.clamp(-10.0, 10.0)
        self._drive(actions)
        wheel_target = (actions[:, 4:] * 10.0).clamp(-self.wheel_target_limit, self.wheel_target_limit)
        self.robot.control_dofs_velocity(wheel_target * self.wheel_direction, self.wheel_dofs)
        for _ in range(self.decimation):
            self._apply_springs()
            self.scene.step(update_visualizer=self.step_visualizer)

        self.episode_length_buf += 1
        lin_vel, ang_vel, gravity, height = self._base_state()
        contact = self.robot.get_links_net_contact_force()
        normal_forces = contact[:, :, 2].clamp_min(0.0)
        wheel_forces = normal_forces[:, self.wheel_links]
        normal_load = normal_forces.sum(dim=1)
        wheel_load = wheel_forces.sum(dim=1)
        wheel_support = wheel_load / normal_load.clamp_min(self.min_support_load)
        other_support = (normal_load - wheel_load) / normal_load.clamp_min(self.min_support_load)
        height_reward, fallen, out_of_bounds = self._height_and_termination(height, gravity, contact)
        time_outs = ((self.episode_length_buf >= self.max_episode_length) | out_of_bounds) & ~fallen
        done = fallen | time_outs
        speed_error = (lin_vel[:, 0] - self.commands[:, 0]).square() + lin_vel[:, 1].square()
        yaw_error = (ang_vel[:, 2] - self.commands[:, 1]).square()
        tilt = gravity[:, :2].square()
        action_rate = (actions - self.last_actions).square().sum(dim=1)
        weights = self.cfg["reward"]
        reward = self.control_dt * (
            weights["speed"] * torch.exp(-speed_error / 0.5)
            + (2.0 * weights["speed"]) * torch.exp(-speed_error / 0.1) - weights["speed"] * speed_error
            + weights["yaw"] * torch.exp(-yaw_error / 0.25) - (0.25 * weights["yaw"]) * yaw_error
            + weights["upright"] * torch.exp(-tilt[:, 0] / 0.02) + weights["upright"] * torch.exp(-tilt[:, 1] / 0.01)
            - (2.0 * weights["upright"]) * tilt.sum(dim=1)
            + weights["height"] * height_reward
            - self._task_cost(lin_vel, ang_vel, actions)
            + (2.0 * weights["wheel_support"]) * wheel_support - (2.0 * weights["wheel_support"]) * other_support
            - (0.5 * weights["vertical_velocity"]) * lin_vel[:, 2].square()
            - (0.05 * weights["angular_velocity"]) * ang_vel[:, :2].square().sum(dim=1)
            - (0.01 * weights["action_rate"]) * action_rate - (200.0 * weights["fall"]) * fallen
        )
        extras = {
            "time_outs": time_outs,
            "fallen": fallen,
            "wheel_contacts": wheel_forces > 1.0,
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
