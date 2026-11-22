"""Genesis adapter for the SCUT Wheelbipe V14.2 reference robot."""

import genesis as gs
import torch

from configs.experiment import TRAINING_DIR
from envs.flat import FlatEnv


class WheelbipeFlatEnv(FlatEnv):
    model_path = TRAINING_DIR / "assets/wheelbipe_v14_2.xml"
    label = "Wheelbipe V14.2"
    physics_dt = 0.005
    decimation = 4
    height_target = 0.22
    min_height = 0.10
    wheel_radius = 0.06
    leg_action_scale = 0.5
    leg_joints = ("left_front1_joint", "right_front1_joint", "left_rear1_joint", "right_rear1_joint")
    wheel_joints = ("left_wheel_joint", "right_wheel_joint")
    spring_joints = ("left_spring2_joint", "right_spring2_joint")
    wheel_bodies = ("left_wheel_link", "right_wheel_link")
    body_contacts = ("base_link", "gimbal_yaw_link", "gimbal_pitch_link")
    expected_equalities = 6
    min_support_load = 60.0

    def _robot_morph(self, model_path, viewer):
        return gs.morphs.MJCF(file=str(model_path), pos=(0.0, 0.0, 0.23), visualization=viewer)

    def scripted_wheel_speed(self, velocity, angular_velocity, gravity, speed_integral):
        speed_error = self.commands[:, 0] - velocity[:, 0]
        surface_speed = (self.commands[:, 0] + 2.5 * gravity[:, 0] + 0.2 * angular_velocity[:, 1]
                         + 1.1 * speed_error + 2.0 * speed_integral)
        return surface_speed / self.wheel_radius

    def _set_drives(self):
        robot = self.robot
        robot.set_dofs_kp([60.0] * 4, self.leg_dofs)
        robot.set_dofs_kv([2.0] * 4, self.leg_dofs)
        robot.set_dofs_force_range([-40.0] * 4, [40.0] * 4, self.leg_dofs)
        robot.set_dofs_armature([1.95e-4 * 9.0**2] * 4, self.leg_dofs)
        robot.set_dofs_kv([0.2, 0.2], self.wheel_dofs)
        robot.set_dofs_force_range([-5.0, -5.0], [5.0, 5.0], self.wheel_dofs)
        robot.set_dofs_damping([50.0, 50.0], self.spring_dofs)
        robot.set_dofs_armature([1e-4, 1e-4], self.spring_dofs)
        self.pitch_dofs = self._dofs(("gimbal_pitch_joint",))
        self.pitch_target = torch.full((self.num_envs, 1), -0.5, device=self.device)
        robot.set_dofs_kp([20.0], self.pitch_dofs)
        robot.set_dofs_kv([0.5], self.pitch_dofs)
        robot.set_dofs_force_range([-10.0], [10.0], self.pitch_dofs)
        robot.set_dofs_damping([0.5], self._dofs(("gimbal_yaw_joint",)))
        passive = [joint.dofs_idx_local[0] for joint in robot.joints
                   if joint.name.endswith(("_front2_joint", "_front3_joint", "_front4_joint",
                                            "_rear2_joint", "_spring1_joint", "_guide_joint"))]
        robot.set_dofs_damping([0.01] * len(passive), passive)
        robot.set_dofs_armature([1e-4] * len(passive), passive)

    def _drive(self, actions):
        leg_targets = (actions[:, :4] * self.leg_action_scale).clamp(-3.14, 3.14)
        self.robot.control_dofs_position(leg_targets, self.leg_dofs)
        self.robot.control_dofs_position(self.pitch_target, self.pitch_dofs)

    def _apply_springs(self):
        spring_pos = self.robot.get_dofs_position(self.spring_dofs)
        spring_force = 400.0 + (600.0 - 400.0) / 0.07 * (0.06076 - spring_pos).clamp_min(0.0)
        self.robot.control_dofs_force(spring_force, self.spring_dofs)
