"""Genesis adapter for the in-house Wheel Leg V0.2 CAD export."""

import xml.etree.ElementTree as ET

import genesis as gs
import mujoco
import numpy as np

from configs.experiment import TRAINING_DIR
from envs.flat import FlatEnv


MODEL_PATH = TRAINING_DIR / "assets/wheel_leg_xml_V0.2/wheel_leg_ground.xml"
HOME_HEIGHT = 0.2370327302
# Solved on the CAD assembly branch with both axles under the center of mass
# and the passive knees 0.20 rad inside their travel limits.
HOME_JOINTS = {
    "hip_L": -0.585,
    "hip_R": 0.585,
    "hipknee_L": -0.8201266288,
    "hipknee_R": 0.8201266397,
    "knee_L": 1.4566159,
    "knee_R": 1.8566159,
    "gas_up_L": -0.0011092105,
    "gas_up_R": 0.0011092105,
    "gas_slide_L": -0.0100408686,
    "gas_slide_R": -0.0100408686,
    "drive_l1_L": 0.2000000000,
    "drive_l1_R": -0.2000000043,
    "l1_node_L": -0.1648733712,
    "l1_node_R": 0.1648733647,
    "node_l2_L": 0.2000000000,
    "node_l2_R": -0.2000000001,
}


def _vector(values):
    return " ".join(f"{value:.10g}" for value in values)


def _collision_model(path, *, include_visuals=False):
    """Fit one collider per rigid body and preserve the CAD joint tree and inertials."""
    xml = ET.parse(path).getroot()
    cad = mujoco.MjModel.from_xml_path(str(path.resolve()))
    world = xml.find("worldbody")
    for geom in world.findall("geom"):
        world.remove(geom)
    if include_visuals:
        compiler = xml.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(xml, "compiler")
        mesh_dir = compiler.get("meshdir", compiler.get("assetdir", ""))
        compiler.set("meshdir", (path.resolve().parent / mesh_dir).as_posix())
    else:
        xml.remove(xml.find("asset"))
    # Genesis renders groups 0–2 by default. Group 3 keeps these geoms collision-only.
    collision_group = {"group": "3"} if include_visuals else {}
    for body in world.iter("body"):
        name = body.attrib["name"]
        body_id = cad.body(name).id
        vertices = []
        for geom_id in np.flatnonzero(cad.geom_bodyid == body_id):
            mesh_id = cad.geom_dataid[geom_id]
            start = cad.mesh_vertadr[mesh_id]
            end = start + cad.mesh_vertnum[mesh_id]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, cad.geom_quat[geom_id])
            vertices.append(cad.mesh_vert[start:end] @ rotation.reshape(3, 3).T + cad.geom_pos[geom_id])
        points = np.concatenate(vertices)
        for geom in body.findall("geom"):
            if include_visuals:
                geom.attrib.update(contype="0", conaffinity="0", mass="0")
            else:
                body.remove(geom)

        if name.startswith(("wheel_", "guide_")):
            joint = cad.joint(cad.body_jntadr[body_id])
            offset = points - joint.pos
            axial = offset @ joint.axis
            radius = np.linalg.norm(offset - axial[:, None] * joint.axis, axis=1)
            # Fit the outer tread using vertices near the maximum radius.
            tread = axial[radius >= 0.95 * radius.max()]
            ends = joint.pos + np.array([tread.min(), tread.max()])[:, None] * joint.axis
            ET.SubElement(body, "geom", name=f"{name}_contact", type="cylinder",
                          size=_vector([radius.max()]), fromto=_vector(ends.ravel()),
                          rgba="0.16 0.18 0.22 1", **collision_group)
        else:
            lower, upper = points.min(axis=0), points.max(axis=0)
            ET.SubElement(body, "geom", name=f"{name}_contact", type="box",
                          size=_vector((upper - lower) / 2), pos=_vector((upper + lower) / 2),
                          rgba="0.55 0.62 0.7 1", **collision_group)
    return xml


class WheelLegFlatEnv(FlatEnv):
    model_path = MODEL_PATH
    label = "Wheel Leg V0.2"
    physics_dt = 0.002
    decimation = 10
    height_target = HOME_HEIGHT
    min_height = 0.12
    wheel_radius = 0.058
    leg_torque_limit = 12.0
    leg_action_scale = 0.20
    leg_joints = ("hip_L", "hip_R", "hipknee_L", "hipknee_R")
    wheel_joints = ("wheel_L", "wheel_R")
    spring_joints = ("gas_slide_L", "gas_slide_R")
    wheel_bodies = wheel_joints
    body_contacts = ("base_link", "gimbal_yaw", "gimbal_pitch")
    leg_signs = (1.0, -1.0, 1.0, -1.0)
    wheel_signs = (1.0, -1.0)
    expected_equalities = 12
    min_support_load = 100.0

    def _robot_morph(self, model_path, viewer):
        return gs.morphs.MJCF(file=_collision_model(model_path, include_visuals=viewer),
                              visualization=viewer, default_armature=1e-4)

    def _set_home_pose(self):
        base = self.robot.get_joint("base_link").qs_idx_local
        self.home_qpos[:, base[2]] = HOME_HEIGHT
        for name, position in HOME_JOINTS.items():
            self.home_qpos[:, self.robot.get_joint(name).qs_idx_local[0]] = position

    def scripted_wheel_speed(self, velocity, angular_velocity, gravity, speed_integral):
        speed_error = self.commands[:, 0] - velocity[:, 0]
        # Move the wheels under the leaning body. A positive speed error first induces forward lean.
        surface_speed = (velocity[:, 0] + 4.0 * gravity[:, 0] + 0.28 * angular_velocity[:, 1]
                         - 0.9 * speed_error - 0.12 * speed_integral)
        return surface_speed / self.wheel_radius

    def _set_drives(self):
        robot = self.robot
        leg_kp = 100.0
        robot.set_dofs_kp([leg_kp] * 4, self.leg_dofs)
        robot.set_dofs_kv([4.0] * 4, self.leg_dofs)
        robot.set_dofs_force_range([-self.leg_torque_limit] * 4, [self.leg_torque_limit] * 4, self.leg_dofs)
        robot.set_dofs_armature([0.01] * 4, self.leg_dofs)
        # Convert the static hip torque needed at the neutral pose into a PD target offset.
        self.leg_bias = self.robot.get_qpos().new_tensor((-4.25, 4.30, 4.31, -4.32)) / leg_kp
        robot.set_dofs_kv([0.6] * 2, self.wheel_dofs)
        robot.set_dofs_force_range([-5.0] * 2, [5.0] * 2, self.wheel_dofs)
        robot.set_dofs_damping([5.0] * 2, self.spring_dofs)
        self.spring_forces = self.robot.get_qpos().new_full((self.num_envs, 2), 200.0)
        self.gimbal_dofs = self._dofs(("gimbal_yaw_joint", "gimbal_pitch_joint"))
        self.gimbal_target = self.robot.get_qpos().new_zeros((self.num_envs, 2))
        robot.set_dofs_kp([20.0] * 2, self.gimbal_dofs)
        robot.set_dofs_kv([1.0] * 2, self.gimbal_dofs)
        robot.set_dofs_force_range([-10.0] * 2, [10.0] * 2, self.gimbal_dofs)
        self.passive_dofs = [joint.dofs_idx_local[0] for joint in robot.joints
                             if joint.name.startswith(("knee_", "drive_l1_", "l1_node_", "node_l2_", "gas_up_"))]
        robot.set_dofs_damping([0.04] * len(self.passive_dofs), self.passive_dofs)

    def _drive(self, actions):
        offsets = (actions[:, :4] * self.leg_action_scale).clamp(-0.35, 0.35)
        targets = self.home_leg_pos + offsets * self.leg_direction + self.leg_bias
        self.robot.control_dofs_position(targets, self.leg_dofs)
        self.robot.control_dofs_position(self.gimbal_target, self.gimbal_dofs)

    def _apply_springs(self):
        self.robot.control_dofs_force(self.spring_forces, self.spring_dofs)
