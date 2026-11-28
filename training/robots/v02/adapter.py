"""Genesis adapter for the in-house Wheel Leg V0.2 CAD export."""

import xml.etree.ElementTree as ET

import genesis as gs
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from robots.v02.spec import (
    BODY_CONTACTS,
    DECIMATION,
    EXPECTED_EQUALITIES,
    HOME_HEIGHT,
    HOME_JOINTS,
    LABEL,
    LEG_ACTION_SCALE,
    LEG_ARMATURE,
    LEG_BIAS_TORQUE,
    LEG_JOINTS,
    LEG_KP,
    LEG_KV,
    LEG_SIGNS,
    LEG_TORQUE_LIMIT_DEFAULT,
    MIN_HEIGHT,
    MODEL_PATH,
    PHYSICS_DT,
    SPRING_FORCE_NOMINAL,
    SPRING_JOINTS,
    WHEEL_BODIES,
    WHEEL_JOINTS,
    WHEEL_KV,
    WHEEL_RADIUS,
    WHEEL_SIGNS,
    WHEEL_TORQUE_LIMIT_DEFAULT,
)
from tasks.base_env import FlatEnv


def _vector(values):
    return " ".join(f"{value:.10g}" for value in values)


def _add_cad_hull(assets, body, points, geom_attrs):
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    vertex_indices = np.empty(len(points), dtype=np.intp)
    vertex_indices[hull.vertices] = np.arange(len(vertices))
    face_indices = vertex_indices[hull.simplices]
    triangles = vertices[face_indices]
    winding = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    inward = np.einsum("ij,ij->i", winding, hull.equations[:, :3]) < 0.0
    face_indices[inward] = face_indices[inward, ::-1]
    ET.SubElement(assets, "mesh", name="base_link_cad_hull", vertex=_vector(vertices.ravel()),
                  face=" ".join(map(str, face_indices.ravel())))
    ET.SubElement(body, "geom", name="base_link_contact", type="mesh", mesh="base_link_cad_hull", **geom_attrs)


def _collision_model(path, *, include_visuals=False, use_cad_base_hull=False):
    """Fit one collider per rigid body while preserving the CAD joint tree and inertials."""
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
        if use_cad_base_hull:
            ET.SubElement(xml, "asset")
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

        if name == "base_link" and use_cad_base_hull:
            _add_cad_hull(xml.find("asset"), body, points, {"rgba": "0.55 0.62 0.7 1", **collision_group})
        elif name.startswith(("wheel_", "guide_")):
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
    label = LABEL
    physics_dt = PHYSICS_DT
    decimation = DECIMATION
    height_target = HOME_HEIGHT
    min_height = MIN_HEIGHT
    wheel_radius = WHEEL_RADIUS
    leg_torque_limit = LEG_TORQUE_LIMIT_DEFAULT
    wheel_torque_limit = WHEEL_TORQUE_LIMIT_DEFAULT
    wheel_armature = 0.0
    leg_action_scale = LEG_ACTION_SCALE
    leg_joints = LEG_JOINTS
    wheel_joints = WHEEL_JOINTS
    spring_joints = SPRING_JOINTS
    wheel_bodies = WHEEL_BODIES
    body_contacts = BODY_CONTACTS
    leg_signs = LEG_SIGNS
    wheel_signs = WHEEL_SIGNS
    expected_equalities = EXPECTED_EQUALITIES

    def _robot_morph(self, model_path, viewer):
        return gs.morphs.MJCF(file=_collision_model(model_path, include_visuals=viewer,
                                                  use_cad_base_hull=self.cfg["use_cad_base_hull"]),
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
        robot.set_dofs_kp([LEG_KP] * 4, self.leg_dofs)
        robot.set_dofs_kv([LEG_KV] * 4, self.leg_dofs)
        robot.set_dofs_force_range([-self.leg_torque_limit] * 4, [self.leg_torque_limit] * 4, self.leg_dofs)
        robot.set_dofs_armature([LEG_ARMATURE] * 4, self.leg_dofs)
        # Convert the static hip torque needed at the neutral pose into a PD target offset.
        self.leg_bias = self.robot.get_qpos().new_tensor(LEG_BIAS_TORQUE) / LEG_KP
        robot.set_dofs_kv([WHEEL_KV] * 2, self.wheel_dofs)
        robot.set_dofs_force_range([-self.wheel_torque_limit] * 2, [self.wheel_torque_limit] * 2, self.wheel_dofs)
        if self.wheel_armature > 0.0:
            robot.set_dofs_armature([self.wheel_armature] * 2, self.wheel_dofs)
        robot.set_dofs_damping([5.0] * 2, self.spring_dofs)
        self.spring_forces = self.robot.get_qpos().new_full((self.num_envs, 2), SPRING_FORCE_NOMINAL)
        self.gimbal_dofs = self._dofs(("gimbal_yaw_joint", "gimbal_pitch_joint"))
        self.gimbal_target = self.robot.get_qpos().new_zeros((self.num_envs, 2))
        robot.set_dofs_kp([20.0] * 2, self.gimbal_dofs)
        robot.set_dofs_kv([1.0] * 2, self.gimbal_dofs)
        robot.set_dofs_force_range([-10.0] * 2, [10.0] * 2, self.gimbal_dofs)
        self.passive_dofs = [joint.dofs_idx_local[0] for joint in robot.joints
                             if joint.name.startswith(("knee_", "drive_l1_", "l1_node_", "node_l2_", "gas_up_"))]
        robot.set_dofs_damping([0.04] * len(self.passive_dofs), self.passive_dofs)
        self.guide_dofs = [joint.dofs_idx_local[0] for joint in robot.joints if joint.name.startswith("guide_")]
        robot.set_dofs_damping([0.005] * len(self.guide_dofs), self.guide_dofs)
        robot.set_dofs_armature([1e-4] * len(self.guide_dofs), self.guide_dofs)

    def _drive(self, actions):
        offsets = (actions[:, :4] * self.leg_action_scale).clamp(-0.35, 0.35)
        targets = self.home_leg_pos + offsets * self.leg_direction + self.leg_bias
        self.robot.control_dofs_position(targets, self.leg_dofs)
        self.robot.control_dofs_position(self.gimbal_target, self.gimbal_dofs)

    def _apply_springs(self):
        self.robot.control_dofs_force(self.spring_forces, self.spring_dofs)
