"""Solve the CAD closed chain at a grid of continuous hip-to-wheel heights."""

import math

import mujoco
import numpy as np
from scipy.optimize import least_squares

from robots.wheel_leg import HOME_JOINTS


def solve_stances(model_path, height_limits, wheel_radius):
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    data = mujoco.MjData(model)
    joints = [model.joint(name) for name in HOME_JOINTS]
    indices = [int(joint.qposadr[0]) for joint in joints]
    lower = [joint.range[0] if joint.limited[0] else -np.inf for joint in joints]
    upper = [joint.range[1] if joint.limited[0] else np.inf for joint in joints]
    hips = [model.joint(f"hip_{side}").id for side in "LR"]
    wheels = [model.joint(f"wheel_{side}").id for side in "LR"]
    base = model.body("base_link").id
    home = np.array(list(HOME_JOINTS.values()))

    def update(angles):
        data.qpos[indices] = angles
        mujoco.mj_forward(model, data)
        equalities = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        closure = data.efc_pos[:data.nefc][equalities]
        leg_vectors = (data.xanchor[hips] - data.xanchor[wheels])[:, [0, 2]]
        return closure, leg_vectors

    _, home_vectors = update(home)
    heights = np.linspace(*height_limits, max(2, math.ceil((height_limits[1] - height_limits[0]) / 0.005) + 1))
    poses = np.empty((len(heights), len(joints) + 1))
    angles = np.clip(home, np.array(lower) + 1e-8, np.array(upper) - 1e-8)
    for index in np.argsort(np.abs(heights - home_vectors[:, 1].mean())):
        height = heights[index]

        def residual(candidate):
            closure, vectors = update(candidate)
            return np.r_[10.0 * closure, vectors[:, 1] - height,
                         0.1 * (vectors[:, 0] - home_vectors[:, 0]), 0.001 * (candidate - home)]

        angles = least_squares(residual, angles, bounds=(lower, upper), max_nfev=150).x
        closure, vectors = update(angles)
        leg_lengths = np.linalg.norm(vectors, axis=1)
        if (closure.size == 0 or np.max(np.abs(closure)) > 5e-5
                or np.max(np.abs(vectors[:, 1] - height)) > 5e-4
                or np.any((leg_lengths < 0.16) | (leg_lengths > 0.335))):
            raise ValueError(f"Cannot solve a closed-chain stance at {height:.3f} m within the joint and leg limits")
        wheels_z = data.xanchor[wheels, 2] - data.xpos[base, 2]
        poses[index, 0] = wheel_radius - wheels_z.min() + 0.002
        poses[index, 1:] = angles
    return poses
