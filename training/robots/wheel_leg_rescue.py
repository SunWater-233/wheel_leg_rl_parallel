"""Firmware-style RESCUE and GETUP, using the Genesis CAD linkage for VMC."""

from math import pi
import xml.etree.ElementTree as ET

from genesis.utils.geom import inv_transform_by_quat, transform_by_quat
import torch


def wrap(angle):
    return torch.atan2(angle.sin(), angle.cos())


def cross2(first, second):
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def rotate_xz(vector):
    """Linear velocity in the sagittal plane for one rad/s about body +Y."""
    return torch.stack((vector[..., 1], -vector[..., 0]), dim=-1)


class CadVmc:
    """Reduce the two closed CAD loops to the two driven hip coordinates."""

    def __init__(self, env, model):
        self.env = env
        names = ("hip", "hipknee", "drive_l1", "l1_node", "node_l2", "knee", "wheel", "gas_up", "gas_slide")
        self.joints = [env.robot.get_joint(f"{name}_{side}") for side in "LR" for name in names]
        self.joint_count = len(names)
        xml = ET.parse(model)
        sites = {
            site.attrib["name"]: (body.attrib["name"], [float(x) for x in site.attrib["pos"].split()])
            for body in xml.iter("body") for site in body.findall("site")
        }
        endpoints = ("node_thigh_{0}_a", "node_thigh_{0}_b", "l2_shin_{0}_a", "l2_shin_{0}_b", "gas_low_{0}_b")
        pairs = [sites[f"{name.format(side)}_minus"] for side in "LR" for name in endpoints]
        positions = [position for _, position in pairs]
        self.links = [env.robot.get_link(name).idx_local for name, _ in pairs]
        self.site_positions = env.commands.new_tensor(positions)[None, :, :].expand(env.num_envs, -1, -1).contiguous()
        quat = env.robot.get_quat()[:, None, None, :]
        axes = self._joint_vectors("get_anchor_axis")
        local_axes = inv_transform_by_quat(axes, quat.expand(*axes.shape[:-1], 4).contiguous())
        self.axis = local_axes[0, ..., 1].sign()
        # Signed incidences of the two sides of each connect constraint.
        self.incidence = env.commands.new_tensor((
            ((0, 1, 1, 1, 0, 0), (-1, 0, 0, 0, 0, 0)),
            ((0, 1, 1, 1, 1, 0), (-1, 0, 0, 0, 0, -1)),
        ))

    def _joint_vectors(self, method):
        vectors = torch.stack([getattr(joint, method)() for joint in self.joints], dim=1)
        return vectors.reshape(-1, 2, self.joint_count, 3)

    def measure(self):
        robot = self.env.robot
        base = robot.get_pos()[:, None, None, :]
        quat = robot.get_quat()[:, None, None, :]
        anchors = self._joint_vectors("get_anchor_pos")
        slide_axis = torch.stack([self.joints[i].get_anchor_axis() for i in (8, 17)], dim=1)
        sites = robot.get_links_pos(self.links) + transform_by_quat(self.site_positions, robot.get_links_quat(self.links))
        points = torch.cat((anchors, sites.reshape(-1, 2, 5, 3), (slide_axis[:, :, None] + base)), dim=2)
        xz = inv_transform_by_quat((points - base).contiguous(), quat.expand(*points.shape[:-1], 4).contiguous())
        xz = xz[..., (0, 2)]
        # Differentiate both loop closures, holding chassis motion fixed.
        ends = xz[..., 9:13, :].reshape(-1, 2, 2, 2, 2)
        lever = ends[..., None, :] - xz[:, :, None, None, :6, :]
        derivatives = rotate_xz(lever) * self.axis[None, :, None, None, :6, None]
        closure = (derivatives * self.incidence[None, None, ..., None]).sum(dim=3)
        closure = closure.transpose(-2, -1).reshape(-1, 2, 4, 6)
        passive, info = torch.linalg.solve_ex(closure[..., 2:].contiguous(), -closure[..., :2].contiguous())
        valid = (info == 0) & passive.isfinite().all(dim=(-1, -2))
        passive = torch.where(valid[..., None, None], passive, 0.0)

        wheel, hip, knee = xz[..., 6, :], xz[..., 0, :], xz[..., 5, :]
        virtual_leg = wheel - hip
        length = virtual_leg.norm(dim=-1).clamp_min(1e-4)
        tangent = torch.stack((-virtual_leg[..., 1], virtual_leg[..., 0]), dim=-1) / length[..., None]
        # phi0 = atan2(-z, x), with down positive, as in the firmware.
        phi = torch.atan2(-virtual_leg[..., 1], virtual_leg[..., 0])
        d_wheel_dhip = rotate_xz(virtual_leg) * self.axis[None, :, 0, None]
        d_wheel_dknee = rotate_xz(wheel - knee) * self.axis[None, :, 5, None]
        wheel_jac = d_wheel_dknee[..., :, None] * passive[..., 3, None, :]
        wheel_jac[..., 0] += d_wheel_dhip
        d_length = (wheel_jac * (virtual_leg / length[..., None])[..., :, None]).sum(dim=-2)
        d_phi = -(wheel_jac * tangent[..., :, None]).sum(dim=-2) / length[..., None]

        # Cancel the CAD gas spring's contribution in the driven coordinates.
        spring_tip = xz[..., 13, :]
        gas_rotation = rotate_xz(spring_tip - xz[..., 7, :]) * self.axis[None, :, 7, None]
        tip_motion = rotate_xz(spring_tip - knee) * self.axis[None, :, 5, None]
        denominator = cross2(gas_rotation, xz[..., 14, :])
        gas_valid = denominator.abs() > 1e-5
        d_slide = torch.where(gas_valid, cross2(gas_rotation, tip_motion) /
                             torch.where(gas_valid, denominator, 1.0), 0.0)
        spring_torque = self.env.spring_forces[..., None] * d_slide[..., None] * passive[..., 3, :]
        return length, phi, d_length, d_phi, spring_torque, valid


class FirmwareRecovery:
    """Batched port of WL_Rescue and WL_GetUP for the level-ground probe."""

    EXTEND, SWING, SINGLE, RETRACT, BALANCE = range(5)

    def __init__(self, env, model):
        self.env = env
        self.vmc = CadVmc(env, model)
        self.dt = env.physics_dt
        self.mode = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.enabled = torch.ones_like(self.mode, dtype=torch.bool)
        self.ready = torch.zeros_like(self.enabled)
        self.support_leg = torch.zeros_like(env.spring_forces, dtype=torch.bool)
        self.wait = env.commands[:, 0].clone()
        self.hold = torch.zeros_like(self.wait)
        self.boost = torch.zeros_like(self.wait)
        self.stall = torch.zeros_like(env.spring_forces)
        self.stall_boost = torch.zeros_like(self.stall)
        self.rate_integral = torch.zeros_like(self.stall)
        self.last_rate = torch.zeros_like(self.stall)
        self.rate_derivative = torch.zeros_like(self.stall)
        self.length_rate = torch.zeros_like(self.stall)
        self.last_length = None
        self.wheel_gain = env.commands.new_tensor((
            (.896700, 11.845123, 1.786277, .333522),
            (.896700, 11.845123, 1.786277, -.333522),
        ))

    def advance(self):
        length, phi, d_length, d_phi, spring, valid = self.vmc.measure()
        vel, ang, gravity, _ = self.env._base_state()
        # Preserve the sign beyond 90 degrees so front and back falls get different swing directions.
        pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
        roll = torch.atan2(-gravity[:, 1], torch.hypot(gravity[:, 0], gravity[:, 2]))
        theta = wrap(phi - pi / 2 + pitch[:, None])
        current = self.mode.clone()
        extend = self.enabled & (current == self.EXTEND)
        swing = self.enabled & ((current == self.SWING) | (current == self.SINGLE))
        single = current == self.SINGLE
        retract = self.enabled & (current == self.RETRACT)
        balance = self.enabled & (current == self.BALANCE)
        leg_ready = ((theta + .7).abs() < .3) & (length < .53)
        upright = (pitch.abs() < pi / 6) & (roll.abs() < pi / 6)
        select_single = extend & upright & (leg_ready[:, 0] != leg_ready[:, 1])
        long_enough = extend & ~select_single & (length >= .28).all(dim=1)
        direct_getup = extend & ~select_single & ~long_enough & leg_ready.all(dim=1)
        self.hold = torch.where(extend, torch.where(long_enough, self.hold + self.dt, 0.0), self.hold)
        self.wait += self.dt * (extend & ~select_single & ~long_enough & ~direct_getup)
        self.mode.masked_fill_(long_enough & (self.hold >= .05), self.SWING)
        self.mode.masked_fill_(select_single, self.SINGLE)
        self.support_leg.copy_(torch.where(select_single[:, None], leg_ready, self.support_leg))

        swing_dir = torch.where((pitch > pi / 3) & (pitch < pi), 1.0, -1.0)
        angle_error = wrap(pi / 2 - phi)
        rate_target = torch.where(balance[:, None], 4.0 * (angle_error / .1).clamp(-1.0, 1.0),
                                  2.5 * swing_dir[:, None])
        supporting = single[:, None] & self.support_leg
        active_rate = (swing[:, None] & ~supporting) | balance[:, None]
        joint_rate = self.env.robot.get_dofs_velocity(self.env.leg_dofs).reshape(-1, 2, 2).transpose(1, 2)
        rate = (d_phi * joint_rate).sum(dim=-1)
        alpha = self.dt / (.05 + self.dt)
        self.rate_derivative.lerp_((rate - self.last_rate) / self.dt, alpha)
        self.last_rate.copy_(rate)
        error = rate_target - rate
        self.rate_integral = torch.where(active_rate, (self.rate_integral + 12.0 * error * self.dt).clamp(-8, 8), 0.0)
        rate_output = (6.0 * error + self.rate_integral - .002 * self.rate_derivative).clamp(-8.0, 8.0)
        rate_output = torch.where(active_rate, rate_output, 0.0)

        touched = active_rate & (rate_output.abs() >= 7.999) & (rate.abs() <= .5)
        both_touch = touched.all(dim=1)
        good_pose = (pitch.abs() <= pi / 3) & (wrap(phi - pi).abs() < pi / 3).all(dim=1)
        support_ok = upright & (leg_ready & self.support_leg).any(dim=1)
        single_lost = swing & single & ~support_ok
        self.mode.masked_fill_(single_lost, self.EXTEND)
        swing_complete = swing & torch.where(single, support_ok & (leg_ready.all(dim=1) | touched.any(dim=1)),
                                             both_touch & good_pose)
        self.hold = torch.where(swing, torch.where(swing_complete, self.hold + self.dt, 0.0), self.hold)
        to_retract = direct_getup | (swing_complete & (self.hold >= .05))
        self.mode.masked_fill_(to_retract, self.RETRACT)
        self.mode.masked_fill_(retract & (length < .20).all(dim=1), self.BALANCE)

        need_boost = swing & ~single & both_touch & ~good_pose
        self.boost = (self.boost + self.dt * torch.where(need_boost, 600.0, -60.0)).clamp(0.0, 60.0)
        stalled = swing[:, None] & touched & ~both_touch[:, None]
        self.stall = torch.where(stalled, self.stall + self.dt, 0.0)
        self.stall_boost = torch.where(
            self.stall >= 1.0, (self.stall_boost + self.dt * 10.0 * rate_output.sign()).clamp(-20.0, 20.0),
            self.stall_boost.sign() * (self.stall_boost.abs() - 20.0 * self.dt).clamp_min(0.0),
        )
        gravity_torque = 9.81 * (.64848 * length + (0.423 * length - .0379) * 1.16693) * theta.sin()
        swing_torque = rate_output + gravity_torque + self.boost[:, None] * swing_dir[:, None] + self.stall_boost
        torque = torch.where(swing[:, None] & ~supporting, swing_torque, 0.0)
        torque = torch.where(retract[:, None], -.5, torque)
        torque = torch.where(balance[:, None], rate_output * 2.5, torque)

        target_length = torch.where((retract | balance)[:, None], .17, .33)
        target_length = torch.where(supporting, length, target_length)
        if self.last_length is None:
            self.last_length = length
        self.length_rate.lerp_((length - self.last_length) / self.dt, alpha)
        self.last_length = length
        length_force = (1200.0 * (target_length - length) - 300.0 * self.length_rate).clamp(-100, 100)
        push = torch.where(swing[:, None] & ~supporting, 80.0, 0.0)
        push = torch.where(((self.wait >= 1.0) & (extend | (swing & ~single)))[:, None], 180.0, push)
        push = torch.where((retract | balance)[:, None], -50.0, push)
        axial_force = (length_force + push).clamp(-300.0, 300.0)
        joint_torque = axial_force[..., None] * d_length + torque[..., None] * d_phi - spring
        joint_torque = torch.where(valid[..., None], joint_torque, 0.0).clamp(-20.0, 20.0)
        joint_torque = joint_torque.transpose(1, 2).reshape(-1, 4)

        wheel_enabled = balance & (angle_error.abs() < .20).all(dim=1)
        wheel_state = torch.stack((vel[:, 0], pitch, ang[:, 1], ang[:, 2]), dim=1)
        # Respect the wheel effort limit of the Genesis locomotion model during this probe.
        wheel_torque = torch.where(wheel_enabled[:, None], (wheel_state @ self.wheel_gain.T).clamp(-5.0, 5.0), 0.0)
        stable = balance & (pitch.abs() < pi / 18) & (theta.abs() < pi / 18).all(dim=1)
        self.hold = torch.where(balance, torch.where(stable, self.hold + self.dt, 0.0), self.hold)
        self.ready |= stable & (self.hold >= .05)
        changed = self.mode != current
        self.hold.masked_fill_(changed, 0)
        self.rate_integral.masked_fill_(changed[:, None], 0)
        self.rate_derivative.masked_fill_(changed[:, None], 0)
        self.boost.masked_fill_(changed, 0)
        self.stall.masked_fill_(changed[:, None], 0)
        self.stall_boost.masked_fill_(changed[:, None], 0)
        self.wait.masked_fill_(to_retract | select_single, 0)
        ids = self.env.all_envs[self.enabled]
        if ids.numel():
            forces = torch.cat((joint_torque, wheel_torque * self.env.wheel_direction), dim=1)
            self.env.robot.control_dofs_force(forces[ids], self.env.action_dofs, envs_idx=ids)
        return length, valid
