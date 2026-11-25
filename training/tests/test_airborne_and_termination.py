from math import cos, radians, sin
from types import SimpleNamespace
import unittest

import torch

from configs.experiment import load_config
from envs.flat import FlatEnv
from robots.scut_rough_reward import ScutRoughReward
from robots.wheel_leg_locomotion import WheelLegLocomotionEnv


def _gravity_at_pitch(pitch_deg):
    angle = radians(pitch_deg)
    return torch.tensor([[sin(angle), 0.0, -cos(angle)]])


class AirborneAndTerminationTest(unittest.TestCase):
    def _make_termination_stub(self):
        cfg = load_config()["task"]["termination"]
        return SimpleNamespace(
            control_dt=0.02,
            instant_pitch_rad=radians(cfg["instant_pitch_deg"]),
            sustained_pitch_rad=radians(cfg["sustained_pitch_deg"]),
            sustained_pitch_seconds=cfg["sustained_pitch_seconds"],
            pitch_exceed_time=torch.zeros(1),
        )

    def test_stair_pitch_and_brief_transient_do_not_terminate(self):
        env = self._make_termination_stub()
        for _ in range(40):
            self.assertFalse(FlatEnv._pitch_fallen(env, _gravity_at_pitch(10.0)).item())
        for _ in range(10):
            self.assertFalse(FlatEnv._pitch_fallen(env, _gravity_at_pitch(25.0)).item())
        self.assertFalse(FlatEnv._pitch_fallen(env, _gravity_at_pitch(8.0)).item())
        self.assertEqual(env.pitch_exceed_time.item(), 0.0)

    def test_sustained_nose_down_tricycle_pitch_terminates_at_threshold(self):
        env = self._make_termination_stub()
        for _ in range(14):
            self.assertFalse(FlatEnv._pitch_fallen(env, _gravity_at_pitch(27.0)).item())
        self.assertTrue(FlatEnv._pitch_fallen(env, _gravity_at_pitch(27.0)).item())

    def test_instant_extreme_pitch_and_recovery_mask(self):
        env = self._make_termination_stub()
        self.assertFalse(FlatEnv._pitch_fallen(env, _gravity_at_pitch(40.0), active=torch.tensor([False])).item())
        self.assertEqual(env.pitch_exceed_time.item(), 0.0)
        self.assertTrue(FlatEnv._pitch_fallen(env, _gravity_at_pitch(36.0)).item())

    def test_airborne_exit_timer_pauses_across_landing_bounce_and_excludes_high_stairs(self):
        cfg = load_config()["task"]
        robot = SimpleNamespace(
            n_links=5,
            get_dofs_limit=lambda _dofs: (torch.tensor([0.5, 0.5]), torch.tensor([2.5, 2.5])),
        )
        env = SimpleNamespace(
            cfg=cfg,
            control_dt=0.02,
            num_envs=1,
            commands=torch.zeros((1, 3)),
            last_actions=torch.zeros((1, 6)),
            robot=robot,
            wheel_links=[0, 1],
            body_links=[2],
            wheel_radius=0.058,
            body_clearance=torch.tensor([0.32]),
            surface=torch.tensor([0]),
            high_stair_surfaces=torch.tensor([False, True]),
            _dofs=lambda _names: [0, 1],
            ground_height_at=lambda pts: torch.zeros(pts.shape[:-1]),
        )
        reward = ScutRoughReward(env, airborne_minimum_height=0.17)
        wheels_high = torch.tensor([[[0.0, 0.2, 0.16], [0.0, -0.2, 0.16]]])
        contact = torch.zeros((1, 5, 3))

        # High-stair lane does not enter airborne.
        env.surface.fill_(1)
        airborne, _ = reward._airborne(wheels_high, contact, torch.zeros((1, 2)))
        self.assertFalse(airborne.item())

        # Normal drop enters airborne, pauses exit timer on a 1-step bounce, and exits after 3 loaded ticks (0.06 s).
        env.surface.fill_(0)
        airborne, _ = reward._airborne(wheels_high, contact, torch.zeros((1, 2)))
        self.assertTrue(airborne.item())

        env.body_clearance.fill_(0.20)
        wheels_low = torch.tensor([[[0.0, 0.2, 0.058], [0.0, -0.2, 0.058]]])
        wheels_rebound = torch.tensor([[[0.0, 0.2, 0.085], [0.0, -0.2, 0.085]]])
        loaded = torch.tensor([[35.0, 35.0]])
        unloaded = torch.zeros((1, 2))

        self.assertTrue(reward._airborne(wheels_low, contact, loaded)[0].item())
        self.assertAlmostEqual(reward.wheel_contact_time.amax().item(), 0.02)
        self.assertTrue(reward._airborne(wheels_rebound, contact, unloaded)[0].item())
        self.assertAlmostEqual(reward.wheel_contact_time.amax().item(), 0.02)
        self.assertTrue(reward._airborne(wheels_low, contact, loaded)[0].item())
        self.assertFalse(reward._airborne(wheels_low, contact, loaded)[0].item())

    def test_heading_hold_waits_until_turn_settles_before_latching(self):
        def quat_for_yaw(yaw_rad):
            return torch.tensor([[cos(0.5 * yaw_rad), 0.0, 0.0, sin(0.5 * yaw_rad)]])

        state = {"yaw": 0.0}
        env = SimpleNamespace(
            robot=SimpleNamespace(get_quat=lambda: quat_for_yaw(state["yaw"])),
            commands=torch.zeros((1, 3)),
            heading_target=torch.zeros(1),
            heading_turning=torch.zeros(1, dtype=torch.bool),
            heading_gain=2.5,
            heading_deadband=radians(0.5),
            heading_max_yaw_rate=0.40,
            _yaw_from_quat=WheelLegLocomotionEnv._yaw_from_quat,
        )

        # Active turn tracks yaw without engaging hold.
        state["yaw"] = radians(30.0)
        env.commands[0, 1] = 4.0
        out = WheelLegLocomotionEnv._heading_yaw_target(env, torch.tensor([4.0]), torch.tensor([3.5]))
        self.assertAlmostEqual(out.item(), 4.0, places=5)
        self.assertTrue(env.heading_turning.item())

        # On release, robot coasts another 20 deg while yaw command and angular velocity decelerate:
        # heading hold must output 0 (no snapback) and follow the coasting yaw.
        state["yaw"] = radians(45.0)
        env.commands[0, 1] = 1.5
        out = WheelLegLocomotionEnv._heading_yaw_target(env, torch.tensor([0.0]), torch.tensor([1.8]))
        self.assertAlmostEqual(out.item(), 0.0, places=5)
        self.assertTrue(env.heading_turning.item())

        state["yaw"] = radians(52.0)
        env.commands[0, 1] = 0.0
        out = WheelLegLocomotionEnv._heading_yaw_target(env, torch.tensor([0.0]), torch.tensor([0.4]))
        self.assertAlmostEqual(out.item(), 0.0, places=5)
        self.assertTrue(env.heading_turning.item())

        # Once settled, latch the final 55 deg heading and keep outputting 0.
        state["yaw"] = radians(55.0)
        out = WheelLegLocomotionEnv._heading_yaw_target(env, torch.tensor([0.0]), torch.tensor([0.05]))
        self.assertAlmostEqual(out.item(), 0.0, places=5)
        self.assertFalse(env.heading_turning.item())
        self.assertAlmostEqual(env.heading_target.item(), radians(55.0), places=5)

        # Subsequent external disturbance while driving straight triggers immediate correction back to 55 deg.
        state["yaw"] = radians(65.0)
        out = WheelLegLocomotionEnv._heading_yaw_target(env, torch.tensor([0.0]), torch.tensor([0.6]))
        self.assertLess(out.item(), -0.35)
        self.assertFalse(env.heading_turning.item())


if __name__ == "__main__":
    unittest.main()
