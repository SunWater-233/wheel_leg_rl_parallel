import unittest
from types import SimpleNamespace

import torch

from configs.experiment import TRAINING_DIR, _merge_config, load_config
from envs.flat import FlatEnv
from robots.domain_randomization import randomize_dynamics
from robots.wheel_leg_locomotion import WheelLegLocomotionEnv


class Sim2RealDelayAndConfigTest(unittest.TestCase):
    def test_sim2real_configs_load_and_validate(self):
        s1 = load_config(TRAINING_DIR / "configs/v02_sim2real_stage1_flat.toml")
        s2 = load_config(TRAINING_DIR / "configs/v02_sim2real_stage2_robust.toml")

        for cfg in (s1, s2):
            loco = cfg["task"]["locomotion"]
            dr = cfg["task"]["domain_randomization"]
            self.assertEqual(loco["leg_target_delay_steps"], [4, 9])
            self.assertEqual(loco["wheel_target_delay_steps_range"], [5, 11])
            self.assertEqual(loco["imu_obs_delay_steps"], [1, 4])
            self.assertAlmostEqual(loco["wheel_armature"], 0.005)
            self.assertTrue(dr["enabled"])
            self.assertEqual(dr["spring_force_scale"], [0.40, 1.15])
            self.assertEqual(dr["wheel_armature_scale"], [0.50, 1.60])
            self.assertEqual(dr["com_jitter_xyz"], [0.02, 0.02, 0.02])

        self.assertEqual(s1["task"]["locomotion"]["terrain"], "plane")
        self.assertEqual(s2["task"]["locomotion"]["terrain"], "scut_ground")

    def test_delay_and_armature_config_validation_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            _merge_config({"task": {"locomotion": {"leg_target_delay_steps": [8, 4]}}})
        with self.assertRaises(ValueError):
            _merge_config({"task": {"locomotion": {"leg_target_delay_steps": [0, 20]}}})
        with self.assertRaises(ValueError):
            _merge_config({"task": {"locomotion": {"imu_obs_delay_steps": [0, 10]}}})
        with self.assertRaises(ValueError):
            _merge_config({"task": {"locomotion": {"wheel_armature": -0.001}}})
        with self.assertRaises(ValueError):
            _merge_config({
                "task": {
                    "locomotion": {"enabled": True},
                    "domain_randomization": {"enabled": True, "wheel_armature_scale": [0.0, 1.5]},
                }
            })
        with self.assertRaises(ValueError):
            _merge_config({
                "task": {
                    "locomotion": {"enabled": True},
                    "domain_randomization": {"enabled": True, "com_jitter_xyz": [0.01, -0.01, 0.0]},
                }
            })

    def test_select_delayed_actions_handles_substep_and_multi_step_delays(self):
        env = object.__new__(WheelLegLocomotionEnv)
        env.decimation = 10
        env.leg_delay_steps = torch.tensor([[0], [4], [10], [12]], dtype=torch.long)
        env.wheel_delay_steps = torch.tensor([[5], [10], [12], [0]], dtype=torch.long)
        env.prev2_actions = torch.full((4, 6), 1.0)
        env.last_actions = torch.full((4, 6), 2.0)
        curr = torch.full((4, 6), 3.0)

        # substep=0: d=0 -> curr(3), d in [1..10] -> last(2), d=12 -> prev2(1)
        a0 = env._select_delayed_actions(curr, 0)
        self.assertEqual(a0[:, 0].tolist(), [3.0, 2.0, 2.0, 1.0])
        self.assertEqual(a0[:, 4].tolist(), [2.0, 2.0, 1.0, 3.0])

        # substep=2: d=12 transitions from prev2(1) to last(2)
        a2 = env._select_delayed_actions(curr, 2)
        self.assertEqual(a2[:, 0].tolist(), [3.0, 2.0, 2.0, 2.0])
        self.assertEqual(a2[:, 4].tolist(), [2.0, 2.0, 2.0, 3.0])

        # substep=4: leg d=4 transitions to curr(3); wheel d=5 still last(2)
        a4 = env._select_delayed_actions(curr, 4)
        self.assertEqual(a4[:, 0].tolist(), [3.0, 3.0, 2.0, 2.0])
        self.assertEqual(a4[:, 4].tolist(), [2.0, 2.0, 2.0, 3.0])

        # substep=5: wheel d=5 transitions to curr(3)
        a5 = env._select_delayed_actions(curr, 5)
        self.assertEqual(a5[:, 4].tolist(), [3.0, 2.0, 2.0, 3.0])

    def test_imu_obs_delay_applies_to_actor_only_not_critic(self):
        class DummyRobot:
            def get_quat(self):
                return torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

            def get_vel(self):
                return torch.zeros((2, 3))

            def get_ang(self):
                return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

            def get_pos(self):
                return torch.tensor([[0.0, 0.0, 0.25], [0.0, 0.0, 0.25]])

            def get_dofs_position(self, dofs):
                return torch.zeros((2, len(dofs)))

            def get_dofs_velocity(self, dofs):
                return torch.zeros((2, len(dofs)))

        class DummyJoint:
            def __init__(self, z):
                self._z = z

            def get_anchor_pos(self):
                return torch.tensor([[0.0, 0.0, self._z], [0.0, 0.0, self._z]])

        env = object.__new__(WheelLegLocomotionEnv)
        env.num_envs = 2
        env.device = torch.device("cpu")
        env.all_envs = torch.arange(2)
        env.robot = DummyRobot()
        env.world_gravity = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
        env.hips = [DummyJoint(0.25), DummyJoint(0.25)]
        env.wheels = [DummyJoint(0.05), DummyJoint(0.05)]
        env.pose_table = torch.zeros((2, 17))
        env.height_low, env.height_high = 0.17, 0.33
        env.leg_columns = [1, 2, 3, 4]
        env.leg_dofs = [0, 1, 2, 3]
        env.action_dofs = [0, 1, 2, 3, 4, 5]
        env.leg_direction = torch.ones(4)
        env.action_direction = torch.ones(6)
        env.commands = torch.tensor([[0.0, 0.0, 0.20], [0.0, 0.0, 0.20]])
        env.motion_scales = torch.tensor([1.0, 1.0, 5.0])
        env.height_rate = torch.zeros(2)
        env.last_actions = torch.zeros((2, 6))
        env.critic_scan_offsets = None
        env.observation_noise = None
        env.actor_true_forward_speed = False
        env.policy_history = None
        env.imu_obs_delay_steps = torch.tensor([0, 2], dtype=torch.long)
        env._imu_ang_buf = torch.zeros((2, 3, 3))
        env._imu_ang_buf[0, 0] = torch.tensor([1.0, 2.0, 3.0])
        env._imu_ang_buf[1, 2] = torch.tensor([0.1, 0.2, 0.3])
        env._imu_grav_buf = torch.tensor([
            [[0.0, 0.0, -1.0]] * 3,
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.2, -0.1, -0.97]],
        ])

        obs = env.get_observations()
        # Env 1 actor sees delayed gyro [0.1, 0.2, 0.3] while critic keeps ground-truth [4.0, 5.0, 6.0].
        torch.testing.assert_close(obs["policy"][1, 4:7], torch.tensor([0.1, 0.2, 0.3]))
        torch.testing.assert_close(obs["critic"][1, 4:7], torch.tensor([4.0, 5.0, 6.0]))
        torch.testing.assert_close(obs["policy"][1, 7:10], torch.tensor([0.2, -0.1, -0.97]))
        torch.testing.assert_close(obs["critic"][1, 7:10], torch.tensor([0.0, 0.0, -1.0]))

    def test_domain_randomization_supports_asymmetric_springs_com_jitter_and_wheel_armature(self):
        torch.manual_seed(7)
        count = 64

        class DummyRobot:
            n_links = 2

            def __init__(self):
                self.mass = torch.ones((count, 1)) * 10.0
                self.com = torch.zeros((count, 1, 3))
                self.inertia = torch.eye(3).expand(count, 1, 3, 3).clone()
                self.fric = {i: torch.zeros((count, 2 if i in (2, 3) else 4)) for i in range(4)}
                self.damp = {i: torch.zeros((count, 2 if i in (2, 3) else 4)) for i in range(4)}
                self.kp = torch.full((count, 4), 100.0)
                self.kv_leg = torch.full((count, 4), 4.0)
                self.kv_wheel = torch.full((count, 2), 0.6)
                self.wheel_arm = torch.full((count, 2), 0.005)
                self.friction_ratio = None

            def set_friction_ratio(self, r):
                self.friction_ratio = r

            def get_link(self, name):
                return SimpleNamespace(idx_local=0)

            def get_links_mass(self, idx):
                return self.mass

            def get_links_COM(self, idx):
                return self.com

            def get_links_inertia(self, idx):
                return self.inertia

            def set_links_mass(self, m, idx):
                self.mass = m

            def set_links_COM(self, c, idx):
                self.com = c

            def set_links_inertia(self, i, idx):
                self.inertia = i

            def get_dofs_frictionloss(self, dofs):
                return torch.zeros((count, len(dofs)))

            def get_dofs_damping(self, dofs):
                return torch.zeros((count, len(dofs)))

            def set_dofs_frictionloss(self, v, dofs):
                pass

            def set_dofs_damping(self, v, dofs):
                pass

            def get_dofs_kp(self, dofs):
                return self.kp

            def set_dofs_kp(self, v, dofs):
                self.kp = v

            def get_dofs_kv(self, dofs):
                return self.kv_leg if len(dofs) == 4 else self.kv_wheel

            def set_dofs_kv(self, v, dofs):
                if len(dofs) == 4:
                    self.kv_leg = v
                else:
                    self.kv_wheel = v

            def get_dofs_armature(self, dofs):
                return self.wheel_arm

            def set_dofs_armature(self, v, dofs):
                self.wheel_arm = v

        cfg = load_config(TRAINING_DIR / "configs/v02_sim2real_stage2_robust.toml")["task"]
        robot = DummyRobot()
        env = SimpleNamespace(
            robot=robot,
            terrain_contacts=[],
            commands=torch.zeros((count, 3)),
            num_envs=count,
            device=torch.device("cpu"),
            cfg=cfg,
            leg_dofs=[0, 1, 2, 3],
            passive_dofs=[4, 5, 6, 7],
            spring_dofs=[8, 9],
            wheel_dofs=[10, 11],
            spring_forces=torch.full((count, 2), 200.0),
        )
        randomize_dynamics(env, cfg["domain_randomization"])

        # Asymmetric left/right spring forces should differ on non-nominal envs.
        diff = (env.spring_forces[:, 0] - env.spring_forces[:, 1]).abs()
        self.assertGreater(diff.max().item(), 10.0)
        self.assertGreaterEqual(env.spring_forces.min().item(), 200.0 * 0.40 - 1e-4)
        self.assertLessEqual(env.spring_forces.max().item(), 200.0 * 1.15 + 1e-4)
        # Wheel armature randomized around 0.005 * [0.5, 1.6].
        self.assertGreaterEqual(robot.wheel_arm.min().item(), 0.0025 - 1e-6)
        self.assertLessEqual(robot.wheel_arm.max().item(), 0.0080 + 1e-6)
        self.assertGreater((robot.wheel_arm[:, 0] - robot.wheel_arm[:, 1]).abs().max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
