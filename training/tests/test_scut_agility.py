import unittest
from types import SimpleNamespace

import torch

from configs.experiment import TRAINING_DIR, _merge_config, load_config, normalize_range_spec
from robots.scut_rough_reward import BrakingPitchCost, ScutRoughReward
from robots.special_mode_commands import SpecialModeUniformVelocityCommand
from robots.wheel_leg_locomotion import WheelLegLocomotionEnv


class ScutAgilityFeaturesTest(unittest.TestCase):
    def test_stage1_and_stage2_configs_load_and_validate(self):
        s1 = load_config(TRAINING_DIR / "configs/v02_scut_stage1_flat_speed.toml")
        s2 = load_config(TRAINING_DIR / "configs/v02_scut_stage2_terrain.toml")
        self.assertTrue(s1["task"]["locomotion"]["direct_velocity_commands"])
        self.assertTrue(s1["task"]["locomotion"]["special_mode_commands"]["enabled"])
        self.assertTrue(s1["task"]["domain_randomization"]["enabled"])
        self.assertTrue(s1["task"]["scut_rough_reward"]["tracking"]["vel_height_gate_enabled"])
        self.assertEqual(len(s1["task"]["locomotion"]["special_mode_commands"]["modes"]), 3)
        self.assertEqual(s2["task"]["locomotion"]["terrain"], "scut_ground")
        self.assertTrue(s2["task"]["locomotion"]["special_mode_commands"]["enabled"])
        self.assertEqual(s2["task"]["locomotion"]["special_mode_commands"]["modes"][0]["iteration_start"], 0)

    def test_config_rejects_conflicting_maneuvers_and_bad_gate(self):
        with self.assertRaises(ValueError):
            _merge_config({
                "task": {
                    "locomotion": {
                        "enabled": True,
                        "speed_maneuvers": True,
                        "special_mode_commands": {"enabled": True},
                    }
                }
            })
        with self.assertRaises(ValueError):
            _merge_config({
                "task": {
                    "scut_rough_reward": {
                        "tracking": {
                            "vel_height_gate_enabled": True,
                            "vel_height_gate_full_error": 0.10,
                            "vel_height_gate_zero_error": 0.05,
                        }
                    }
                }
            })

    def test_normalize_range_spec_merges_and_validates(self):
        merged = normalize_range_spec([[1.0, 2.0], [-2.5, -1.5], [1.8, 2.5]], limit=2.5)
        self.assertEqual(merged, ((-2.5, -1.5), (1.0, 2.5)))
        with self.assertRaises(ValueError):
            normalize_range_spec([[-3.0, -1.0]], limit=2.5)

    def test_special_mode_sampler_iteration_gating_and_readiness(self):
        torch.manual_seed(0)
        cfg = {
            "enabled": True,
            "rel_standing_envs": 0.10,
            "rel_zero_yaw_envs": 0.0,
            "resampling_time_range": [2.0, 4.0],
            "min_episode_time": 2.0,
            "require_stable": True,
            "stable_gravity_xy_max": 0.30,
            "stable_lin_vel_max": 3.0,
            "stable_ang_vel_max": 10.0,
            "normal_forward_range": [-1.0, 1.0],
            "normal_yaw_range": [-2.0, 2.0],
            "modes": [
                {
                    "name": "dash",
                    "rel_envs": 0.50,
                    "iteration_start": 100,
                    "iteration_end": -1,
                    "lin_vel_x": [[-2.5, -2.0], [2.0, 2.5]],
                    "ang_vel_z": [[-4.0, 4.0]],
                },
                {
                    "name": "spin",
                    "rel_envs": 0.50,
                    "iteration_start": 200,
                    "iteration_end": 300,
                    "lin_vel_x": [[-0.05, 0.05]],
                    "ang_vel_z": [[-9.0, -7.0], [7.0, 9.0]],
                },
            ],
        }
        n = 400
        sampler = SpecialModeUniformVelocityCommand(
            cfg,
            num_envs=n,
            device=torch.device("cpu"),
            default_forward_range=[-2.5, 2.5],
            default_yaw_range=[-10.0, 10.0],
        )
        ids = torch.arange(n)
        ep_time = torch.full((n,), 3.0)
        gravity = torch.tensor([[0.0, 0.0, -1.0]]).expand(n, 3).clone()
        lin_vel = torch.zeros((n, 3))
        ang_vel = torch.zeros((n, 3))

        # Before iteration_start=100, no special mode activates.
        sampler.set_training_iteration(50, log=False)
        motion, _ = sampler.sample(ids, episode_time=ep_time, gravity=gravity, lin_vel=lin_vel, ang_vel=ang_vel)
        self.assertEqual((sampler.special_mode_id >= 0).sum().item(), 0)
        self.assertLessEqual(motion[:, 0].abs().max().item(), 1.0 + 1e-5)

        # At iter 100, only dash activates, and unstable / early envs are held back.
        sampler.set_training_iteration(100, log=False)
        ep_time[:100] = 0.5  # too early
        gravity[100:200, 0] = 0.5  # tilted beyond 0.30
        motion, _ = sampler.sample(ids, episode_time=ep_time, gravity=gravity, lin_vel=lin_vel, ang_vel=ang_vel)
        self.assertEqual((sampler.special_mode_id[:200] >= 0).sum().item(), 0)
        dash_mask = sampler.special_mode_id == 0
        self.assertGreater(dash_mask[200:].sum().item(), 50)
        self.assertTrue((motion[dash_mask, 0].abs() >= 2.0 - 1e-5).all().item())

        # At iter 250, both dash and spin are active across ready envs; standing stays zero.
        ep_time[:] = 3.0
        gravity[:, 0] = 0.0
        sampler.set_training_iteration(250, log=False)
        motion, _ = sampler.sample(ids, episode_time=ep_time, gravity=gravity, lin_vel=lin_vel, ang_vel=ang_vel)
        self.assertTrue((motion[sampler.is_standing_env] == 0.0).all().item())
        spin_mask = sampler.special_mode_id == 1
        self.assertGreater(spin_mask.sum().item(), 100)
        self.assertTrue((motion[spin_mask, 1].abs() >= 7.0 - 1e-5).all().item())

        # At iter 300, spin expires (iteration_end=300).
        sampler.set_training_iteration(300, log=False)
        sampler.sample(ids, episode_time=ep_time, gravity=gravity, lin_vel=lin_vel, ang_vel=ang_vel)
        self.assertEqual((sampler.special_mode_id == 1).sum().item(), 0)

    def test_vel_height_gate_scales_forward_and_yaw_rewards_linearly(self):
        class DummyRobot:
            n_links = 3

            def get_dofs_velocity(self, dofs):
                return torch.zeros((4, 6))

            def get_dofs_control_force(self, dofs):
                return torch.zeros((4, 6))

            def get_quat(self):
                return torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)

        class DummyWheel:
            def get_anchor_pos(self):
                return torch.zeros((4, 3))

        cfg = load_config()["task"]
        tracking = cfg["scut_rough_reward"]["tracking"]
        tracking["vel_height_gate_enabled"] = True
        tracking["vel_height_gate_full_error"] = 0.03
        tracking["vel_height_gate_zero_error"] = 0.08
        tracking["forward_reward_weight"] = 2.0
        tracking["yaw_reward_weight"] = 1.0
        tracking["pitch_reward_weight"] = 0.0
        tracking["roll_reward_weight"] = 0.0
        tracking["height_reward_weight"] = 0.0
        tracking["height_error_scale"] = 0.0

        env = SimpleNamespace(
            cfg=cfg,
            num_envs=4,
            control_dt=0.02,
            robot=DummyRobot(),
            action_dofs=[0, 1, 2, 3, 4, 5],
            wheels=[DummyWheel(), DummyWheel()],
            wheel_links=[1, 2],
            wheel_direction=torch.ones(2),
            wheel_radius=0.06,
            last_actions=torch.zeros((4, 6)),
            commands=torch.tensor([
                [1.0, 2.0, 0.25],
                [1.0, 2.0, 0.25],
                [1.0, 2.0, 0.25],
                [1.0, 2.0, 0.25],
            ]),
            body_height_offset=0.0,
            # Height errors: 0.00 (gate=1), 0.03 (gate=1), 0.055 (gate=0.5), 0.10 (gate=0)
            body_clearance=torch.tensor([0.25, 0.28, 0.305, 0.35]),
            measured_heights=torch.full((4, 2), 0.25),
            surface=torch.zeros(4, dtype=torch.long),
            heading_hold_enabled=False,
        )
        reward_fn = ScutRoughReward(env, airborne_minimum_height=None)
        lin_vel = torch.tensor([[1.0, 0.0, 0.0]] * 4)
        ang_vel = torch.tensor([[0.0, 0.0, 2.0]] * 4)
        gravity = torch.tensor([[0.0, 0.0, -1.0]] * 4)
        contact = torch.full((4, 3, 3), 0.0)
        contact[:, 1:, 2] = 50.0  # wheels loaded
        fallen = torch.zeros(4, dtype=torch.bool)
        actions = torch.zeros((4, 6))

        rew = reward_fn(lin_vel, ang_vel, gravity, contact, fallen, actions)
        # Perfect forward (2.0) + perfect yaw (1.0) = 3.0 scaled by [1.0, 1.0, 0.5, 0.0]
        torch.testing.assert_close(rew, torch.tensor([3.0, 3.0, 1.5, 0.0]), atol=1e-4, rtol=1e-4)

    def test_direct_velocity_commands_step_immediately_and_bypass_heading_in_special_mode(self):
        class DummyRobot:
            def get_quat(self):
                return torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        env = object.__new__(WheelLegLocomotionEnv)
        env.robot = DummyRobot()
        env.control_dt = 0.02
        env.half_track = 0.21865
        env.wheel_radius = 0.06
        env.wheel_target_limit = 2.9224 / 0.06
        env.auto_commands = True
        env.direct_velocity_commands = True
        env.heading_hold_enabled = True
        env.heading_target = torch.tensor([1.0, 1.0])
        env.heading_turning = torch.tensor([False, False])
        env.heading_gain = 2.5
        env.heading_deadband = 0.0
        env.heading_max_yaw_rate = 0.40
        env.recovery_fraction = 0.0
        env.manual_brake_acceleration = None
        env.height_low, env.height_high = 0.17, 0.33
        env.height_step = 0.01
        env.height_acceleration = 0.0
        env.height_rate = torch.zeros(2)
        env.measured_heights = torch.full((2, 2), 0.22)
        env.command_lower = torch.tensor([-2.5, -10.0, 0.17])
        env.command_upper = torch.tensor([2.5, 10.0, 0.33])
        env.commands = torch.tensor([[0.0, 0.0, 0.22], [0.0, 0.0, 0.22]])
        # Env 0: normal zero-yaw -> heading_hold corrects yaw (+0.40) and vx jumps directly to +2.5.
        # Env 1: special mode active -> skips heading_hold, keeps requested yaw=0.0, jumps directly to -2.5.
        env.goals = torch.tensor([[2.5, 0.0, 0.22], [-2.5, 0.0, 0.22]])
        env.motion_step = torch.tensor([[0.01, 0.48], [0.01, 0.48]])
        env.special_command_sampler = SimpleNamespace(
            is_special_active=lambda: torch.tensor([False, True])
        )

        env._advance_commands(torch.zeros((2, 3)), torch.zeros((2, 3)))
        # Env 0 jumped from 0 to ~2.5 in a single tick (slew ramp completely bypassed) and applied heading hold.
        self.assertGreater(env.commands[0, 0].item(), 2.4)
        self.assertGreater(env.commands[0, 1].item(), 0.35)
        # Env 1 jumped from 0 to -2.5 directly and bypassed heading hold (yaw stayed 0.0).
        self.assertAlmostEqual(env.commands[1, 0].item(), -2.5, places=4)
        self.assertAlmostEqual(env.commands[1, 1].item(), 0.0, places=5)
        self.assertAlmostEqual(env.heading_target[1].item(), 0.0, places=5)

    def test_flat_only_height_gate_and_decoupled_spin_yaw_scale(self):
        class DummyRobot:
            n_links = 3

            def get_dofs_velocity(self, dofs):
                return torch.zeros((3, 6))

            def get_dofs_control_force(self, dofs):
                return torch.zeros((3, 6))

            def get_quat(self):
                return torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)

        class DummyWheel:
            def get_anchor_pos(self):
                return torch.zeros((3, 3))

        cfg = load_config()["task"]
        tracking = cfg["scut_rough_reward"]["tracking"]
        tracking["vel_height_gate_enabled"] = True
        tracking["vel_height_gate_flat_only"] = True
        tracking["vel_height_gate_full_error"] = 0.03
        tracking["vel_height_gate_zero_error"] = 0.08
        tracking["forward_reward_weight"] = 2.0
        tracking["yaw_reward_weight"] = 0.0
        tracking["pitch_reward_weight"] = 0.0
        tracking["roll_reward_weight"] = 0.0
        tracking["height_reward_weight"] = 0.0
        tracking["height_error_scale"] = 0.0
        tracking["yaw_error_scale"] = 0.50
        tracking["spin_yaw_threshold"] = 4.5
        tracking["spin_yaw_error_scale"] = 0.18

        # Env 0: flat (surface=0) with 0.10m height error -> gate=0 -> forward reward=0.
        # Env 1: obstacle (surface=3) with 0.10m height error -> flat_only bypasses gate -> forward reward=2.0.
        # Env 2: flat spin (yaw_target=8.0 > 4.5, yaw_error=4.0) -> uses spin_yaw_error_scale=0.18 instead of 0.50.
        env = SimpleNamespace(
            cfg=cfg,
            num_envs=3,
            control_dt=0.02,
            robot=DummyRobot(),
            action_dofs=[0, 1, 2, 3, 4, 5],
            wheels=[DummyWheel(), DummyWheel()],
            wheel_links=[1, 2],
            wheel_direction=torch.ones(2),
            wheel_radius=0.06,
            last_actions=torch.zeros((3, 6)),
            commands=torch.tensor([
                [1.0, 2.0, 0.25],
                [1.0, 2.0, 0.25],
                [1.0, 8.0, 0.25],
            ]),
            body_height_offset=0.0,
            body_clearance=torch.tensor([0.35, 0.35, 0.35]),
            measured_heights=torch.full((3, 2), 0.25),
            surface=torch.tensor([0, 3, 0], dtype=torch.long),
            heading_hold_enabled=False,
        )
        reward_fn = ScutRoughReward(env, airborne_minimum_height=None)
        lin_vel = torch.tensor([[1.0, 0.0, 0.0]] * 3)
        ang_vel = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
        gravity = torch.tensor([[0.0, 0.0, -1.0]] * 3)
        contact = torch.zeros((3, 3, 3))
        contact[:, 1:, 2] = 50.0
        rew = reward_fn(lin_vel, ang_vel, gravity, contact, torch.zeros(3, dtype=torch.bool), torch.zeros((3, 6)))
        # Env 0: forward gated to 0, yaw_error=2.0 with scale=0.50 -> -(1.0)^2 = -1.0
        # Env 1: forward=2.0 (ungated on terrain!), yaw_error=2.0 with scale=0.50 -> 2.0 - 1.0 = +1.0
        # Env 2: forward=0.0 (gated on flat), yaw_error=4.0 with spin_scale=0.18 -> -(0.72)^2 = -0.5184
        torch.testing.assert_close(rew, torch.tensor([-1.0, 1.0, -0.5184]), atol=1e-4, rtol=1e-4)

    def test_braking_pitch_cost_handles_instant_step_stop_with_physical_envelope(self):
        cfg = load_config()["task"]["scut_rough_reward"]["braking"]
        cfg["enabled"] = True
        commands = torch.zeros((2, 3))
        cost_fn = BrakingPitchCost(commands, cfg, dt=0.02)
        eligible = torch.tensor([True, True])
        zero = torch.zeros(2)
        # Step 0: cruising at 2.0 m/s
        cost_fn(torch.tensor([2.0, 2.0]), torch.tensor([2.0, 2.0]), torch.tensor([2.0, 2.0]), zero, zero, eligible)
        # Step 1: Env 0 gets instantaneous direct step command (forward=0.0, goal=0.0);
        # Env 1 gets 4.0 m/s^2 slewed command (forward=1.92, goal=0.0).
        # Both should follow the same 4.0 m/s^2 physical reference envelope (1.92 m/s), NOT jump to 0.0 in 1 tick!
        c1 = cost_fn(torch.tensor([0.0, 1.92]), torch.tensor([0.0, 0.0]), torch.tensor([1.92, 1.92]), zero, zero, eligible)
        self.assertAlmostEqual(cost_fn.reference_forward[0].item(), 1.92, places=5)
        self.assertAlmostEqual(cost_fn.reference_forward[1].item(), 1.92, places=5)
        self.assertEqual(cost_fn.direction[0].item(), 1.0)
        torch.testing.assert_close(c1[0], c1[1], atol=1e-6, rtol=1e-6)
        self.assertEqual(cost_fn.distance_error[0].item(), 0.0)

    def test_stage2_scut_aligned_terrain_commands_curriculum_and_heights(self):
        s2 = load_config(TRAINING_DIR / "configs/v02_scut_stage2_terrain.toml")
        loco = s2["task"]["locomotion"]
        self.assertTrue(loco["terrain_commands"]["enabled"])
        self.assertTrue(loco["terrain_curriculum"]["enabled"])
        self.assertTrue(loco["events"]["enabled"])
        self.assertTrue(s2["task"]["scut_rough_reward"]["tracking"]["vel_height_gate_flat_only"])
        self.assertTrue(s2["task"]["scut_rough_reward"]["braking"]["enabled"])

        # Verify per-terrain height sampling keeps obstacle heights within safe bounds while preserving flat [0.17, 0.33].
        env = object.__new__(WheelLegLocomotionEnv)
        env.device = torch.device("cpu")
        env.lanes = SimpleNamespace(variants=3)
        env.height_low, env.height_high = 0.17, 0.33
        env.height_sampling_floor = 0.17
        env.height_nominal = 0.20
        env.height_max_probability = 0.12
        env.obstacle_height_range = (0.18, 0.30)
        env.hard_obstacle_height_range = (0.18, 0.27)
        env.hard_obstacle_surfaces = torch.tensor([False, False, True])  # 0=flat, 1=slope, 2=bars
        env.surface = torch.tensor([0] * 200 + [1] * 200 + [2] * 200)
        heights = env._sample_heights_for_envs(torch.arange(600))
        self.assertLessEqual(heights[:200].min().item(), 0.17 + 1e-6)
        self.assertGreaterEqual(heights[200:400].min().item(), 0.18 - 1e-6)
        self.assertLessEqual(heights[200:400].max().item(), 0.30 + 1e-6)
        self.assertGreaterEqual(heights[400:].min().item(), 0.18 - 1e-6)
        self.assertLessEqual(heights[400:].max().item(), 0.27 + 1e-6)


if __name__ == "__main__":
    unittest.main()
