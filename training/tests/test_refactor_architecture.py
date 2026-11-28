"""Regression tests for refactored package boundaries, config inheritance, and V0.2 deploy parity."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from configs.schema import TRAINING_DIR, load_config
from eval.harness import configure_eval_locomotion
from rl.checkpoint import CheckpointBatch
from robots.v02 import spec as v02_spec


class RefactorArchitectureTest(unittest.TestCase):
    def test_toml_extends_inheritance_merges_and_validates(self):
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            parent = tmp_dir / "parent.toml"
            child = tmp_dir / "child.toml"
            parent.write_text(
                "[task]\n"
                "forward_command = [-1.5, 1.5]\n"
                "[task.locomotion]\n"
                "enabled = true\n"
                "actor_history_steps = 5\n"
                "[ppo.algorithm]\n"
                "learning_rate = 0.0002\n",
                encoding="utf-8",
            )
            child.write_text(
                'extends = "parent.toml"\n'
                "[task.locomotion]\n"
                "leg_torque_limit = 40.0\n"
                "[ppo.algorithm]\n"
                "entropy_coef = 0.002\n",
                encoding="utf-8",
            )
            cfg = load_config(child)
            self.assertEqual(cfg["task"]["forward_command"], [-1.5, 1.5])
            self.assertEqual(cfg["task"]["locomotion"]["actor_history_steps"], 5)
            self.assertEqual(cfg["task"]["locomotion"]["leg_torque_limit"], 40.0)
            self.assertAlmostEqual(cfg["ppo"]["algorithm"]["learning_rate"], 0.0002)
            self.assertAlmostEqual(cfg["ppo"]["algorithm"]["entropy_coef"], 0.002)

    def test_toml_extends_rejects_cycles_and_missing_parent(self):
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            a = tmp_dir / "a.toml"
            b = tmp_dir / "b.toml"
            a.write_text('extends = "b.toml"\n', encoding="utf-8")
            b.write_text('extends = "a.toml"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(a)

            missing = tmp_dir / "missing.toml"
            missing.write_text('extends = "does_not_exist.toml"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(missing)

    def test_v02_spec_matches_deploy_export_policy_constants(self):
        import importlib.util

        export_path = TRAINING_DIR.parent / "deploy" / "host" / "export_policy.py"
        if not export_path.is_file():
            self.skipTest("deploy/host/export_policy.py not present")
        module_spec = importlib.util.spec_from_file_location("_deploy_export_policy", export_path)
        mod = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(mod)

        self.assertEqual(v02_spec.HOME_JOINTS, mod.HOME_JOINTS)
        self.assertEqual(v02_spec.LEG_JOINTS, mod.LEG_JOINTS)
        np.testing.assert_allclose(v02_spec.LEG_SIGNS, mod.LEG_SIGNS)
        np.testing.assert_allclose(v02_spec.LEG_BIAS_RAD, mod.LEG_BIAS_RAW_RAD)
        self.assertAlmostEqual(v02_spec.WHEEL_RADIUS, mod.WHEEL_RADIUS)
        self.assertAlmostEqual(v02_spec.WHEEL_TARGET_LIMIT, mod.WHEEL_TARGET_LIMIT)
        self.assertAlmostEqual(v02_spec.STANCE_SPAWN_GAP, mod.STANCE_SPAWN_GAP)

    def test_eval_harness_locomotion_alignment(self):
        batch = CheckpointBatch(
            paths=(Path("a.pt"), Path("b.pt")),
            snapshots=(({}, {}), ({}, {})),
            speed_inputs=(False, True),
            history_steps=(1, 5),
        )
        locomotion = {"actor_true_forward_speed": False, "actor_history_steps": 1}
        configure_eval_locomotion(locomotion, batch)
        self.assertTrue(locomotion["actor_true_forward_speed"])
        self.assertEqual(locomotion["actor_history_steps"], 5)

    def test_backward_compatible_reexports_resolve_to_canonical_symbols(self):
        from configs.experiment import load_config as legacy_load_config
        from envs.flat import FlatEnv as LegacyFlatEnv
        from envs.terrain import ScutGroundLanes as LegacyScutGroundLanes
        from main import add_actor_history_inputs as main_add_hist
        from main import load_checkpoint as main_load_ckpt
        from rl.checkpoint import add_actor_history_inputs, load_checkpoint
        from robots.domain_randomization import randomize_dynamics as legacy_dr
        from robots.scut_rough_reward import ScutRoughReward as LegacyReward
        from robots.special_mode_commands import SpecialModeUniformVelocityCommand as LegacySpecial
        from robots.wheel_leg import WheelLegFlatEnv as LegacyV02
        from robots.wheel_leg_locomotion import WheelLegLocomotionEnv as LegacyLoco
        from tasks.base_env import FlatEnv
        from tasks.commands.special_mode import SpecialModeUniformVelocityCommand
        from tasks.locomotion_env import WheelLegLocomotionEnv
        from tasks.randomization.dynamics import randomize_dynamics
        from tasks.rewards.scut_rough import ScutRoughReward
        from tasks.terrains import ScutGroundLanes
        from tools.bump_acceptance import BumpAcceptance as LegacyBump
        from tools.command_slope import KeyboardCommandSlope as LegacySlope
        from eval.bump import BumpAcceptance
        from viz.command_slope import KeyboardCommandSlope

        self.assertIs(legacy_load_config, load_config)
        self.assertIs(LegacyFlatEnv, FlatEnv)
        self.assertIs(LegacyScutGroundLanes, ScutGroundLanes)
        self.assertIs(main_load_ckpt, load_checkpoint)
        self.assertIs(main_add_hist, add_actor_history_inputs)
        self.assertIs(legacy_dr, randomize_dynamics)
        self.assertIs(LegacyReward, ScutRoughReward)
        self.assertIs(LegacySpecial, SpecialModeUniformVelocityCommand)
        self.assertIs(LegacyLoco, WheelLegLocomotionEnv)
        self.assertIs(LegacyBump, BumpAcceptance)
        self.assertIs(LegacySlope, KeyboardCommandSlope)
        self.assertEqual(LegacyV02.wheel_radius, v02_spec.WHEEL_RADIUS)


if __name__ == "__main__":
    unittest.main()
