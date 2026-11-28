from types import SimpleNamespace
import unittest

import torch

from configs.experiment import TRAINING_DIR, load_config
from robots.wheel_leg_locomotion import WheelLegLocomotionEnv


class FlatHeightCurriculumTest(unittest.TestCase):
    def test_flat_training_keeps_joint_limits_and_uses_wheel_delay(self):
        settings = load_config(TRAINING_DIR / "configs/archive/v02_sim2real_flat_height.toml")["task"]["locomotion"]
        self.assertEqual(settings["terrain"], "plane")
        self.assertEqual(settings["leg_torque_limit"], 40.0)
        self.assertEqual(settings["wheel_target_delay_steps"], 5)
        defaults = load_config()["task"]["locomotion"]
        self.assertEqual(defaults["wheel_target_delay_steps"], 0)
        self.assertEqual(defaults["wheel_torque_limit"], 5.0)

    def test_repeated_nominal_in_sequence_does_not_skip_next_height(self):
        env = SimpleNamespace(
            _sample_heights=lambda count: torch.full((count,), 0.28),
            height_maneuvers=True,
            height_maneuver_sequence=torch.tensor([0.20, 0.24, 0.20, 0.33]),
            height_sequence_index=torch.zeros(3, dtype=torch.long),
            height_mode=6,
            episode_steps=torch.zeros(3, dtype=torch.long),
            surface=torch.tensor([0, 0, 1]),
            maneuver=torch.tensor([6, 0, 6]),
            goals=torch.tensor([[0.0, 0.0, 0.20], [0.0, 0.0, 0.21], [0.0, 0.0, 0.25]]),
            stationary_episode_fraction=0.0,
            low_rough_slope=None,
            forward_drop=False,
            speed_maneuvers=False,
        )
        for height in (0.20, 0.24, 0.20, 0.33, 0.20):
            WheelLegLocomotionEnv._sample_height_goals(env, torch.arange(3))
            torch.testing.assert_close(env.goals[:, 2], torch.tensor([height, 0.21, 0.28]))
            env.episode_steps += 1

