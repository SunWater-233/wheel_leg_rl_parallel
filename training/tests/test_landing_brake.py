import unittest

import torch

from configs.experiment import TRAINING_DIR, load_config
from robots.scut_rough_reward import BrakingPitchCost


class LandingBrakeTest(unittest.TestCase):
    def setUp(self):
        self.dt = 0.02
        self.settings = load_config(TRAINING_DIR / 'configs/archive/v02_landing_brake_control.toml')[
            'task']['scut_rough_reward']['braking']
        self.cost = BrakingPitchCost(torch.zeros((1, 3)), self.settings, self.dt)

    def tick(self, forward, *, goal=0.0, measured=0.6, grounded=True, suspend=False):
        return self.cost(torch.tensor([forward]), torch.tensor([goal]), torch.tensor([measured]),
                         torch.zeros(1), torch.zeros(1), torch.tensor([grounded]),
                         suspended=torch.tensor([suspend])).item()

    def test_ground_only_clears_stop_on_liftoff(self):
        self.tick(0.6, goal=0.6)
        self.tick(0.55)
        self.assertEqual(self.cost.direction.item(), 1.0)
        self.tick(0.0, grounded=False)
        self.assertEqual(self.cost.direction.item(), 0.0)
        self.assertEqual(self.tick(0.0), 0.0)

    def test_airborne_stop_is_latched_without_airborne_cost(self):
        self.tick(0.6, goal=0.6)
        self.assertEqual(self.tick(0.55, grounded=False, suspend=True), 0.0)
        for _ in range(60):
            self.assertEqual(self.tick(0.0, grounded=False, suspend=True), 0.0)
        self.assertEqual(self.cost.direction.item(), 1.0)
        self.assertEqual(self.cost.recovery_elapsed.item(), 0.0)
        self.assertEqual(self.cost.distance_error.item(), 0.0)
        self.assertGreater(self.tick(0.0), 0.0)
        self.assertAlmostEqual(self.cost.recovery_elapsed.item(), self.dt)

    def test_new_command_or_reset_clears_latch(self):
        self.tick(-0.6, goal=-0.6)
        self.tick(-0.55, grounded=False, suspend=True)
        self.assertEqual(self.cost.direction.item(), -1.0)
        self.tick(-0.5, goal=0.5, grounded=False, suspend=True)
        self.assertEqual(self.cost.direction.item(), 0.0)
        self.tick(-0.45, grounded=False, suspend=True)
        self.cost.reset(torch.tensor([0]))
        self.assertEqual(self.cost.direction.item(), 0.0)
        self.assertEqual(self.cost.previous_forward.item(), 0.0)
