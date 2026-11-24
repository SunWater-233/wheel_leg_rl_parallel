import unittest

import numpy as np

from configs.experiment import TRAINING_DIR, load_config
from envs.terrain import TransitionLanes


class TransitionTerrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(TRAINING_DIR / "configs/v02_random_transitions.toml")
        cls.settings = cls.config["task"]["locomotion"]
        cls.lanes = TransitionLanes(cls.settings)

    def test_seeded_sequences_and_steps(self):
        lanes = self.lanes
        np.testing.assert_array_equal(lanes.field, TransitionLanes(self.settings).field)
        held_out = TransitionLanes({**self.settings, "terrain_seed": self.settings["terrain_seed"] + 1})
        self.assertFalse(np.array_equal(lanes.field, held_out.field))
        self.assertEqual(lanes.surfaces, ("flat", "transitions"))
        self.assertTrue(all(len(set(sequence)) == 4 for sequence in lanes.segment_families))
        for index, joins in enumerate(lanes.seam_steps, lanes.variants):
            self.assertEqual(lanes.forward_only[index], min(joins) < -0.1)

    def test_high_speed_runup_shares_spawn_elevation(self):
        lanes = self.lanes
        row = lambda x: round((x + lanes.half_length) / lanes.spacing)
        for i in range(lanes.variants, len(lanes.centers)):
            col = round(lanes.centers[i] / lanes.spacing)
            self.assertAlmostEqual(lanes.field[row(-9), col] * lanes.vertical_scale, lanes.spawn_z[i])
            self.assertAlmostEqual(lanes.field[row(9), col] * lanes.vertical_scale, lanes.reverse_spawn_z[i])

    def test_speed_levels_cover_both_command_limits(self):
        speeds = self.settings["command_speed_levels"]
        self.assertEqual(self.config["task"]["forward_command"], [-speeds[-1], speeds[-1]])
        self.assertEqual(speeds[0], 0.10)
        self.assertEqual(self.settings["terrain_forward_limit"], 0.0)
