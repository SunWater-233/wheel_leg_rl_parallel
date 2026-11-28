import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from configs.experiment import TRAINING_DIR, load_config
from envs.terrain import ScutGroundLanes
from robots.wheel_leg_locomotion import WheelLegLocomotionEnv


class TerrainPreviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config(TRAINING_DIR / 'configs/archive/v02_complex_terrain_bars_finetune.toml')
        cls.lanes = ScutGroundLanes(config['task']['locomotion'])

    @patch('tasks.terrains.lanes.gs.morphs.Terrain', side_effect=lambda **kwargs: kwargs)
    def test_preview_keeps_height_samples_at_the_original_world_coordinates(self, _terrain):
        lanes = self.lanes
        width = round(lanes.lane_width / lanes.spacing)
        bars = lanes.surfaces.index('bars') * lanes.variants + lanes.variants - 1
        for lane in (0, bars, len(lanes.centers) - 1):
            with self.subTest(lane=lane):
                preview = lanes.morph(True, lane)
                x, y, z = preview['pos']
                self.assertEqual((x, z), (-lanes.half_length, 0.0))
                self.assertEqual(preview['horizontal_scale'], lanes.spacing)
                self.assertEqual(preview['vertical_scale'], lanes.vertical_scale)
                first = round(y / lanes.spacing)
                start = lane * width - first
                end = start + width + 1
                self.assertGreaterEqual(start, 0)
                self.assertLessEqual(end, preview['height_field'].shape[1])
                np.testing.assert_array_equal(preview['height_field'][:, start:end],
                                              lanes.field[:, lane * width:(lane + 1) * width + 1])
                self.assertLess(preview['height_field'].shape[1], lanes.field.shape[1])

    @patch('tasks.terrains.lanes.gs.morphs.Terrain', side_effect=lambda **kwargs: kwargs)
    def test_default_morph_still_uses_the_full_course(self, _terrain):
        lanes = self.lanes
        full = lanes.morph(False)
        self.assertEqual(full['pos'], (-lanes.half_length, 0.0, 0.0))
        np.testing.assert_array_equal(full['height_field'], lanes.field)

    def test_free_drive_crosses_lanes_but_keeps_the_map_boundary(self):
        lanes = self.lanes
        env = SimpleNamespace(lanes=lanes, free_drive=True, surface=torch.zeros(6, dtype=torch.long),
                              spawn_y=torch.full((6,), lanes.centers[2]))
        pos = torch.tensor([[-4.2, lanes.centers[2], 0.2], [-4.2, lanes.centers[3], 0.2],
                            [0.0, lanes.width + 0.1, 0.2], [lanes.half_length, lanes.centers[2], 0.2],
                            [0.0, lanes.centers[2], 0.2], [0.0, -0.1, 0.2]])
        ground = torch.tensor([0., 0., 0., 0., float('nan'), 0.])
        check = WheelLegLocomotionEnv._terrain_out_of_bounds
        self.assertEqual(check(env, pos, ground).tolist(), [False, False, True, True, True, True])
        env.free_drive = False
        self.assertEqual(check(env, pos, ground).tolist(), [False, True, True, True, True, True])

    def test_free_drive_does_not_time_out(self):
        env = SimpleNamespace(free_drive=True, episode_length_buf=torch.tensor([1000, 10000]))
        self.assertEqual(WheelLegLocomotionEnv._episode_timeout(env).tolist(), [False, False])
