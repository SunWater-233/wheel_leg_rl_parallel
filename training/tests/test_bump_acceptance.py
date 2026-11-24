
import unittest

from tools.bump_acceptance import BumpAcceptance, Sample


class BumpAcceptanceTest(unittest.TestCase):
    def feed(self, test, time, *, target=0.0, vx=0.0, pitch=0.0, roll=0.0, loaded=True,
             hit=False, beyond=False):
        test.update(Sample(time, target, vx, pitch, roll, loaded, hit, beyond))

    def start_brake(self):
        test = BumpAcceptance(approach_speed=0.8, reverse_speed=0.5)
        self.feed(test, 1.0, hit=True)
        self.feed(test, 1.2, beyond=True)
        self.assertEqual(test.command, 0.0)
        return test

    def test_pass_requires_holds_and_real_reverse_tracking(self):
        test = self.start_brake()
        for tick in range(61, 73):
            self.feed(test, tick / 50)
        self.assertEqual(test.command, 0.0)
        self.feed(test, 73 / 50)
        self.assertEqual(test.command, -0.5)
        for tick in range(74, 86):
            self.feed(test, tick / 50, target=-0.5, vx=-0.47)
        self.assertFalse(test.result()["passed"])
        self.feed(test, 86 / 50, target=-0.5, vx=-0.47)
        self.assertTrue(test.result()["passed"])

    def test_crossing_without_contact_does_not_start_recovery(self):
        test = BumpAcceptance(0.8, 0.5)
        for tick in range(1, 301):
            self.feed(test, tick / 50, beyond=True)
        self.assertTrue(test.finished)
        self.assertIsNone(test.result()["crossed_s"])
        self.assertFalse(test.result()["passed"])

    def test_forward_drift_after_crossing_cannot_pass_even_with_reverse_command(self):
        test = self.start_brake()
        for tick in range(61, 161):
            self.feed(test, tick / 50, vx=1.33, pitch=27.0)
        self.assertEqual(test.command, -0.5)
        for tick in range(161, 261):
            self.feed(test, tick / 50, target=-0.5, vx=1.33, pitch=27.0)
        result = test.result()
        self.assertTrue(test.finished)
        self.assertFalse(result["passed"])
        self.assertTrue(all(result[key] is None for key in ("upright_s", "stopped_s", "reverse_tracking_s")))

    def test_loading_or_roll_excursion_breaks_continuous_hold(self):
        test = self.start_brake()
        for tick in range(61, 73):
            self.feed(test, tick / 50)
        self.feed(test, 73 / 50, loaded=False)
        for tick in range(74, 86):
            self.feed(test, tick / 50)
        self.feed(test, 86 / 50, roll=12.0)
        self.assertIsNone(test.result()["stopped_s"])
        self.assertEqual(test.command, 0.0)

    def test_mirrored_speed_bumps_raise_only_the_selected_wheel_track(self):
        from tools.terrain_obstacle_acceptance import GRID_M, VERTICAL_SCALE_M, speed_bump_field

        field, (_, y_start, z_start) = speed_bump_field(0.04, 0.30)
        crest = field[len(field) // 2]

        def height_at(y):
            return crest[round((y - y_start) / GRID_M)] * VERTICAL_SCALE_M + z_start

        for track in (-0.22, 0.22, 2.22, 3.78):
            self.assertAlmostEqual(height_at(track), 0.04)
        for track in (1.78, 4.22):
            self.assertLess(height_at(track), 0.0)
        self.assertLess(field[0].max() * VERTICAL_SCALE_M + z_start, 0.0)

