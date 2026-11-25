"""Unit tests for firmware-aligned keyboard command slope planner."""

import unittest

from tools.command_slope import (
    FIRMWARE_LEG_LENGTH_SLOPE_SPEED,
    KeyboardCommandSlope,
    SLOPE_PLANNING_FIRST,
    SLOPE_REAL_FIRST,
    Slope,
)


class CommandSlopeTest(unittest.TestCase):
    def test_planning_first_accelerates_and_decelerates_with_separate_rates(self):
        slope = Slope(increase_rate=2.0, decrease_rate=4.0, slope_first=SLOPE_PLANNING_FIRST)
        dt = 0.02
        # Accelerate from 0 -> 1.0: +0.04 per tick, ignoring now_real
        self.assertAlmostEqual(slope.calculate(1.0, dt, now_real=0.8), 0.04, places=6)
        self.assertAlmostEqual(slope.calculate(1.0, dt, now_real=0.9), 0.08, places=6)
        # Brake toward 0.0: -0.08 per tick -> clamps at 0.0 without overshoot
        self.assertAlmostEqual(slope.calculate(0.0, dt, now_real=0.5), 0.0, places=6)

    def test_real_first_catches_up_when_measured_leads_plan_toward_target(self):
        slope = Slope(increase_rate=2.0, decrease_rate=4.0, slope_first=SLOPE_REAL_FIRST)
        dt = 0.02
        # Target = 2.0, plan = 0.0, real = 1.0 (in [0.0, 2.0]) -> starts from 1.0 + 0.04
        self.assertAlmostEqual(slope.calculate(2.0, dt, now_real=1.0), 1.04, places=6)
        # Braking to 0.0 when real = 0.5 (in [0.0, 1.04]) -> starts from 0.5 - 0.08
        self.assertAlmostEqual(slope.calculate(0.0, dt, now_real=0.5), 0.42, places=6)
        # If real is outside [target, current], do not jump past current
        self.assertAlmostEqual(slope.calculate(0.0, dt, now_real=1.5), 0.34, places=6)

    def test_real_first_negative_direction_and_reversal(self):
        slope = Slope(increase_rate=1.8, decrease_rate=3.0, slope_first=SLOPE_REAL_FIRST)
        dt = 0.02
        # Negative acceleration from 0 -> -2.0 with real = -0.5
        self.assertAlmostEqual(slope.calculate(-2.0, dt, now_real=-0.5), -0.536, places=6)
        # Reversing from -0.536 toward +2.0 uses decrease_rate (3.0 * 0.02 = 0.06)
        self.assertAlmostEqual(slope.calculate(2.0, dt, now_real=-0.536), -0.476, places=6)

    def test_zero_step_holds_current_value(self):
        slope = Slope(increase_rate=0.0, decrease_rate=0.0, slope_first=SLOPE_PLANNING_FIRST, initial=0.5)
        self.assertAlmostEqual(slope.calculate(1.0, 0.02), 0.5, places=6)

    def test_keyboard_command_slope_bundle_and_reset(self):
        bundle = KeyboardCommandSlope(vx_accel=1.8, vx_decel=3.0, wz_accel=8.0, wz_decel=12.0, initial_height=0.22)
        dt = 0.02
        vx, wz, h = bundle.calculate(2.5, 4.5, dt, vx_real=0.0, wz_real=2.0, height_target=0.30)
        self.assertAlmostEqual(vx, 1.8 * dt, places=6)
        # wz is planning-first: ignores wz_real=2.0
        self.assertAlmostEqual(wz, 8.0 * dt, places=6)
        # height is planning-first with FIRMWARE_LEG_LENGTH_SLOPE_SPEED
        self.assertAlmostEqual(h, 0.22 + FIRMWARE_LEG_LENGTH_SLOPE_SPEED * dt, places=6)

        # Decelerate wz uses wz_decel
        _, wz2 = bundle.calculate(0.0, 0.0, dt, vx_real=0.0)
        self.assertAlmostEqual(wz2, 0.0, places=6)

        bundle.reset(height=0.20)
        self.assertEqual(bundle.v_x_slope.now_planning, 0.0)
        self.assertEqual(bundle.w_z_slope.now_planning, 0.0)
        self.assertEqual(bundle.leg_length_slope.now_planning, 0.20)


if __name__ == "__main__":
    unittest.main()
