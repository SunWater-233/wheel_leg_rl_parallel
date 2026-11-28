"""Firmware-aligned command slope (ramp) planner for keyboard driving.

Mirrors ``slope.c`` and ``wheel_leg.c`` from the real chassis firmware:
- ``SLOPE_REAL_FIRST`` (used for ``v_x_slope``): starts from the previous planned output,
  but catches up to ``now_real`` whenever the measured value already lies between the
  previous plan and the target.
- ``SLOPE_PLANNING_FIRST`` (used for ``w_z_slope`` and ``leg_length_slope``): always
  advances from the previous planned output.
- Accelerating (increasing ``|value|``) uses ``increase_rate * dt``; decelerating or
  reversing uses ``decrease_rate * dt``.
"""

from __future__ import annotations

from math import isfinite

SLOPE_REAL_FIRST = "real_first"
SLOPE_PLANNING_FIRST = "planning_first"

# Real chassis reference limits from robot_param.h.
FIRMWARE_MAX_ACC = 4.0
FIRMWARE_MAX_WZ_ACC = 24.0
FIRMWARE_LEG_LENGTH_SLOPE_SPEED = 1.2


def _move_towards(current: float, target: float, max_step: float) -> float:
    delta = target - current
    if delta > max_step:
        return current + max_step
    if delta < -max_step:
        return current - max_step
    return target


class Slope:
    """Scalar target ramp matching firmware ``Slope_t`` / ``Slope_Calculate``."""

    def __init__(
        self,
        increase_rate: float,
        decrease_rate: float | None = None,
        *,
        slope_first: str = SLOPE_PLANNING_FIRST,
        initial: float = 0.0,
    ) -> None:
        if slope_first not in (SLOPE_REAL_FIRST, SLOPE_PLANNING_FIRST):
            raise ValueError(f"Unsupported slope_first mode: {slope_first!r}")
        self.increase_rate = float(increase_rate)
        self.decrease_rate = float(increase_rate if decrease_rate is None else decrease_rate)
        self.slope_first = slope_first
        self.now_planning = float(initial)

    def reset(self, value: float = 0.0) -> None:
        self.now_planning = float(value)

    def calculate(self, target: float, dt: float, now_real: float = 0.0) -> float:
        target = float(target)
        dt = float(dt)
        current = self.now_planning

        if self.slope_first == SLOPE_REAL_FIRST and isfinite(now_real):
            now_real = float(now_real)
            if (target >= current and current <= now_real <= target) or (
                target <= current and target <= now_real <= current
            ):
                current = now_real

        accelerating = (current >= 0.0 and target > current) or (current <= 0.0 and target < current)
        rate = self.increase_rate if accelerating else self.decrease_rate
        step = rate * dt
        if step > 0.0:
            self.now_planning = _move_towards(current, target, step)
        else:
            self.now_planning = current
        return self.now_planning


class KeyboardCommandSlope:
    """Bundled vx / wz / leg-height slope planner matching ``WheelLeg_Init``."""

    def __init__(
        self,
        *,
        vx_accel: float = 1.8,
        vx_decel: float = 3.0,
        wz_accel: float = 8.0,
        wz_decel: float | None = None,
        height_slope_speed: float = FIRMWARE_LEG_LENGTH_SLOPE_SPEED,
        initial_height: float = 0.20,
    ) -> None:
        self.v_x_slope = Slope(
            increase_rate=vx_accel,
            decrease_rate=vx_decel,
            slope_first=SLOPE_REAL_FIRST,
            initial=0.0,
        )
        self.w_z_slope = Slope(
            increase_rate=wz_accel,
            decrease_rate=wz_accel if wz_decel is None else wz_decel,
            slope_first=SLOPE_PLANNING_FIRST,
            initial=0.0,
        )
        self.leg_length_slope = Slope(
            increase_rate=height_slope_speed,
            decrease_rate=height_slope_speed,
            slope_first=SLOPE_PLANNING_FIRST,
            initial=initial_height,
        )

    def reset(self, *, vx: float = 0.0, wz: float = 0.0, height: float | None = None) -> None:
        self.v_x_slope.reset(vx)
        self.w_z_slope.reset(wz)
        if height is not None:
            self.leg_length_slope.reset(height)

    def calculate(
        self,
        vx_target: float,
        wz_target: float,
        dt: float,
        *,
        vx_real: float = 0.0,
        wz_real: float = 0.0,
        height_target: float | None = None,
        height_real: float = 0.0,
    ) -> tuple[float, float] | tuple[float, float, float]:
        vx = self.v_x_slope.calculate(vx_target, dt, vx_real)
        wz = self.w_z_slope.calculate(wz_target, dt, wz_real)
        if height_target is None:
            return vx, wz
        height = self.leg_length_slope.calculate(height_target, dt, height_real)
        return vx, wz, height
