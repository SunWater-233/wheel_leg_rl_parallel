"""Wheel Leg V0.2 robot hardware specification, Genesis adapter, closed-chain stance solver, and rescue."""

from robots.v02.adapter import WheelLegFlatEnv, _collision_model
from robots.v02.rescue import CadVmc, FallSelfRighting, FirmwareRecovery
from robots.v02.spec import (
    BODY_CONTACTS,
    HALF_TRACK,
    HOME_HEIGHT,
    HOME_JOINTS,
    LEG_ACTION_SCALE,
    LEG_BIAS_RAD,
    LEG_BIAS_TORQUE,
    LEG_JOINTS,
    LEG_KP,
    LEG_KV,
    LEG_SIGNS,
    MODEL_PATH,
    STANCE_SPAWN_GAP,
    WHEEL_JOINTS,
    WHEEL_KV,
    WHEEL_RADIUS,
    WHEEL_SIGNS,
    WHEEL_TARGET_LIMIT,
)
from robots.v02.stance import solve_stances

__all__ = [
    "BODY_CONTACTS",
    "HALF_TRACK",
    "HOME_HEIGHT",
    "HOME_JOINTS",
    "LEG_ACTION_SCALE",
    "LEG_BIAS_RAD",
    "LEG_BIAS_TORQUE",
    "LEG_JOINTS",
    "LEG_KP",
    "LEG_KV",
    "LEG_SIGNS",
    "MODEL_PATH",
    "STANCE_SPAWN_GAP",
    "WHEEL_JOINTS",
    "WHEEL_KV",
    "WHEEL_RADIUS",
    "WHEEL_SIGNS",
    "WHEEL_TARGET_LIMIT",
    "CadVmc",
    "FallSelfRighting",
    "FirmwareRecovery",
    "WheelLegFlatEnv",
    "_collision_model",
    "solve_stances",
]
