"""Terrain surface identifiers, visual styles, and analytical profiles."""

import numpy as np


SURFACES = ("flat", "slope", "rough", "bars", "stairs")
SURFACE_WEIGHTS = (0.40, 0.25, 0.20, 0.075, 0.075)
SCUT_V1_SURFACES = SURFACES + (
    "inv_slope",
    "inv_stairs_low",
    "inv_stairs_high",
    "stairs_slow",
    "stairs_fast",
    "cliff",
)
SCUT_GROUND_SURFACES = SURFACES + ("inv_slope", "inv_stairs_low", "inv_stairs_high")
SCUT_GROUND_COMPOUND_SURFACES = ("rough_slope", "staggered_bars")
SCUT_GROUND_SINGLE_SIDE_SURFACES = ("single_side_bars_left", "single_side_bars_right")
SCUT_GROUND_SPEED_BUMP = "speed_bump"
SURFACE_STYLES = {
    "flat": ("平地", (116, 141, 160)),
    "slope": ("坡道", (112, 157, 123)),
    "rough": ("粗糙路", (161, 143, 113)),
    "bars": ("横条障碍", (182, 151, 91)),
    "stairs": ("细台阶", (145, 130, 174)),
    "inv_slope": ("凹坡", (95, 155, 158)),
    "inv_stairs_low": ("低沉台阶", (121, 139, 182)),
    "inv_stairs_high": ("高沉台阶", (145, 123, 169)),
    "stairs_slow": ("低速高台阶", (177, 132, 110)),
    "stairs_fast": ("高速高台阶", (169, 115, 117)),
    "cliff": ("断崖", (122, 144, 153)),
    "rough_slope": ("粗糙坡道", (119, 149, 107)),
    "staggered_bars": ("错位凸条", (185, 130, 90)),
    "single_side_bars_left": ("单侧凸条（左轮先过）", (174, 116, 86)),
    "single_side_bars_right": ("单侧凸条（右轮先过）", (177, 100, 83)),
    SCUT_GROUND_SPEED_BUMP: ("曲面减速带", (191, 144, 79)),
    "transitions": ("随机地形过渡", (132, 145, 121)),
}


def raised_cosine_profile(x, length):
    """Unit-height bump with zero height and slope at both ends."""
    return 0.5 * (1 + np.cos(np.pi * np.clip(2 * x / length, -1, 1)))
