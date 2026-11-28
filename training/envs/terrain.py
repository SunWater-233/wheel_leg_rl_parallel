"""Backward-compatible re-export for `tasks.terrains`."""

import genesis as gs

from tasks.terrains import (
    SCUT_GROUND_COMPOUND_SURFACES,
    SCUT_GROUND_SINGLE_SIDE_SURFACES,
    SCUT_GROUND_SPEED_BUMP,
    SCUT_GROUND_SURFACES,
    SCUT_V1_SURFACES,
    SURFACES,
    SURFACE_STYLES,
    SURFACE_WEIGHTS,
    TERRAIN_LANES,
    ScutGroundLanes,
    ScutV1Lanes,
    TerrainLanes,
    TransitionLanes,
    raised_cosine_profile,
)

__all__ = [
    "SCUT_GROUND_COMPOUND_SURFACES",
    "SCUT_GROUND_SINGLE_SIDE_SURFACES",
    "SCUT_GROUND_SPEED_BUMP",
    "SCUT_GROUND_SURFACES",
    "SCUT_V1_SURFACES",
    "SURFACES",
    "SURFACE_STYLES",
    "SURFACE_WEIGHTS",
    "TERRAIN_LANES",
    "ScutGroundLanes",
    "ScutV1Lanes",
    "TerrainLanes",
    "TransitionLanes",
    "gs",
    "raised_cosine_profile",
]
