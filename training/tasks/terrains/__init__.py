"""Deterministic height-field terrain families and registries."""

from tasks.terrains.lanes import ScutGroundLanes, ScutV1Lanes, TerrainLanes
from tasks.terrains.registry import (
    SCUT_GROUND_COMPOUND_SURFACES,
    SCUT_GROUND_SINGLE_SIDE_SURFACES,
    SCUT_GROUND_SPEED_BUMP,
    SCUT_GROUND_SURFACES,
    SCUT_V1_SURFACES,
    SURFACE_STYLES,
    SURFACE_WEIGHTS,
    SURFACES,
    raised_cosine_profile,
)
from tasks.terrains.transitions import TransitionLanes

TERRAIN_LANES = {
    "mixed": TerrainLanes,
    "scut_v1": ScutV1Lanes,
    "scut_ground": ScutGroundLanes,
    "transitions": TransitionLanes,
}

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
    "raised_cosine_profile",
]
