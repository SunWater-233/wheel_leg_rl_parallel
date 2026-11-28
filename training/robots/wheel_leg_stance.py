"""Backward-compatible re-export for `robots.v02.stance`."""

from robots.v02.spec import STANCE_SPAWN_GAP
from robots.v02.stance import solve_stances

__all__ = ["STANCE_SPAWN_GAP", "solve_stances"]
