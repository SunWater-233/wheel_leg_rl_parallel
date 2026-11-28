"""Backward-compatible shim for `eval.terrain_obstacle`."""

from eval.terrain_obstacle import *  # noqa: F401, F403
from eval.terrain_obstacle import main

if __name__ == "__main__":
    main()
