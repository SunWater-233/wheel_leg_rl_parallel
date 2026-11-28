"""Backward-compatible shim for `eval.complex_terrain`."""

from eval.complex_terrain import *  # noqa: F401, F403
from eval.complex_terrain import main

if __name__ == "__main__":
    main()
