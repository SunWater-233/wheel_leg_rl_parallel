"""Backward-compatible shim for `eval.transition`."""

from eval.transition import *  # noqa: F401, F403
from eval.transition import main

if __name__ == "__main__":
    main()
