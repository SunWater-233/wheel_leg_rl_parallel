"""Backward-compatible shim for `eval.landing_brake`."""

from eval.landing_brake import *  # noqa: F401, F403
from eval.landing_brake import main

if __name__ == "__main__":
    main()
