"""Backward-compatible shim for `eval.landing_runaway_probe`."""

from eval.landing_runaway_probe import *  # noqa: F401, F403
from eval.landing_runaway_probe import main

if __name__ == "__main__":
    main()
