"""Backward-compatible shim for `scripts.convert_wheelbipe`."""

from scripts.convert_wheelbipe import *  # noqa: F401, F403
from scripts.convert_wheelbipe import main

if __name__ == "__main__":
    main()
