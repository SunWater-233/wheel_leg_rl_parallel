"""Backward-compatible shim for `eval.flat_height`."""

from eval.flat_height import *  # noqa: F401, F403
from eval.flat_height import main

if __name__ == "__main__":
    main()
