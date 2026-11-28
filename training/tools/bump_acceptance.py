"""Backward-compatible shim for `eval.bump`."""

from eval.bump import *  # noqa: F401, F403
from eval.bump import main

if __name__ == "__main__":
    main()
