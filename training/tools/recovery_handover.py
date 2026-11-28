"""Backward-compatible shim for `eval.recovery_handover`."""

from eval.recovery_handover import *  # noqa: F401, F403
from eval.recovery_handover import main

if __name__ == "__main__":
    main()
