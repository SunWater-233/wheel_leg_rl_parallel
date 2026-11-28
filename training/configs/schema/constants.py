"""Configuration schema constants."""

from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = TRAINING_DIR / "configs" / "flat.toml"
SNAPSHOT_NAME = "run_config.json"
MAX_FORWARD_SPEED = 2.5
YAW_REFERENCE = 11.0
HEIGHT_LIMITS = (0.17, 0.33)
MAX_HEIGHT_RATE = 1.2
MAX_FORWARD_ACCELERATION = 4.0
