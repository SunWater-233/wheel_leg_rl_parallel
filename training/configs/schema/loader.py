"""TOML loading (with optional `extends` inheritance), snapshot persistence, and RSL-RL config builder."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tomllib

from configs.schema.constants import DEFAULT_CONFIG, SNAPSHOT_NAME, TRAINING_DIR
from configs.schema.validators import validate_settings


def _apply_overrides(settings, overrides, section=""):
    if not isinstance(overrides, dict):
        raise ValueError(f"Expected a table for {section or 'the experiment configuration'}")
    for key, value in overrides.items():
        name = f"{section}.{key}" if section else key
        if key not in settings:
            raise ValueError(f"Unknown setting: {name}")
        default = settings[key]
        if isinstance(default, dict):
            _apply_overrides(default, value, name)
        elif type(value) is type(default) or (type(default) is float and type(value) is int):
            settings[key] = value
        else:
            raise ValueError(f"Invalid value for {name}: {value!r}")


def _deep_merge_raw(base, child):
    merged = deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_raw(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_toml_with_extends(path: Path, seen: frozenset[Path] = frozenset()) -> dict:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Cyclic config extends chain involving {path}")
    with resolved.open("rb") as file:
        raw = tomllib.load(file)
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw
    if not isinstance(parent_ref, str) or not parent_ref.strip():
        raise ValueError(f"Invalid 'extends' value in {path}: {parent_ref!r}")
    parent_path = (resolved.parent / parent_ref).resolve()
    if not parent_path.is_file():
        alt = (TRAINING_DIR / parent_ref).resolve()
        if alt.is_file():
            parent_path = alt
        else:
            raise ValueError(f"Extended base config not found: {parent_ref} (referenced from {path})")
    parent_raw = _load_toml_with_extends(parent_path, seen | {resolved})
    return _deep_merge_raw(parent_raw, raw)


def _merge_config(overrides):
    with DEFAULT_CONFIG.open("rb") as file:
        settings = tomllib.load(file)
    if overrides is not None:
        _apply_overrides(settings, overrides)
    validate_settings(settings)
    return settings


def load_snapshot(path):
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    run = snapshot.pop("run", {})
    return _merge_config(snapshot), run


def load_config(path=None):
    if path is None:
        return _merge_config(None)
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_snapshot(path)[0]
    return _merge_config(_load_toml_with_extends(path))


def runner_config(ppo, *, actor_true_forward_speed=False, actor_history_steps=1):
    # RSL-RL removes class names and alters the configuration it receives.
    config = deepcopy(ppo)
    config.pop("max_action_std", None)
    config.pop("init_checkpoint_min_std", None)
    config["obs_groups"] = {"policy": ["policy"], "critic": ["critic"]}
    if actor_true_forward_speed:
        config["obs_groups"]["policy"].append("body_forward_speed")
    if actor_history_steps > 1:
        config["obs_groups"]["policy"].append("policy_history")
    config["policy"]["class_name"] = "ActorCritic"
    config["algorithm"]["class_name"] = "PPO"
    return config


def save_run_config(output, settings, args, robot_class, num_envs):
    model = args.model.resolve()
    snapshot = {
        **settings,
        "run": {
            "task": robot_class.task_name,
            "robot": args.robot,
            "robot_label": robot_class.label,
            "model": model.relative_to(TRAINING_DIR).as_posix() if model.is_relative_to(TRAINING_DIR) else str(model),
            "model_sha256": sha256(model.read_bytes()).hexdigest(),
            "seed": args.seed,
            "backend": args.backend,
            "num_envs": num_envs,
            "iterations": args.iterations,
            "physics_dt": robot_class.physics_dt,
            "decimation": robot_class.decimation,
            "control_dt": robot_class.physics_dt * robot_class.decimation,
            "init_at_random_ep_len": True,
        },
    }
    if args.resume is not None:
        snapshot["run"]["resume_checkpoint"] = str(args.resume.resolve())
    if args.init_checkpoint is not None:
        snapshot["run"]["init_checkpoint"] = str(args.init_checkpoint.resolve())
    contents = json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {output}. Choose a new --output path.")
    with (output / SNAPSHOT_NAME).open("x", encoding="utf-8") as file:
        file.write(contents)
