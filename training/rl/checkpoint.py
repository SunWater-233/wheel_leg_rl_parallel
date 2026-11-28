"""Checkpoint loading, observation-contract validation, and warm-start weight padding hooks."""

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner
import torch

from configs.schema import SNAPSHOT_NAME, load_snapshot, runner_config
from robots.scut.adapter import WheelbipeFlatEnv
from tasks.locomotion_env import WheelLegLocomotionEnv


def add_actor_speed_input(module, state, prefix, *_):
    name = f"{prefix}actor.0.weight"
    old = state.get(name)
    weight = module.actor[0].weight
    if old is not None and old.shape == (weight.shape[0], weight.shape[1] - 1):
        state[name] = torch.nn.functional.pad(old, (0, 1))
        print("Initialized actor speed input with zero weights", flush=True)


def add_actor_history_inputs(module, state, prefix, *_):
    name = f"{prefix}actor.0.weight"
    old = state.get(name)
    weight = module.actor[0].weight
    if old is None or old.shape[0] != weight.shape[0]:
        return
    extra_dims = weight.shape[1] - old.shape[1]
    if extra_dims > 0:
        state[name] = torch.nn.functional.pad(old, (0, extra_dims))
        print(f"Initialized {extra_dims} actor history inputs with zero weights", flush=True)


def load_checkpoint(runner, path, *, load_optimizer, initialize_actor_speed=False, initialize_actor_history=False):
    def add_critic_inputs(module, state, prefix, *_):
        name = f"{prefix}critic.0.weight"
        old = state.get(name)
        weight = module.critic[0].weight
        extra_dims = runner.env.critic_terrain_dims
        if old is None or old.shape != (weight.shape[0], weight.shape[1] - extra_dims):
            return
        if load_optimizer:
            raise ValueError("Use --init-checkpoint to add terrain observations to an existing critic")
        state[name] = torch.nn.functional.pad(old, (0, extra_dims))
        print(f"Initialized {extra_dims} terrain critic inputs with zero weights", flush=True)

    with ExitStack() as stack:
        if initialize_actor_speed:
            stack.enter_context(runner.alg.policy.register_load_state_dict_pre_hook(add_actor_speed_input))
        if initialize_actor_history:
            if load_optimizer:
                raise ValueError("Use --init-checkpoint to add history observations to an existing actor")
            stack.enter_context(runner.alg.policy.register_load_state_dict_pre_hook(add_actor_history_inputs))
        if isinstance(runner.env, WheelLegLocomotionEnv) and runner.env.critic_scan_offsets is not None:
            stack.enter_context(runner.alg.policy.register_load_state_dict_pre_hook(add_critic_inputs))
        info = runner.load(str(path), load_optimizer=load_optimizer, map_location=str(runner.env.device)) or {}
    # Checkpoints before robot metadata was added only used the SCUT model.
    checkpoint_robot = info.get("robot", WheelbipeFlatEnv.label)
    if checkpoint_robot != runner.env.label:
        raise ValueError(f"Checkpoint was trained on {checkpoint_robot}; selected robot is {runner.env.label}")
    if info.get("task", "flat") != runner.env.task_name:
        raise ValueError("Checkpoint and selected configuration use different observation contracts")
    return info


@dataclass(frozen=True)
class CheckpointBatch:
    """Validated metadata for evaluating one or more locomotion checkpoints in a shared env."""

    paths: tuple[Path, ...]
    snapshots: tuple[tuple[dict, dict], ...]
    speed_inputs: tuple[bool, ...]
    history_steps: tuple[int, ...]

    @property
    def first_settings(self) -> dict:
        return self.snapshots[0][0]

    @property
    def first_run(self) -> dict:
        return self.snapshots[0][1]

    @property
    def any_speed_input(self) -> bool:
        return any(self.speed_inputs)

    @property
    def max_history_steps(self) -> int:
        return max(self.history_steps)


def load_checkpoint_snapshots(checkpoints, *, require_same_model=False):
    paths = tuple(Path(path) for path in checkpoints)
    snapshots = tuple(load_snapshot(path.with_name(SNAPSHOT_NAME)) for path in paths)
    speed_inputs = tuple(bool(cfg["task"]["locomotion"]["actor_true_forward_speed"]) for cfg, _ in snapshots)
    history_steps = tuple(int(cfg["task"]["locomotion"]["actor_history_steps"]) for cfg, _ in snapshots)
    if len({steps for steps in history_steps if steps > 1}) > 1:
        raise ValueError("Checkpoints with observation history must share the same actor_history_steps")
    if require_same_model:
        ref = snapshots[0][1]
        if any(meta.get("model_sha256") != ref.get("model_sha256") or meta.get("task") != "locomotion"
               for _, meta in snapshots):
            raise ValueError("Checkpoints must use the same locomotion robot model")
    return CheckpointBatch(
        paths=paths,
        snapshots=snapshots,
        speed_inputs=speed_inputs,
        history_steps=history_steps,
    )


def build_inference_policy(env, ppo, checkpoint, *, actor_true_forward_speed=False, actor_history_steps=1):
    cfg = runner_config(
        ppo,
        actor_true_forward_speed=actor_true_forward_speed,
        actor_history_steps=actor_history_steps,
    )
    runner = OnPolicyRunner(env, cfg, device=str(env.device))
    load_checkpoint(runner, checkpoint, load_optimizer=False)
    return runner.get_inference_policy(device=env.device)
