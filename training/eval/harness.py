"""Shared multi-checkpoint loading and policy-building helpers for acceptance suites."""

from pathlib import Path

from rl.checkpoint import CheckpointBatch, build_inference_policy, load_checkpoint_snapshots


def load_eval_batch(checkpoints, *, require_same_model=False) -> CheckpointBatch:
    """Load and validate run_config snapshots for one or more evaluation checkpoints."""
    return load_checkpoint_snapshots(checkpoints, require_same_model=require_same_model)


def configure_eval_locomotion(locomotion: dict, batch: CheckpointBatch) -> None:
    """Align shared locomotion observation settings across a batch of checkpoints."""
    locomotion["actor_true_forward_speed"] = batch.any_speed_input
    locomotion["actor_history_steps"] = batch.max_history_steps


def load_eval_policies(env, batch: CheckpointBatch, *, ppo_override: dict | None = None):
    """Build deterministic inference policies for every checkpoint in `batch`."""
    policies = []
    for path, (cfg, _), use_speed, steps in zip(
        batch.paths, batch.snapshots, batch.speed_inputs, batch.history_steps, strict=True
    ):
        ppo = cfg["ppo"] if ppo_override is None else ppo_override
        policies.append(
            build_inference_policy(
                env,
                ppo,
                path,
                actor_true_forward_speed=use_speed,
                actor_history_steps=steps,
            )
        )
    return policies


def load_single_policy(env, ppo: dict, checkpoint: Path, *, actor_true_forward_speed=False, actor_history_steps=1):
    """Build a single deterministic inference policy for an evaluation environment."""
    return build_inference_policy(
        env,
        ppo,
        checkpoint,
        actor_true_forward_speed=actor_true_forward_speed,
        actor_history_steps=actor_history_steps,
    )
