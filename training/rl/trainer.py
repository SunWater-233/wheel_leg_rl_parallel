"""PPO training loop orchestration."""

import torch

from configs.schema import runner_config, save_run_config
from rl.checkpoint import load_checkpoint
from rl.runner import RobotRunner
from tasks.locomotion_env import WheelLegLocomotionEnv


def train(env, options, output, settings, live_view=None):
    if isinstance(env, WheelLegLocomotionEnv):
        env.enable_observation_noise()
    actor_speed = getattr(env, "actor_true_forward_speed", False)
    history_steps = getattr(env, "actor_history_steps", 1)
    runner = RobotRunner(
        env,
        runner_config(settings["ppo"], actor_true_forward_speed=actor_speed, actor_history_steps=history_steps),
        log_dir=str(output),
        device=str(env.device),
    )
    runner.live_view = live_view
    if options.resume is not None:
        load_checkpoint(runner, options.resume, load_optimizer=True)
        # RSL-RL restores the optimizer's rate, but PPO also tracks it for the adaptive schedule.
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
        runner.current_learning_iteration += 1
        print(f"Resuming at iteration {runner.current_learning_iteration} from {options.resume}", flush=True)
    elif options.init_checkpoint is not None:
        load_checkpoint(
            runner,
            options.init_checkpoint,
            load_optimizer=False,
            initialize_actor_speed=actor_speed,
            initialize_actor_history=history_steps > 1,
        )
        runner.current_learning_iteration = 0
        min_std = settings["ppo"].get("init_checkpoint_min_std", 0.0)
        if min_std > 0.0 and isinstance(getattr(runner.alg.policy, "std", None), torch.nn.Parameter):
            with torch.no_grad():
                runner.alg.policy.std.clamp_min_(min_std)
            print(
                f"Raised policy std floor to {min_std:g} on --init-checkpoint "
                f"(range {runner.alg.policy.std.min().item():.3f}-{runner.alg.policy.std.max().item():.3f})",
                flush=True,
            )
        print(f"Initializing from {options.init_checkpoint} with a fresh optimizer", flush=True)
    runner.limit_exploration(settings["ppo"]["max_action_std"])
    if isinstance(env, WheelLegLocomotionEnv):
        env.start_height_curriculum(runner.current_learning_iteration, runner.num_steps_per_env)
    save_run_config(output, settings, options, type(env), env.num_envs)
    try:
        runner.learn(num_learning_iterations=options.iterations, init_at_random_ep_len=True)
    finally:
        if runner.writer is not None:
            runner.writer.close()
    model = output / f"model_{runner.current_learning_iteration}.pt"
    print(f"PPO completed {options.iterations} iterations. Checkpoint: {model.resolve()}")
