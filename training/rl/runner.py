"""RSL-RL OnPolicyRunner subclass with std capping and custom TensorBoard / live-view metrics."""

from statistics import mean

from rsl_rl.runners import OnPolicyRunner
import torch


class RobotRunner(OnPolicyRunner):
    live_view = None
    capped_action_std = None

    def save(self, path, infos=None):
        super().save(path, {**(infos or {}), "robot": self.env.label, "task": self.env.task_name})

    def limit_exploration(self, maximum):
        if not maximum:
            return
        std = getattr(self.alg.policy, "std", None)
        if not isinstance(std, torch.nn.Parameter):
            raise ValueError("ppo.max_action_std requires a per-action standard deviation parameter")

        @torch.no_grad()
        def clamp_std(*_):
            std.clamp_(min=1e-6, max=maximum)

        clamp_std()
        self.alg.optimizer.register_step_post_hook(clamp_std)
        self.capped_action_std = std
        print(f"Per-action exploration std capped at {maximum:g}", flush=True)

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        if getattr(self.env, "transition_course", False):
            peak = self.env.transition_peak_command.masked_fill(self.env.surface == 0, 0.0).max().item()
            self.writer.add_scalar("Transition/max_live_command_m_s", peak, locs["it"])
        if getattr(self.env, "recovery_fraction", 0.0):
            attempts, successes = self.env.recovery_counts.tolist()
            self.env.recovery_counts.zero_()
            if attempts:
                self.writer.add_scalar("Recovery/attempts", attempts, locs["it"])
                self.writer.add_scalar("Recovery/success_rate", successes / attempts, locs["it"])
                print(f"Low recovery stable: {successes}/{attempts} ({successes / attempts:.1%})", flush=True)
        if self.capped_action_std is not None:
            self.writer.add_scalar("Policy/max_noise_std", self.capped_action_std.detach().max().item(), locs["it"])
        elif hasattr(self.alg.policy, "std"):
            std_detached = self.alg.policy.std.detach()
            self.writer.add_scalar("Policy/max_noise_std", std_detached.max().item(), locs["it"])
            self.writer.add_scalar("Policy/mean_noise_std", std_detached.mean().item(), locs["it"])
        if getattr(self.env, "terrain_curriculum_enabled", False):
            mean_lvl = self.env.terrain_level.float().mean().item()
            maxed_frac = self.env.terrain_maxed.float().mean().item()
            self.writer.add_scalar("Curriculum/mean_terrain_level", mean_lvl, locs["it"])
            self.writer.add_scalar("Curriculum/maxed_fraction", maxed_frac, locs["it"])
            print(
                f"Terrain curriculum: mean level {mean_lvl:.2f}/{self.env.lanes.variants - 1}, "
                f"maxed {maxed_frac:.1%}",
                flush=True,
            )
        if self.live_view is None:
            return
        batch_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        seconds = locs["collection_time"] + locs["learn_time"]
        rewards, lengths = locs["rewbuffer"], locs["lenbuffer"]
        losses = locs["loss_dict"]
        self.live_view.add_metrics(
            iteration=locs["it"] - locs["start_iter"] + 1,
            steps=self.tot_timesteps,
            steps_per_second=int(batch_size / seconds),
            elapsed=self.tot_time,
            reward=mean(rewards) if rewards else None,
            episode_length=mean(lengths) if lengths else None,
            value_loss=losses["value_function"],
            policy_loss=losses["surrogate"],
        )
