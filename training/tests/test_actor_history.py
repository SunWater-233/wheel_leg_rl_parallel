import unittest
from types import SimpleNamespace

import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from configs.experiment import TRAINING_DIR, load_config, runner_config
from main import add_actor_history_inputs
from robots.wheel_leg_locomotion import WheelLegLocomotionEnv


class ActorHistoryTest(unittest.TestCase):
    def test_history_warm_start_preserves_actions_and_critic(self):
        obs = TensorDict(
            dict(
                policy=torch.randn(8, 28),
                policy_history=torch.randn(8, 4 * 28),
                critic=torch.randn(8, 47),
            ),
            batch_size=[8],
        )
        ppo = load_config()["ppo"]
        base_cfg = runner_config(ppo, actor_history_steps=1)
        hist_cfg = runner_config(ppo, actor_history_steps=5)
        base = ActorCritic(obs, base_cfg["obs_groups"], 6)
        hist = ActorCritic(obs, hist_cfg["obs_groups"], 6)

        with hist.register_load_state_dict_pre_hook(add_actor_history_inputs):
            hist.load_state_dict(base.state_dict())

        self.assertEqual(hist.actor[0].weight[:, 28:].count_nonzero().item(), 0)
        torch.testing.assert_close(hist.act_inference(obs), base.act_inference(obs))
        torch.testing.assert_close(hist.evaluate(obs), base.evaluate(obs))

        with torch.no_grad():
            hist.actor[0].weight[:, 28:].fill_(0.5)
        self.assertFalse(torch.allclose(hist.act_inference(obs), base.act_inference(obs)))

    def test_history_buffer_fill_idempotence_shift_and_reset(self):
        num_envs, steps, dim = 2, 5, 28
        env = SimpleNamespace(
            num_envs=num_envs,
            actor_history_steps=steps,
            policy_obs_dim=dim,
            policy_history=torch.zeros((num_envs, steps - 1, dim)),
            last_policy_obs=torch.zeros((num_envs, dim)),
            history_needs_fill=torch.ones(num_envs, dtype=torch.bool),
        )

        def record_obs(policy):
            if env.history_needs_fill.any():
                fill = env.history_needs_fill
                env.policy_history[fill] = policy[fill, None, :]
                env.history_needs_fill[fill] = False
            env.last_policy_obs.copy_(policy)
            return env.policy_history.reshape(num_envs, -1).clone()

        def advance():
            env.policy_history[:, 1:] = env.policy_history[:, :-1].clone()
            env.policy_history[:, 0] = env.last_policy_obs

        # Initial fill repeats frame 0 across all 4 history slots and stays idempotent on repeat reads.
        p0 = torch.tensor([[1.0] * dim, [10.0] * dim])
        h0 = record_obs(p0)
        h0_repeat = record_obs(p0)
        torch.testing.assert_close(h0, h0_repeat)
        self.assertTrue((h0[0] == 1.0).all().item())
        self.assertTrue((h0[1] == 10.0).all().item())

        # Step 1 & 2 shift recent frames into slot 0 first: [t-1, t-2, t-3, t-4].
        advance()
        p1 = torch.tensor([[2.0] * dim, [20.0] * dim])
        h1 = record_obs(p1).view(num_envs, steps - 1, dim)
        self.assertEqual(h1[0, 0, 0].item(), 1.0)
        self.assertEqual(h1[0, 1, 0].item(), 1.0)

        advance()
        p2 = torch.tensor([[3.0] * dim, [30.0] * dim])
        h2 = record_obs(p2).view(num_envs, steps - 1, dim)
        self.assertEqual([h2[0, i, 0].item() for i in range(4)], [2.0, 1.0, 1.0, 1.0])
        self.assertEqual([h2[1, i, 0].item() for i in range(4)], [20.0, 10.0, 10.0, 10.0])

        # Resetting env 0 clears stale history and refills with the new episode's frame 0 only.
        advance()
        env.policy_history[0] = 0.0
        env.last_policy_obs[0] = 0.0
        env.history_needs_fill[0] = True
        p3 = torch.tensor([[99.0] * dim, [40.0] * dim])
        h3 = record_obs(p3).view(num_envs, steps - 1, dim)
        self.assertEqual([h3[0, i, 0].item() for i in range(4)], [99.0, 99.0, 99.0, 99.0])
        self.assertEqual([h3[1, i, 0].item() for i in range(4)], [30.0, 20.0, 10.0, 10.0])

    def test_history_config_validation_and_toml(self):
        cfg = load_config(TRAINING_DIR / "configs/v02_history5_complex_finetune.toml")
        self.assertEqual(cfg["task"]["locomotion"]["actor_history_steps"], 5)
        self.assertEqual(
            runner_config(cfg["ppo"], actor_history_steps=5)["obs_groups"]["policy"],
            ["policy", "policy_history"],
        )


if __name__ == "__main__":
    unittest.main()
