import unittest

import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from configs.experiment import TRAINING_DIR, load_config, runner_config
from main import add_actor_speed_input


class ActorSpeedTest(unittest.TestCase):
    def test_matched_configs_only_change_actor_observation(self):
        control = load_config(TRAINING_DIR / 'configs/archive/v02_landing_brake_control.toml')
        treatment = load_config(TRAINING_DIR / 'configs/archive/v02_landing_brake_true_speed.toml')
        self.assertFalse(control['task']['locomotion'].pop('actor_true_forward_speed'))
        self.assertTrue(treatment['task']['locomotion'].pop('actor_true_forward_speed'))
        self.assertEqual(control, treatment)

    def test_speed_warm_start_preserves_actions_and_critic(self):
        obs = TensorDict(dict(policy=torch.randn(8, 28), critic=torch.randn(8, 47),
                              body_forward_speed=torch.randn(8, 1)), batch_size=[8])
        ppo = load_config()['ppo']
        control_cfg = runner_config(ppo)
        treatment_cfg = runner_config(ppo, actor_true_forward_speed=True)
        control = ActorCritic(obs, control_cfg['obs_groups'], 6)
        treatment = ActorCritic(obs, treatment_cfg['obs_groups'], 6)
        with treatment.register_load_state_dict_pre_hook(add_actor_speed_input):
            treatment.load_state_dict(control.state_dict())
        self.assertEqual(treatment.actor[0].weight[:, -1].count_nonzero().item(), 0)
        torch.testing.assert_close(treatment.act_inference(obs), control.act_inference(obs))
        torch.testing.assert_close(treatment.evaluate(obs), control.evaluate(obs))

        # A learned speed column responds to the ablation but leaves the input group intact.
        with torch.no_grad():
            treatment.actor[0].weight[:, -1].fill_(1.0)
        masked = obs.copy()
        masked['body_forward_speed'] = torch.zeros_like(obs['body_forward_speed'])
        torch.testing.assert_close(treatment.act_inference(masked), control.act_inference(obs))
        self.assertFalse(torch.allclose(treatment.act_inference(obs), treatment.act_inference(masked)))
