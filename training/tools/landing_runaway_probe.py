"""Isolate wheel drive from landing dynamics without changing a trained policy."""

import argparse
import csv
from itertools import product
from pathlib import Path

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from configs.experiment import TRAINING_DIR, load_config, load_snapshot, runner_config
from main import load_checkpoint
from tools.landing_brake_acceptance import APPROACH_S, HEIGHTS, STOP_S, LandingEnv, LandingLanes, drive


MODES = ('policy', 'coast_on_request', 'coast_at_zero', 'brake_only_at_zero',
         'fast_stop_reference', 'unlimited_stop_reference')


class ProbeEnv(LandingEnv):
    def _advance_commands(self, lin_vel, ang_vel):
        # Change only the stop ramp; approach commands and the wheel servo remain unchanged.
        nominal = self.motion_step[:, 0].clone()
        stopping = self.goals[:, 0].abs() <= 1e-6
        self.motion_step[:, 0].masked_fill_(stopping & (self.probe_mode == 4), 2.0 * self.control_dt)
        self.motion_step[:, 0].masked_fill_(stopping & (self.probe_mode == 5), 2.5)
        try:
            super()._advance_commands(lin_vel, ang_vel)
        finally:
            self.motion_step[:, 0].copy_(nominal)

    def _apply_controls(self, actions):
        actions = super()._apply_controls(actions)
        self.policy_wheel_target = (actions[:, 4:] * 10).clamp(-self.wheel_target_limit, self.wheel_target_limit)
        return actions

    def _apply_springs(self):
        super()._apply_springs()
        stop_requested = self.goals[:, 0].abs() <= 1e-6
        zero_reference = stop_requested & (self.commands[:, 0].abs() <= 1e-6)
        coast = (stop_requested & (self.probe_mode == 1)) | (zero_reference & (self.probe_mode == 2))
        coast_ids = self.all_envs[coast]
        if coast_ids.numel():
            self.robot.control_dofs_force(self.commands.new_zeros((coast_ids.numel(), 2)), self.wheel_dofs,
                                          envs_idx=coast_ids)
        brake_ids = self.all_envs[zero_reference & (self.probe_mode == 3)]
        if brake_ids.numel():
            # Keep the velocity servo's opposing torque while removing targets that
            # would accelerate either wheel in the forward direction.
            current = self.robot.get_dofs_velocity(self.wheel_dofs, envs_idx=brake_ids) * self.wheel_direction
            target = torch.minimum(self.policy_wheel_target[brake_ids], current)
            self.robot.control_dofs_velocity(target * self.wheel_direction, self.wheel_dofs, envs_idx=brake_ids)


def run(args):
    config = load_config(args.config)
    task = config['task']
    settings = task['locomotion']
    settings['terrain_seed'] = args.seed
    settings['low_recovery']['flat_episode_fraction'] = 0.0
    snapshots, run_meta = load_snapshot(args.checkpoint.with_name('run_config.json'))
    cases = list(product((v for v, (drop, _) in enumerate(LandingLanes.variations) if drop == 0.185),
                         settings['transition_speed_levels'], HEIGHTS))
    if not cases or settings['terrain'] != 'transitions' or not task['scut_rough_reward']['braking']['enabled']:
        raise ValueError('Use a landing-brake configuration with approach speeds')
    env = ProbeEnv(TRAINING_DIR / run_meta['model'], len(cases) * len(MODES),
                   episode_seconds=APPROACH_S + STOP_S + 1, task_config=task)
    env.probe_mode = torch.arange(len(MODES), device=env.device).repeat_interleave(len(cases))
    runner = OnPolicyRunner(env, runner_config(snapshots['ppo']), device=str(env.device))
    load_checkpoint(runner, args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    with args.trace.open('x', encoding='utf-8', newline='') as stream:
        trace = csv.writer(stream)
        trace.writerow('mode_index case_index t_s goal_m_s target_m_s x_m y_m z_m vx_m_s vy_m_s vz_m_s '
                       'pitch_deg roll_deg pitch_rate_rad_s hip_l hip_r knee_l knee_r wheel_l wheel_r '
                       'wheel_l_rad_s wheel_r_rad_s wheel_l_nm wheel_r_nm wheel_l_load_n wheel_r_load_n '
                       'airborne brake_active fall outside reset'.split())
        values = drive(env, [policy] * len(MODES), cases, trace)
    with args.output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('mode', 'case_index', 'skew', 'speed_m_s', 'height_m', 'crossed', 'stopped', 'passed',
                         'fallen', 'outside', 'airborne_during_stop', 'brake_while_airborne', 'elapsed_s', 'stop_s',
                         'distance_m', 'distance_limit_m', 'peak_speed_after_touchdown_m_s', 'peak_pitch_deg', 'peak_roll_deg'))
        n = len(cases)
        for i, mode in enumerate(MODES):
            group = values[i * n:(i + 1) * n]
            print(f'{mode}: {sum(row[1] for row in group):g}/{n} stopped; '
                  f'{sum(row[2] for row in group):g}/{n} passed; {sum(row[3] for row in group):g}/{n} fell', flush=True)
            for j, ((variant, speed, height), row) in enumerate(zip(cases, group, strict=True)):
                writer.writerow((mode, j, LandingLanes.variations[variant][1], speed, height, *row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=541)
    args = parser.parse_args()
    if args.output.exists() or args.trace.exists():
        parser.error('Use new output and trace paths')
    gs.init(backend=gs.cuda, seed=123, logging_level='warning')
    try:
        run(args)
    finally:
        gs.destroy()


if __name__ == '__main__':
    main()
