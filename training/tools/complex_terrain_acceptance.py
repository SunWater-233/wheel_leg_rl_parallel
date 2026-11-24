"""Compare V0.2 policies on the same held-out, full-length SCUT lanes."""

import argparse
import csv
from itertools import product
from pathlib import Path

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from configs.experiment import TRAINING_DIR, load_snapshot, runner_config
from main import load_checkpoint
from robots.wheel_leg_locomotion import WheelLegLocomotionEnv


SURFACES = ('rough_slope', 'staggered_bars', 'stairs', 'inv_stairs_high',
            'rough', 'bars', 'single_side_bars_left', 'single_side_bars_right')
SPEEDS = (0.65, 0.80)
HEIGHTS = (0.18, 0.24)
COURSE_END_X = 3.7
TIMEOUT_S = 16.0


class AcceptanceEnv(WheelLegLocomotionEnv):
    def begin_courses(self, cases):
        self.auto_commands = False
        for case in dict.fromkeys(cases):
            surface, reverse, speed, height = case
            self.demo_initial_height = height
            self.demo_surface = self.lanes.surfaces.index(surface)
            self.demo_reverse = reverse
            self.demo_variant = self.lanes.variants - 1
            ids = self.surface.new_tensor([i for i, other in enumerate(cases) if other == case])
            self._reset_envs(ids)
            self.goals[ids, 0] = -speed if reverse else speed


@torch.inference_mode()
def drive(env, policies, cases):
    """Measure only the first attempt, stopping metrics at the far edge or first reset."""
    direction = env.commands.new_tensor([-1.0 if reverse else 1.0 for _, reverse, _, _ in cases])
    active = torch.ones_like(direction, dtype=torch.bool)
    passed = torch.zeros_like(active)
    fallen = torch.zeros_like(active)
    ticks = torch.zeros_like(direction)
    peak_pitch = torch.zeros_like(direction)
    peak_roll = torch.zeros_like(direction)
    peak_drift = torch.zeros_like(direction)
    progress = torch.zeros_like(direction)
    speed_error = torch.zeros_like(direction)
    speed_samples = torch.zeros_like(direction)
    spawn_x, spawn_y = env.spawn_x.clone(), env.spawn_y.clone()
    count = len(cases) // len(policies)
    for tick in range(1, round(TIMEOUT_S / env.control_dt) + 1):
        lin_vel, ang_vel, _, _ = env._base_state()
        env._advance_commands(lin_vel, ang_vel)
        obs = env.get_observations()
        actions = torch.cat([policy(obs[i * count:(i + 1) * count]) for i, policy in enumerate(policies)])
        target = env.commands[:, 0].clone()
        _, _, done, extras = env.step(actions)
        x, y, _ = env.robot.get_pos().unbind(1)
        valid = active & ~done
        reached = valid & (direction * x >= COURSE_END_X)
        peak_pitch = torch.maximum(peak_pitch, extras['pitch_deg'].abs().masked_fill(~active, 0))
        peak_roll = torch.maximum(peak_roll, extras['posture'][:, 0].masked_fill(~active, 0))
        peak_drift = torch.maximum(peak_drift, (y - spawn_y).abs().masked_fill(~valid, 0))
        progress = torch.maximum(progress, (direction * (x - spawn_x)).masked_fill(~valid, 0))
        if tick * env.control_dt >= 1.5:
            vx = env._base_state()[0][:, 0]
            speed_error += (vx - target).abs().masked_fill(~valid, 0)
            speed_samples += valid
        ticks += active
        passed |= reached
        fallen |= active & extras['fallen']
        active &= ~(reached | done)
        if not active.any().item():
            break
    return torch.stack((passed, fallen, active, ticks * env.control_dt, progress,
                        speed_error / speed_samples.clamp_min(1), speed_samples,
                        peak_pitch, peak_roll, peak_drift), dim=1).tolist()


def compare(args):
    snapshots = [load_snapshot(path.with_name('run_config.json')) for path in args.checkpoints]
    settings, run = snapshots[0]
    task = settings['task']
    locomotion = task['locomotion']
    if any(meta.get('model_sha256') != run['model_sha256'] or meta.get('task') != 'locomotion'
           for _, meta in snapshots):
        raise ValueError('Checkpoints must use the same locomotion robot model')
    if locomotion['terrain'] != 'scut_ground' or not locomotion['terrain_traversal']:
        raise ValueError('Use a SCUT ground checkpoint as the first checkpoint')
    locomotion.update(terrain_seed=args.terrain_seed, scut_ground_scale=1.0,
                      scut_ground_compound_fraction=0.10, scut_ground_single_side_fraction=0.10)
    locomotion['low_recovery']['flat_episode_fraction'] = 0.0
    cases = list(product(SURFACES, (False, True), SPEEDS, HEIGHTS))
    trials = list(product(args.checkpoints, cases))
    courses = cases * len(args.checkpoints)
    env = AcceptanceEnv(TRAINING_DIR / run['model'], len(trials), episode_seconds=TIMEOUT_S + 1, task_config=task)
    policies = []
    for checkpoint, (config, _) in zip(args.checkpoints, snapshots, strict=True):
        runner = OnPolicyRunner(env, runner_config(config['ppo']), device=str(env.device))
        load_checkpoint(runner, checkpoint, load_optimizer=False)
        policies.append(runner.get_inference_policy(device=env.device))
    with torch.inference_mode():
        env.begin_courses(courses)
        outcomes = drive(env, policies, courses)
    rows = []
    for (checkpoint, (surface, reverse, speed, height)), values in zip(trials, outcomes, strict=True):
        passed, fallen, timed_out, elapsed, progress, speed_mae, samples, pitch, roll, drift = values
        status = 'crossed' if passed else 'fall' if fallen else 'timeout' if timed_out else 'left_lane'
        rows.append(dict(checkpoint=str(checkpoint), surface=surface, reverse=reverse, speed_m_s=speed, height_m=height,
                         terrain_seed=args.terrain_seed, passed=bool(passed), status=status, elapsed_s=elapsed,
                         progress_m=progress, speed_mae_m_s=speed_mae if samples else None,
                         peak_abs_pitch_deg=pitch, peak_abs_roll_deg=roll, peak_drift_m=drift))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for checkpoint in args.checkpoints:
        group = [row for row in rows if row['checkpoint'] == str(checkpoint)]
        print(f'{checkpoint}: {sum(row["passed"] for row in group)}/{len(group)} crossings', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path, nargs='+')
    parser.add_argument('--terrain-seed', type=int, default=271)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    gs.init(backend=gs.cuda, seed=123, logging_level='warning')
    try:
        compare(args)
    finally:
        gs.destroy()


if __name__ == '__main__':
    main()
