"""Compare policies on held-out terrain transitions and stops at nine approach speeds."""

import argparse
import csv
from pathlib import Path

import genesis as gs
import torch

from configs.schema import TRAINING_DIR, load_config
from eval.harness import configure_eval_locomotion, load_eval_batch, load_eval_policies
from tasks.locomotion_env import WheelLegLocomotionEnv
from tasks.terrains import TransitionLanes


APPROACH_TIMEOUT_S = 27.0
STOP_TIMEOUT_S = 6.0
STOP_HOLD_S = 0.24
LEG_HEIGHT_M = 0.20


def prepare(env, cases, policy_count):
    env.auto_commands = False
    env.demo_initial_height = LEG_HEIGHT_M
    env.demo_surface = env.lanes.surfaces.index('transitions')
    count = len(cases)
    signed_speed = env.commands.new_tensor([-speed if reverse else speed for _, reverse, speed in cases])
    signed_speed = signed_speed.repeat(policy_count)
    for variant, reverse in dict.fromkeys((variant, reverse) for variant, reverse, _ in cases):
        env.demo_variant = variant
        env.demo_reverse = reverse
        ids = env.surface.new_tensor([j * count + i for j in range(policy_count)
                                      for i, case in enumerate(cases) if case[:2] == (variant, reverse)])
        env._reset_envs(ids)
    # Match the training run-up at high speeds. This part of each lane is level.
    high = signed_speed.abs() >= 1.6
    qpos = env.robot.get_qpos().clone()
    spawn_x = torch.where(high, -9.0 * signed_speed.sign(), env.spawn_x)
    qpos[:, env.base_xy[0]] = spawn_x
    env.robot.set_qpos(qpos, zero_velocity=True)
    return signed_speed


@torch.inference_mode()
def drive(env, policies, cases):
    speed = prepare(env, cases, len(policies))
    direction = speed.sign()
    first_join = abs(env.lanes.seams[0])
    crossed = torch.zeros_like(speed, dtype=torch.bool)
    stopped = torch.zeros_like(crossed)
    fallen = torch.zeros_like(crossed)
    left_lane = torch.zeros_like(crossed)
    active = ~crossed
    entry_speed = torch.zeros_like(speed)
    cross_time = torch.zeros_like(speed)
    stop_time = torch.zeros_like(speed)
    peak_vx = torch.zeros_like(speed)
    peak_pitch = torch.zeros_like(speed)
    peak_roll = torch.zeros_like(speed)
    still_steps = torch.zeros_like(speed, dtype=torch.long)
    count = len(cases)
    dt = env.control_dt
    for step in range(1, round((APPROACH_TIMEOUT_S + STOP_TIMEOUT_S) / dt) + 1):
        seconds = step * dt
        env.goals[:, 0] = speed.masked_fill(crossed | ~active, 0.0)
        vx, ang, _, _ = env._base_state()
        env._advance_commands(vx, ang)
        target = env.commands[:, 0].clone()
        obs = env.get_observations()
        actions = torch.cat([policy(obs[j * count:(j + 1) * count]) for j, policy in enumerate(policies)])
        _, _, done, extras = env.step(actions)
        measured, _, _, _ = env._base_state()
        pitch = extras['pitch_deg'].abs()
        roll = extras['posture'][:, 0]
        valid = active & ~done
        along = direction * env.robot.get_pos()[:, 0]
        arrived = valid & ~crossed & (along >= -first_join + 0.45)
        entry_speed = torch.where(arrived, direction * measured[:, 0], entry_speed)
        cross_time.masked_fill_(arrived, seconds)
        crossed |= arrived
        stable = (valid & crossed & (target.abs() <= 0.025) & (measured[:, :2].norm(dim=1) <= 0.08)
                  & (pitch <= 8.0) & (roll <= 8.0) & extras['wheel_contacts'].all(dim=1))
        still_steps = torch.where(stable, still_steps + 1, 0)
        settled = still_steps >= round(STOP_HOLD_S / dt)
        stop_time = torch.where(settled, seconds - cross_time, stop_time)
        stopped |= settled
        fallen |= active & extras['fallen']
        left_lane |= active & done & ~extras['fallen']
        peak_vx = torch.maximum(peak_vx, measured[:, 0].abs().masked_fill(~valid, 0.0))
        peak_pitch = torch.maximum(peak_pitch, pitch.masked_fill(~active, 0.0))
        peak_roll = torch.maximum(peak_roll, roll.masked_fill(~active, 0.0))
        expired = torch.where(crossed, seconds - cross_time >= STOP_TIMEOUT_S, seconds >= APPROACH_TIMEOUT_S)
        active &= ~(done | settled | expired)
        if not active.any().item():
            break
    return torch.stack((crossed, stopped, fallen, left_lane, cross_time, stop_time,
                        entry_speed, peak_vx, peak_pitch, peak_roll), dim=1).tolist()


def compare(args):
    config = load_config(args.config)
    task = config['task']
    settings = task['locomotion']
    if settings['terrain'] != 'transitions':
        raise ValueError('A transitions configuration is required')
    settings['terrain_seed'] = args.terrain_seed
    settings['low_recovery']['flat_episode_fraction'] = 0.0
    lanes = TransitionLanes(settings)
    cases = [(i, reverse, speed) for i in range(lanes.variants) for reverse in (False, True)
             if not (reverse and lanes.forward_only[lanes.variants + i])
             for speed in settings['command_speed_levels']]
    batch = load_eval_batch(args.checkpoints, require_same_model=True)
    configure_eval_locomotion(settings, batch)
    run = batch.first_run
    env = WheelLegLocomotionEnv(TRAINING_DIR / run['model'], len(cases) * len(batch.paths),
                                episode_seconds=APPROACH_TIMEOUT_S + STOP_TIMEOUT_S + 1, task_config=task)
    policies = load_eval_policies(env, batch)
    outcomes = drive(env, policies, cases)
    rows = []
    for j, checkpoint in enumerate(args.checkpoints):
        group = outcomes[j * len(cases):(j + 1) * len(cases)]
        for (variant, reverse, speed), result in zip(cases, group, strict=True):
            crossed, stopped, fallen, left, crossing, stop, entry, peak, pitch, roll = result
            source, dest = lanes.segment_families[variant][::-1 if reverse else 1][:2]
            step = -lanes.seam_steps[variant][-1] if reverse else lanes.seam_steps[variant][0]
            rows.append(dict(checkpoint=str(checkpoint), seed=args.terrain_seed, variant=variant + 1,
                             reverse=reverse, speed_m_s=speed, from_terrain=source, to_terrain=dest,
                             join_step_cm=100 * step, crossed=bool(crossed), stopped=bool(stopped), fallen=bool(fallen),
                             left_lane=bool(left), crossing_s=crossing if crossed else None,
                             stop_s=stop if stopped else None, speed_at_crossing_m_s=entry if crossed else None,
                             peak_abs_speed_m_s=peak, peak_abs_pitch_deg=pitch, peak_abs_roll_deg=roll))
        print(f'{checkpoint}: {sum(row[0] for row in group):g}/{len(group)} crossed; '
              f'{sum(row[1] for row in group):g}/{len(group)} stopped', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path, nargs='+')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--terrain-seed', type=int, default=619)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    gs.init(backend=gs.cuda, seed=123, logging_level='warning')
    try:
        compare(args)
    finally:
        gs.destroy()


if __name__ == '__main__':
    main()
