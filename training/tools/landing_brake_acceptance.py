"""Fixed slope-to-rough transitions; score braking and optionally trace every control step."""

import argparse
import csv
from itertools import product
from math import ceil
from pathlib import Path

import genesis as gs
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from configs.experiment import TRAINING_DIR, load_config, load_snapshot, runner_config
from envs.terrain import TransitionLanes
from main import load_checkpoint
from robots.wheel_leg_locomotion import WheelLegLocomotionEnv


DROPS = (0.0, 0.060, 0.110, 0.185)
SKEWS = (0.0, 0.32)
HEIGHTS = (0.20, 0.24)
APPROACH_S = 20.0
STOP_S = 4.5
HOLD_S = 0.24
EXTRA_STOP_M = 0.30


class LandingLanes(TransitionLanes):
    variations = tuple(product(DROPS, SKEWS))

    def __init__(self, settings):
        super().__init__(settings)
        self.field.fill(0)
        self.spawn_z.fill(0)
        self.forward_only[self.variants:] = True
        rng = np.random.default_rng(settings['terrain_seed'])
        nx = self.field.shape[0]
        ny = round(self.lane_width / self.spacing)
        x = (np.arange(nx) * self.spacing - self.half_length)[:, None]
        y = (np.arange(ny) * self.spacing - self.lane_width / 2)[None, :]
        for lane, (drop, skew) in enumerate(self.variations, self.variants):
            along = x - skew * y
            rise = 0.10 * np.clip(along + 2.4, 0.0, 2.4)
            rough = rng.uniform(-0.008, 0.008, ((nx + 3) // 4, (ny + 3) // 4))
            rough = rough.repeat(4, axis=0).repeat(4, axis=1)[:nx, :ny]
            lower = 0.24 - drop + rough * np.clip(along / 0.30, 0.0, 1.0)
            tile = np.where(along < -1e-8, rise, lower)
            self.field[:, lane * ny:(lane + 1) * ny] = np.rint(tile / self.vertical_scale)
            self.reverse_spawn_z[lane] = self.field[-1, lane * ny + ny // 2] * self.vertical_scale
        self.field[:, -1] = self.field[:, -2]


class LandingEnv(WheelLegLocomotionEnv):
    def _make_lanes(self, settings):
        return LandingLanes(settings)

    def _termination(self, height, gravity, contact):
        fallen, outside = super()._termination(height, gravity, contact)
        self.transition_contacts_z = contact[:, self.wheel_links, 2]
        self.transition_outside = outside
        return fallen, outside

    def _after_step(self, extras, done, lin_vel, ang_vel):
        super()._after_step(extras, done, lin_vel, ang_vel)
        extras['landing_pos'] = self.robot.get_pos().clone()
        extras['landing_vel'] = lin_vel.clone()
        extras['landing_wheel_load'] = self.transition_contacts_z.clone()
        extras['landing_edge'] = torch.stack([wheel.get_anchor_pos()[:, :2] for wheel in self.wheels], dim=1)
        extras['landing_outside'] = self.transition_outside.clone()
        extras['landing_airborne'] = self.scut_reward.air_time > 0
        extras['landing_brake_active'] = self.scut_reward.braking_cost.direction != 0
        if self.collect_trace:
            extras['landing_wheel_torque'] = self.robot.get_dofs_control_force(self.wheel_dofs) * self.wheel_direction
            extras['landing_wheel_speed'] = self.robot.get_dofs_velocity(self.wheel_dofs) * self.wheel_direction
            extras['landing_pitch_rate'] = ang_vel[:, 1].clone()


def prepare(env, cases, model_count):
    env.auto_commands = False
    env.demo_surface = env.lanes.surfaces.index('transitions')
    env.demo_reverse = False
    n = len(cases)
    for variant, height in dict.fromkeys((v, h) for v, _, h in cases):
        env.demo_initial_height = height
        env.demo_variant = variant
        ids = env.surface.new_tensor([j * n + i for j in range(model_count)
                                       for i, (v, _, h) in enumerate(cases) if (v, h) == (variant, height)])
        env._reset_envs(ids)


@torch.inference_mode()
def drive(env, policies, cases, trace):
    prepare(env, cases, len(policies))
    env.collect_trace = trace is not None
    speed = env.commands.new_tensor([s for _, s, _ in cases]).repeat(len(policies))
    skew = env.commands.new_tensor([LandingLanes.variations[v][1] for v, _, _ in cases]).repeat(len(policies))
    max_stop_distance = speed.square() / (2 * env.manual_forward_acceleration) + EXTRA_STOP_M
    center_y = env.spawn_y.clone()
    active = torch.ones_like(speed, dtype=torch.bool)
    crossed = torch.zeros_like(active)
    stopped = torch.zeros_like(active)
    fallen = torch.zeros_like(active)
    outside = torch.zeros_like(active)
    airborne_during_stop = torch.zeros_like(active)
    brake_while_airborne = torch.zeros_like(active)
    was_airborne = torch.zeros_like(active)
    touched = torch.zeros_like(active)
    stop_requested = torch.zeros_like(active)
    request_t = torch.zeros_like(speed)
    settled_t = torch.zeros_like(speed)
    request_x = torch.zeros_like(speed)
    excess_distance = torch.zeros_like(speed)
    pitch_peak = torch.zeros_like(speed)
    roll_peak = torch.zeros_like(speed)
    post_contact_speed = torch.zeros_like(speed)
    still = torch.zeros_like(speed, dtype=torch.long)
    elapsed = torch.zeros_like(speed)
    n = len(cases)
    dt = env.control_dt
    for i in range(ceil((APPROACH_S + STOP_S) / dt)):
        t = (i + 1) * dt
        goal = speed.masked_fill(stop_requested | ~active, 0.0)
        env.goals[:, 0] = goal
        vx, ang, _, _ = env._base_state()
        env._advance_commands(vx, ang)
        target = env.commands[:, 0].clone()
        obs = env.get_observations()
        actions = torch.cat([policy(obs[j * n:(j + 1) * n]) for j, policy in enumerate(policies)])
        _, _, done, extra = env.step(actions)
        pos, vel, wheels = (extra[k] for k in ('landing_pos', 'landing_vel', 'landing_edge'))
        x, y, z = pos.unbind(1)
        wheel_x = wheels[..., 0] - skew[:, None] * (wheels[..., 1] - center_y[:, None])
        crossed |= active & ~done & (wheel_x >= 0.10).all(1)
        # Release the throttle as the leading wheel rolls past the edge, whether or not
        # the trailing wheel has landed. The scored request starts at the next control tick.
        release = active & ~done & ~stop_requested & (wheel_x.amax(1) >= 0.05)
        request_t.masked_fill_(release, t)
        request_x = torch.where(release, x, request_x)
        braking = active & stop_requested
        loaded = extra['landing_wheel_load'] > 5.0
        pitch, roll = extra['pitch_deg'], extra['posture'][:, 0]
        airborne = extra['landing_airborne']
        airborne_during_stop |= braking & airborne
        brake_while_airborne |= braking & airborne & extra['landing_brake_active']
        was_airborne |= active & ~loaded.any(1)
        touched |= active & was_airborne & loaded.any(1)
        post_contact_speed = torch.maximum(post_contact_speed, vel[:, 0].abs().masked_fill(~(braking & touched), 0))
        excess_distance = torch.maximum(excess_distance, (x - request_x).masked_fill(~braking, 0))
        pitch_peak = torch.maximum(pitch_peak, pitch.abs().masked_fill(~active, 0))
        roll_peak = torch.maximum(roll_peak, roll.masked_fill(~active, 0))
        stable = (braking & crossed & ~done & (target.abs() <= 0.025) & (vel[:, :2].norm(dim=1) <= 0.08)
                  & (pitch.abs() <= 8.0) & (roll <= 8.0) & loaded.all(1))
        still = torch.where(stable, still + 1, 0)
        settled = still >= round(HOLD_S / dt)
        settled_t = torch.where(settled, t - request_t, settled_t)
        stopped |= settled
        fallen |= active & extra['fallen']
        outside |= active & extra['landing_outside']
        stop_requested |= release
        expired = torch.where(stop_requested, t - request_t >= STOP_S, t >= APPROACH_S)
        elapsed.masked_fill_(active, t)
        if trace is not None:
            record = active & ((wheel_x.amax(1) >= -0.55) | done)
            values = torch.column_stack((goal, target, pos, vel, pitch, roll, extra['landing_pitch_rate'],
                                          actions, extra['landing_wheel_speed'], extra['landing_wheel_torque'],
                                          extra['landing_wheel_load'], airborne, extra['landing_brake_active'],
                                          extra['fallen'], extra['landing_outside'], done))
            for j, values_row in zip(record.nonzero().flatten().tolist(), values[record].tolist(), strict=True):
                trace.writerow((j // n, j % n, round(t, 3), *(round(v, 6) for v in values_row)))
        active &= ~(done | expired | settled)
        if not active.any().item():
            break
    passed = stopped & (excess_distance <= max_stop_distance)
    return torch.column_stack((crossed, stopped, passed, fallen, outside, airborne_during_stop, brake_while_airborne,
                               elapsed, settled_t, excess_distance, max_stop_distance, post_contact_speed,
                               pitch_peak, roll_peak)).tolist()


def compare(args):
    config = load_config(args.config)
    task = config['task']
    settings = task['locomotion']
    if settings['terrain'] != 'transitions' or not task['scut_rough_reward']['braking']['enabled']:
        raise ValueError('The fixed landing test requires transitions and enabled braking')
    settings['terrain_seed'] = args.seed
    settings['low_recovery']['flat_episode_fraction'] = 0.0
    cases = list(product(range(len(LandingLanes.variations)), settings['transition_speed_levels'], HEIGHTS))
    if not cases:
        raise ValueError('Set transition_speed_levels in the test configuration')
    snapshots = [load_snapshot(p.with_name('run_config.json')) for p in args.checkpoints]
    speed_inputs = [cfg['task']['locomotion']['actor_true_forward_speed'] for cfg, _ in snapshots]
    settings['actor_true_forward_speed'] = any(speed_inputs)
    run = snapshots[0][1]
    if any(meta['model_sha256'] != run['model_sha256'] or meta['task'] != 'locomotion' for _, meta in snapshots):
        raise ValueError('Checkpoint models must match')
    masked_count = sum(speed_inputs) if args.mask_actor_speed else 0
    env = LandingEnv(TRAINING_DIR / run['model'], len(cases) * (len(snapshots) + masked_count),
                     episode_seconds=APPROACH_S + STOP_S + 1, task_config=task)
    policies = []
    for path, (settings, _), use_speed in zip(args.checkpoints, snapshots, speed_inputs, strict=True):
        runner = OnPolicyRunner(env, runner_config(settings['ppo'], actor_true_forward_speed=use_speed),
                                device=str(env.device))
        load_checkpoint(runner, path, load_optimizer=False)
        policies.append(runner.get_inference_policy(device=env.device))
    labels = ['as_trained'] * len(policies)
    checkpoints = list(args.checkpoints)
    if args.mask_actor_speed:
        for path, policy, use_speed in zip(args.checkpoints, policies[:], speed_inputs, strict=True):
            if use_speed:
                def masked(obs, policy=policy):
                    obs = obs.copy()
                    obs['body_forward_speed'] = torch.zeros_like(obs['body_forward_speed'])
                    return policy(obs)
                policies.append(masked)
                checkpoints.append(path)
                labels.append('zero_speed_input')
    if args.trace:
        with args.trace.open('x', encoding='utf-8', newline='') as stream:
            trace = csv.writer(stream)
            trace.writerow('model_index case_index t_s goal_m_s target_m_s x_m y_m z_m vx_m_s vy_m_s vz_m_s '
                           'pitch_deg roll_deg pitch_rate_rad_s hip_l hip_r knee_l knee_r wheel_l wheel_r '
                           'wheel_l_rad_s wheel_r_rad_s wheel_l_nm wheel_r_nm wheel_l_load_n wheel_r_load_n '
                           'airborne brake_active fall outside reset'.split())
            raw = drive(env, policies, cases, trace)
    else:
        raw = drive(env, policies, cases, None)
    rows = []
    for j, (checkpoint, label) in enumerate(zip(checkpoints, labels, strict=True)):
        group = raw[j * len(cases):(j + 1) * len(cases)]
        for i, ((variant, speed, height), values) in enumerate(zip(cases, group, strict=True)):
            drop, skew = LandingLanes.variations[variant]
            crossed, stopped, passed, fallen, outside, airborne, brake_air, elapsed, stop, distance, limit, peak, pitch, roll = values
            rows.append(dict(checkpoint=str(checkpoint), input_mode=label, model_index=j, case_index=i, drop_cm=100 * drop,
                             skew=skew, speed_m_s=speed, height_m=height, crossed=bool(crossed), stopped=bool(stopped),
                             passed=bool(passed), fallen=bool(fallen), outside=bool(outside),
                             airborne_during_stop=bool(airborne), brake_while_airborne=bool(brake_air), elapsed_s=elapsed,
                             stop_s=stop if stopped else None, distance_m=distance, distance_limit_m=limit,
                             peak_speed_after_touchdown_m_s=peak, peak_abs_pitch_deg=pitch, peak_abs_roll_deg=roll))
        print(f'{checkpoint} [{label}]: {sum(row[0] for row in group):g} crossed, {sum(row[1] for row in group):g} stopped, '
              f'{sum(row[2] for row in group):g}/{len(group)} passed the distance gate', flush=True)
    with args.output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path, nargs='+')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--mask-actor-speed', action='store_true',
                        help='Also score each speed-input policy with that input held at zero')
    parser.add_argument('--seed', type=int, default=541)
    args = parser.parse_args()
    gs.init(backend=gs.cuda, seed=123, logging_level='warning')
    try:
        compare(args)
    finally:
        gs.destroy()


if __name__ == '__main__':
    main()
