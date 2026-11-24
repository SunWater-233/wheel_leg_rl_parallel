"""Compare locomotion checkpoints on the same flat-ground command sequences."""

import argparse
import csv
from itertools import product
from math import isfinite
from pathlib import Path

import genesis as gs
from rsl_rl.runners import OnPolicyRunner
import torch

from configs.experiment import TRAINING_DIR, load_config, load_snapshot, runner_config
from main import load_checkpoint
from robots.wheel_leg_locomotion import WheelLegLocomotionEnv

# Each segment is (seconds, forward speed, hip-to-wheel height).
HEIGHT_TARGETS = (0.20, 0.24, 0.20, 0.33, 0.20)
MOTIONS = {
    "stationary_height": tuple((2.0, 0.0, height) for height in HEIGHT_TARGETS),
    "forward_height": tuple((2.0, 0.50, height) for height in HEIGHT_TARGETS),
    "reverse_height": tuple((2.0, -0.50, height) for height in HEIGHT_TARGETS),
    "cruise": ((0.6, 0.0, 0.20), (4.0, 0.55, 0.20)),
    "stop_and_reverse": ((0.6, 0.0, 0.20), (2.8, 0.8, 0.20), (2.4, 0.0, 0.20),
                         (2.8, -0.5, 0.20), (2.4, 0.0, 0.20)),
}
RECOVERY_OFFSETS = tuple(product((-0.075, 0.075), repeat=2))
RECOVERY_TABLE_OFFSETS = tuple(product((-0.11, -0.075, 0.075, 0.11), repeat=2))


def command_trace(segments, dt, tensor):
    rows = [(speed, 0.0, height) for seconds, speed, height in segments
            for _ in range(round(seconds / dt))]
    return tensor.new_tensor(rows)


def score_motion(results, segments, dt, done):
    vx, pitch, roll, height, axle_x, loaded = results.T
    stop_peak, height_error, speed_error = 0.0, 0.0, 0.0
    end = 0
    for seconds, speed, target_height in segments:
        end += round(seconds / dt)
        tail = slice(end - round(0.40 / dt), end)
        height_error = max(height_error, max(abs(height[tail] - target_height)))
        speed_error = max(speed_error, max(abs(vx[tail] - speed)))
        if speed == 0.0:
            stop_peak = max(stop_peak, max(abs(vx[tail])))
    # The cruise metric uses the last two seconds of motion after the ramp.
    tail = slice(-round(2.0 / dt), None)
    axle_p2p = float(max(axle_x[tail]) - min(axle_x[tail]))
    return {
        "interrupted_s": done * dt if done else "",
        "peak_abs_pitch_deg": float(max(abs(pitch))),
        "peak_abs_roll_deg": float(max(abs(roll))),
        "height_tail_max_error_cm": 100 * height_error,
        "speed_tail_max_error_m_s": speed_error,
        "speed_zero_tail_max_m_s": stop_peak,
        "mean_axle_x_tail_p2p_cm": 100 * axle_p2p,
        "dual_loaded_fraction": float(loaded.mean()),
    }


@torch.inference_mode()
def compare(args):
    snapshots = [load_snapshot(path.with_name("run_config.json")) for path in args.checkpoints]
    source_run = snapshots[0][1]
    settings = load_config(args.config)
    task = settings["task"]
    locomotion = task["locomotion"]
    speed_inputs = [cfg["task"]["locomotion"]["actor_true_forward_speed"] for cfg, _ in snapshots]
    locomotion["actor_true_forward_speed"] = any(speed_inputs)
    if not locomotion["enabled"] or locomotion["terrain"] != "plane":
        raise ValueError("flat acceptance requires a plane locomotion config")
    if args.wheel_torque is not None:
        locomotion["wheel_torque_limit"] = args.wheel_torque
    if args.wheel_delay_steps is not None:
        locomotion["wheel_target_delay_steps"] = args.wheel_delay_steps
    use_recovery = locomotion["low_recovery"]["flat_episode_fraction"] > 0
    names = list(MOTIONS)
    if use_recovery:
        names += [f"recovery_{left:+.3f}_{right:+.3f}" for left, right in RECOVERY_OFFSETS]
    count = len(names)
    env = WheelLegLocomotionEnv(TRAINING_DIR / source_run["model"], len(args.checkpoints) * count,
                                episode_seconds=14.0, task_config=task)
    policies = []
    for checkpoint, use_speed in zip(args.checkpoints, speed_inputs, strict=True):
        runner = OnPolicyRunner(env, runner_config(settings["ppo"], actor_true_forward_speed=use_speed),
                                device=str(env.device))
        load_checkpoint(runner, checkpoint, load_optimizer=False)
        policies.append(runner.get_inference_policy(device=env.device))
    env.begin_demo(0.20, "flat")
    if use_recovery:
        ids = env.all_envs.reshape(-1, count)[:, len(MOTIONS):].reshape(-1)
        indices = [RECOVERY_TABLE_OFFSETS.index(pair) for pair in RECOVERY_OFFSETS]
        pose = env.recovery_poses[indices].repeat(len(args.checkpoints), 1)
        qpos = env.robot.get_qpos()[ids].clone()
        qpos[:, env.pose_indices[0]] = pose[:, 0] + 0.07
        qpos[:, env.pose_indices[1:]] = pose[:, 1:]
        env.robot.set_qpos(qpos, envs_idx=ids, zero_velocity=True)
        env.recovery_steps[ids] = 0

    dt = env.control_dt
    traces = {name: command_trace(segments, dt, env.commands) for name, segments in MOTIONS.items()}
    ticks = max(map(len, traces.values()))
    stopped = torch.zeros_like(env.episode_steps)
    recovered = torch.zeros_like(stopped)
    samples = []
    for tick in range(ticks):
        targets = torch.stack([traces[name][min(tick, len(traces[name]) - 1)] if name in traces
                               else env.commands.new_tensor([0.0, 0.0, 0.20]) for name in names])
        env.goals.copy_(targets.repeat(len(args.checkpoints), 1))
        lin_vel, ang_vel, _, _ = env._base_state()
        env._advance_commands(lin_vel, ang_vel)
        obs = env.get_observations()
        groups = [obs[i * count:(i + 1) * count] for i in range(len(policies))]
        actions = torch.cat([policy(group) for policy, group in zip(policies, groups, strict=True)])
        _, _, done, extras = env.step(actions)
        stopped.masked_fill_(done & (stopped == 0), tick + 1)
        if use_recovery:
            recovered.masked_fill_(extras["recovery_succeeded"] & (recovered == 0), tick + 1)
        vel, _, gravity, _ = env._base_state()
        heights, fore_aft = env._leg_projections(gravity)
        roll = torch.rad2deg(torch.atan2(-gravity[:, 1], -gravity[:, 2]))
        samples.append(torch.stack((vel[:, 0], extras["pitch_deg"], roll, heights.mean(1), fore_aft.mean(1),
                                    extras["wheel_contacts"].all(dim=1).float()), dim=1))
    samples = torch.stack(samples).cpu().numpy()
    stopped, recovered = stopped.tolist(), recovered.tolist()
    rows = []
    for i, checkpoint in enumerate(args.checkpoints):
        for j, name in enumerate(names):
            index = i * count + j
            row = {"checkpoint": str(checkpoint), "scenario": name, "passed": False}
            if name in traces:
                elapsed = len(traces[name])
                interrupted = stopped[index] if stopped[index] <= elapsed else 0
                if interrupted:
                    row["interrupted_s"] = interrupted * dt
                    rows.append(row)
                    continue
                row.update(score_motion(samples[:elapsed, index], MOTIONS[name], dt, interrupted))
                row["passed"] = (not interrupted and row["peak_abs_pitch_deg"] <= 8
                                 and row["peak_abs_roll_deg"] <= 8 and row["height_tail_max_error_cm"] <= 1.5
                                 and row["speed_tail_max_error_m_s"] <= 0.15
                                 and row["speed_zero_tail_max_m_s"] <= 0.08
                                 and (name != "cruise" or row["mean_axle_x_tail_p2p_cm"] <= 1.0))
            else:
                row["recovered_s"] = recovered[index] * dt if recovered[index] else ""
                row["interrupted_s"] = stopped[index] * dt if stopped[index] else ""
                row["passed"] = bool(recovered[index])
            rows.append(row)
        group = rows[-count:]
        print(f"{checkpoint.parent.parent.name}/{checkpoint.name}: "
              f"{sum(row['passed'] for row in group[:len(MOTIONS)])}/{len(MOTIONS)} motions; "
              f"{sum(row['passed'] for row in group[len(MOTIONS):])}/{len(names) - len(MOTIONS)} recovery starts",
              flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel-torque", type=float)
    parser.add_argument("--wheel-delay-steps", type=int, choices=range(10))
    parser.add_argument("checkpoints", type=Path, nargs="+")
    args = parser.parse_args()
    if args.wheel_torque is not None and (not isfinite(args.wheel_torque) or args.wheel_torque <= 0):
        parser.error("--wheel-torque must be finite and positive")
    gs.init(backend=gs.cuda, seed=123, logging_level="warning")
    try:
        compare(args)
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()

