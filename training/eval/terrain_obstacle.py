"""Compare locomotion checkpoints on a rounded speed bump and one-wheel disturbances."""

import argparse
import csv
from dataclasses import asdict
from itertools import product
from math import isfinite
from pathlib import Path

import genesis as gs
import numpy as np
import torch

from configs.schema import TRAINING_DIR
from eval.bump import (
    APPROACH_TIMEOUT_S,
    BAR_CENTER_X_M,
    BRAKE_TIMEOUT_S,
    REVERSE_TIMEOUT_S,
    BumpAcceptance,
    Sample,
)
from eval.harness import configure_eval_locomotion, load_eval_batch, load_eval_policies
from tasks.locomotion_env import WheelLegLocomotionEnv
from tasks.terrains import raised_cosine_profile

GRID_M = 0.01
VERTICAL_SCALE_M = 0.0001
FIELD_SINK_M = 0.004
LANE_HALF_WIDTH_M = 0.80
LANE_SPACING_M = 2.0
EDGE_TAPER_M = 0.05
SCENARIOS = ("both", "left", "right")


def speed_bump_field(height, length):
    """Sample a raised cosine and its two mirrored half-width profiles."""
    half_cells = int(np.ceil(length / (2 * GRID_M))) + 2
    x = np.arange(-half_cells, half_cells + 1) * GRID_M
    y_start = -LANE_HALF_WIDTH_M - 2 * GRID_M
    y_end = 2 * LANE_SPACING_M - y_start
    y = y_start + np.arange(round((y_end - y_start) / GRID_M) + 1) * GRID_M
    x_plate = raised_cosine_profile(x, length)
    full = (LANE_HALF_WIDTH_M - np.abs(y)) / EDGE_TAPER_M
    positive_y = y - LANE_SPACING_M
    positive = np.minimum(positive_y, LANE_HALF_WIDTH_M - positive_y) / EDGE_TAPER_M
    negative_y = 2 * LANE_SPACING_M - y
    negative = np.minimum(negative_y, LANE_HALF_WIDTH_M - negative_y) / EDGE_TAPER_M
    plate = np.clip(np.maximum.reduce((full, positive, negative)), 0.0, 1.0)
    lateral = 0.5 * (1 - np.cos(np.pi * plate))
    shape = (height + FIELD_SINK_M) * x_plate[:, None] * lateral[None, :]
    return shape / VERTICAL_SCALE_M, (BAR_CENTER_X_M + x[0], y_start, -FIELD_SINK_M)


def compare(args):
    class ObstacleCourse(WheelLegLocomotionEnv):
        def _add_ground(self, material, viewer):
            super()._add_ground(material, viewer)
            self.bump = self.scene.add_entity(
                gs.morphs.Terrain(pos=origin, height_field=field, horizontal_scale=GRID_M,
                                  vertical_scale=VERTICAL_SCALE_M, visualization=viewer),
                material=material,
            )

    batch = load_eval_batch(args.checkpoints)
    settings, saved = batch.first_settings, batch.first_run
    task = settings["task"]
    locomotion = task["locomotion"]
    configure_eval_locomotion(locomotion, batch)
    if not locomotion["enabled"]:
        raise ValueError("checkpoints must use the V0.2 locomotion task")
    low_speed, high_speed = task["forward_command"]
    if not low_speed <= -args.reverse_speed <= high_speed:
        raise ValueError("reverse speed must be inside the reference checkpoint command range")
    if not low_speed <= min(args.speed) <= max(args.speed) <= high_speed:
        raise ValueError("approach speed must be inside the reference checkpoint command range")
    if not locomotion["height_range"][0] <= args.leg_height <= locomotion["height_range"][1]:
        raise ValueError("leg height must be inside the reference checkpoint command range")
    locomotion.update(terrain="plane", terrain_traversal=False, wheel_target_delay_steps=args.wheel_delay_steps)
    locomotion["low_recovery"]["flat_episode_fraction"] = 0.0
    field, origin = speed_bump_field(args.bump_height, args.bump_length)
    cases = tuple(product(SCENARIOS, args.speed))
    env = ObstacleCourse(TRAINING_DIR / saved["model"], len(args.checkpoints) * len(cases),
                         episode_seconds=12.0, task_config=task)
    policies = load_eval_policies(env, batch, ppo_override=settings["ppo"])
    env.begin_demo(args.leg_height, "flat")
    left_y = env.wheels[0].get_anchor_pos()[0, 1] - env.wheels[1].get_anchor_pos()[0, 1]
    left_lane = LANE_SPACING_M if left_y.item() > 0.0 else 2 * LANE_SPACING_M
    centers = {"both": 0.0, "left": left_lane, "right": 3 * LANE_SPACING_M - left_lane}
    trials = tuple(product(args.checkpoints, cases))
    lane_y = env.commands.new_tensor([centers[side] for _, (side, _) in trials])
    qpos = env.robot.get_qpos().clone()
    qpos[:, env.base_xy[1]] = lane_y
    env.robot.set_qpos(qpos, zero_velocity=True)
    return trials, *drive(env, policies, trials, lane_y, args)


@torch.inference_mode()
def drive(env, policies, trials, lane_y, args):
    count = len(trials) // len(policies)
    tests = [BumpAcceptance(speed, args.reverse_speed) for _, (_, speed) in trials]
    rows = [[] for _ in trials]
    wheel_links = [env.robot.get_link(name).idx for name in env.wheel_bodies]
    far_edge = BAR_CENTER_X_M + args.bump_length / 2
    ticks = round((APPROACH_TIMEOUT_S + BRAKE_TIMEOUT_S + REVERSE_TIMEOUT_S + env.control_dt) / env.control_dt)
    for tick in range(1, ticks + 1):
        command = [test.command if not test.finished else 0.0 for test in tests]
        env.goals[:, 0] = env.commands.new_tensor(command)
        lin_vel, ang_vel, _, _ = env._base_state()
        env._advance_commands(lin_vel, ang_vel)
        obs = env.get_observations()
        actions = torch.cat([policy(obs[i * count:(i + 1) * count]) for i, policy in enumerate(policies)])
        _, _, done, extras = env.step(actions)
        vel, _, _, _ = env._base_state()
        axle = torch.stack([wheel.get_anchor_pos() for wheel in env.wheels], dim=1)
        beyond = ((axle[..., 0] > far_edge) & ((axle[..., 1] - lane_y[:, None]).abs() < LANE_HALF_WIDTH_M)).all(1)
        contacts = env.robot.get_contacts(with_entity=env.bump)
        elevated = contacts["valid_mask"] & (contacts["position"][..., 2] > 0.001)
        hits = [(elevated & ((contacts["link_a"] == link) | (contacts["link_b"] == link))).any(1)
                for link in wheel_links]
        wheel_load = env.robot.get_links_net_contact_force()[:, env.wheel_links, 2]
        data = torch.stack((env.commands[:, 0], vel[:, 0], extras["pitch_deg"], extras["posture"][:, 0],
                            wheel_load[:, 0], wheel_load[:, 1], *hits, beyond, done,
                            env.robot.get_pos()[:, 1] - lane_y), dim=1).tolist()
        time_s = tick * env.control_dt
        for i, ((_, (side, _)), test, values) in enumerate(zip(trials, tests, data, strict=True)):
            if test.finished:
                continue
            target, vx, pitch, roll, left_load, right_load, hit_l, hit_r, crossed, reset, drift = values
            if reset:
                test.interrupted_s = time_s
                test.finished = True
                continue
            expected_hit = {"both": hit_l or hit_r, "left": hit_l, "right": hit_r}[side]
            sample = Sample(time_s, target, vx, pitch, roll, left_load > 5.0 and right_load > 5.0,
                            bool(expected_hit), bool(crossed))
            rows[i].append({"phase": test.phase, "request_m_s": command[i], **asdict(sample),
                            "left_contact": bool(hit_l), "right_contact": bool(hit_r),
                            "left_load_n": left_load, "right_load_n": right_load, "lateral_m": drift})
            test.update(sample)
        if all(test.finished for test in tests):
            break
    return tests, rows


def write_results(args, trials, tests, traces):
    results = []
    for (checkpoint, (side, speed)), test, trace in zip(trials, tests, traces, strict=True):
        approach = [row for row in trace if row["phase"] == "approach"]
        hit_l = any(row["left_contact"] for row in approach)
        hit_r = any(row["right_contact"] for row in approach)
        isolated = {"both": True, "left": hit_l and not hit_r, "right": hit_r and not hit_l}[side]
        result = test.result()
        result["passed"] &= isolated
        results.append({"checkpoint": str(checkpoint), "scenario": side, "approach_m_s": speed,
                        "leg_height_m": args.leg_height, "bump_height_m": args.bump_height,
                        "bump_length_m": args.bump_length, "wheel_delay_ms": 2 * args.wheel_delay_steps,
                        **result, "approach_left_contact": hit_l, "approach_right_contact": hit_r,
                        "peak_pitch_deg": max((abs(row["pitch_deg"]) for row in trace), default=None),
                        "peak_roll_deg": max((abs(row["roll_deg"]) for row in trace), default=None),
                        "peak_lateral_m": max((abs(row["lateral_m"]) for row in trace), default=None)})
        print(f"{checkpoint.name} {side} {speed:g} m/s: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    if args.trace:
        rows = [{"checkpoint": str(checkpoint), "scenario": side, "approach_m_s": speed, **row}
                for (checkpoint, (side, speed)), trace in zip(trials, traces, strict=True) for row in trace]
        if rows:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            with args.trace.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    return all(row["passed"] for row in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True, help="Write one acceptance row per course and checkpoint")
    parser.add_argument("--trace", type=Path, help="Write the 50 Hz command, posture and wheel contact trace")
    parser.add_argument("--speed", type=float, nargs="+", default=[0.55, 0.8], help="Approach speeds in m/s")
    parser.add_argument("--reverse-speed", type=float, default=0.5, help="Reverse speed magnitude in m/s")
    parser.add_argument("--leg-height", type=float, default=0.20, help="Target hip-to-wheel height in m")
    parser.add_argument("--bump-height", type=float, default=0.04, help="Crest height in m")
    parser.add_argument("--bump-length", type=float, default=0.30, help="Length along travel, in m")
    parser.add_argument("--wheel-delay-steps", type=int, choices=range(10), default=5)
    args = parser.parse_args()
    for value in (*args.speed, args.reverse_speed, args.leg_height, args.bump_height, args.bump_length):
        if not isfinite(value) or value <= 0:
            parser.error("speeds, leg height and bump dimensions must be finite and positive")
    if args.bump_length < 4 * GRID_M:
        parser.error("the bump length must span at least four height-field cells")
    if args.reverse_speed <= 0.03:
        parser.error("reverse speed must exceed the zero-command tolerance of 0.03 m/s")
    gs.init(backend=gs.cuda, seed=123, logging_level="warning")
    try:
        trials, tests, traces = compare(args)
        passed = write_results(args, trials, tests, traces)
    finally:
        gs.destroy()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
