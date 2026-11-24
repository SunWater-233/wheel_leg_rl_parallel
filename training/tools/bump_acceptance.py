
"""Verify bump crossing, upright recovery, braking to zero and reverse tracking."""

import argparse
import csv
from dataclasses import asdict, dataclass, field
import json
from math import isfinite
from pathlib import Path

APPROACH_TIMEOUT_S = 6.0
BRAKE_TIMEOUT_S = 2.0
REVERSE_TIMEOUT_S = 2.0
HOLD_S = 0.24
UPRIGHT_DEG = 8.0
STOP_SPEED_M_S = 0.08
ZERO_COMMAND_M_S = 0.03
REVERSE_ERROR_M_S = 0.15
BAR_CENTER_X_M = 0.87
BAR_TRACK_WIDTH_M = 1.80


@dataclass
class Hold:
    since_s: float | None = None
    passed_s: float | None = None

    def update(self, time_s, ok):
        if not ok:
            self.since_s = None
            return
        if self.since_s is None:
            self.since_s = time_s
        if self.passed_s is None and time_s - self.since_s >= HOLD_S - 1e-9:
            self.passed_s = time_s


@dataclass(frozen=True)
class Sample:
    time_s: float
    target_m_s: float
    vx_m_s: float
    pitch_deg: float
    roll_deg: float
    powered_wheels_loaded: bool
    wheel_touched_bar: bool
    beyond_bar: bool


@dataclass
class BumpAcceptance:
    approach_speed: float
    reverse_speed: float
    phase: str = "approach"
    contact_s: float | None = None
    crossing_s: float | None = None
    reverse_start_s: float | None = None
    upright: Hold = field(default_factory=Hold)
    stopped: Hold = field(default_factory=Hold)
    reversed: Hold = field(default_factory=Hold)
    finished: bool = False
    interrupted_s: float | None = None

    @property
    def command(self):
        if self.phase == "approach":
            return self.approach_speed
        if self.phase == "brake":
            return 0.0
        return -self.reverse_speed

    def update(self, sample):
        t = sample.time_s
        upright = (sample.powered_wheels_loaded and abs(sample.pitch_deg) <= UPRIGHT_DEG
                   and abs(sample.roll_deg) <= UPRIGHT_DEG)
        if self.phase == "approach":
            if self.contact_s is None and sample.wheel_touched_bar:
                self.contact_s = t
            if self.contact_s is not None and sample.beyond_bar and t <= APPROACH_TIMEOUT_S:
                self.crossing_s = t
                self.phase = "brake"
            elif t >= APPROACH_TIMEOUT_S:
                self.finished = True
        elif self.phase == "brake":
            stopped = (upright and abs(sample.target_m_s) <= ZERO_COMMAND_M_S
                       and abs(sample.vx_m_s) <= STOP_SPEED_M_S)
            self.upright.update(t, upright)
            self.stopped.update(t, stopped)
            if ((self.upright.passed_s is not None and self.stopped.passed_s is not None)
                    or t - self.crossing_s >= BRAKE_TIMEOUT_S - 1e-9):
                self.reverse_start_s = t
                self.phase = "reverse"
        else:
            reverse = (upright and abs(sample.target_m_s + self.reverse_speed) <= ZERO_COMMAND_M_S
                       and sample.vx_m_s < 0 and abs(sample.vx_m_s - sample.target_m_s) <= REVERSE_ERROR_M_S)
            self.reversed.update(t, reverse)
            self.finished = (self.reversed.passed_s is not None
                             or t - self.reverse_start_s >= REVERSE_TIMEOUT_S - 1e-9)

    def result(self):
        milestones = {
            "wheel_contact_s": self.contact_s,
            "crossed_s": self.crossing_s,
            "upright_s": self.upright.passed_s,
            "stopped_s": self.stopped.passed_s,
            "reverse_command_s": self.reverse_start_s,
            "reverse_tracking_s": self.reversed.passed_s,
            "interrupted_s": self.interrupted_s,
        }
        passed = (self.finished and self.interrupted_s is None
                  and all(t is not None for name, t in milestones.items() if name != "interrupted_s"))
        return {"passed": passed, **milestones}


def run(args):
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner

    from configs.experiment import TRAINING_DIR, load_snapshot, runner_config
    from main import load_checkpoint
    from robots.wheel_leg_locomotion import WheelLegLocomotionEnv

    class BumpCourse(WheelLegLocomotionEnv):
        def _add_ground(self, material, viewer):
            super()._add_ground(material, viewer)
            self.bar = self.scene.add_entity(
                gs.morphs.Box(pos=(BAR_CENTER_X_M, 0.0, args.bar_height / 2),
                              size=(args.bar_width, BAR_TRACK_WIDTH_M, args.bar_height), fixed=True,
                              visualization=False),
                material=material,
            )

    settings, saved_run = load_snapshot(args.checkpoint.with_name("run_config.json"))
    task = settings["task"]
    locomotion = task["locomotion"]
    if not locomotion["enabled"]:
        raise ValueError("checkpoint must use the V0.2 locomotion task")
    if not task["forward_command"][0] <= -args.reverse_speed < args.speed <= task["forward_command"][1]:
        raise ValueError("approach and reverse speeds must be inside the checkpoint command range")
    if not locomotion["height_range"][0] <= args.leg_height <= locomotion["height_range"][1]:
        raise ValueError("leg height must be inside the checkpoint command range")
    locomotion.update(terrain="plane", terrain_traversal=False)
    locomotion["low_recovery"]["flat_episode_fraction"] = 0.0
    model = TRAINING_DIR / saved_run["model"]
    gs.init(backend=gs.cuda, seed=123, logging_level="warning")
    try:
        env = BumpCourse(model, 1, episode_seconds=12.0, task_config=task)
        runner = OnPolicyRunner(env, runner_config(settings["ppo"]), device=str(env.device))
        load_checkpoint(runner, args.checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        return drive(env, policy, args)
    finally:
        gs.destroy()


def drive(env, policy, args):
    import torch

    acceptance = BumpAcceptance(args.speed, args.reverse_speed)
    env.begin_demo(args.leg_height, "flat")
    wheel_links = [env.robot.get_link(name).idx for name in env.wheel_bodies]
    far_edge = BAR_CENTER_X_M + args.bar_width / 2
    rows = []
    with torch.inference_mode():
        for tick in range(1, round(11 / env.control_dt) + 1):
            command = acceptance.command
            phase = acceptance.phase
            env.set_commands(command, 0.0, args.leg_height)
            target = env.commands[0, 0].item()
            _, _, done, _ = env.step(policy(env.get_observations()))
            time_s = tick * env.control_dt
            if done[0]:
                acceptance.interrupted_s = time_s
                acceptance.finished = True
                break

            vel, _, gravity, _ = env._base_state()
            gx, gy, gz = gravity[0]
            pitch_deg = torch.rad2deg(torch.atan2(gx, torch.hypot(gy, gz))).item()
            roll_deg = torch.rad2deg(torch.atan2(-gy, -gz)).item()
            contacts = env.robot.get_links_net_contact_force()
            wheels_loaded = bool((contacts[0, env.wheel_links, 2] > 5.0).all())
            axle = torch.stack([wheel.get_anchor_pos()[0] for wheel in env.wheels])
            in_track = bool((axle[:, 1].abs() < BAR_TRACK_WIDTH_M / 2).all())
            beyond_bar = in_track and bool((axle[:, 0] > far_edge).all())
            wheel_touched_bar = False
            if phase == "approach":
                collision = env.robot.get_contacts(with_entity=env.bar)
                hit = collision["valid_mask"]
                for link in wheel_links:
                    wheel_touched_bar |= bool((hit & ((collision["link_a"] == link)
                                                     | (collision["link_b"] == link))).any())
            sample = Sample(time_s, target, vel[0, 0].item(), pitch_deg, roll_deg,
                            wheels_loaded, wheel_touched_bar, beyond_bar)
            acceptance.update(sample)
            rows.append({"phase": phase, "request_m_s": command, **asdict(sample)})
            if acceptance.finished:
                break

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields = ("phase", "request_m_s", *Sample.__dataclass_fields__)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return acceptance.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write the 50 Hz command and state trace to CSV")
    parser.add_argument("--speed", type=float, default=0.8, help="Approach speed in m/s")
    parser.add_argument("--reverse-speed", type=float, default=0.5, help="Reverse speed magnitude in m/s")
    parser.add_argument("--leg-height", type=float, default=0.20, help="Target hip-to-wheel height in m")
    parser.add_argument("--bar-height", type=float, default=0.04, help="Bump height in m")
    parser.add_argument("--bar-width", type=float, default=0.12, help="Bump length along travel in m")
    args = parser.parse_args()
    for name in ("speed", "reverse_speed", "leg_height", "bar_height", "bar_width"):
        if not isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.reverse_speed <= ZERO_COMMAND_M_S:
        parser.error("--reverse-speed must exceed the 0.03 m/s zero-command tolerance")
    try:
        result = run(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

