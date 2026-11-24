"""Probe a scripted leg setup followed by learned balance on the V0.2 chassis."""

import argparse
import csv
from itertools import product
from math import ceil, cos, isfinite, radians
from pathlib import Path

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from configs.experiment import TRAINING_DIR, load_snapshot, runner_config
from main import load_checkpoint
from robots.wheel_leg_locomotion import WheelLegLocomotionEnv
from robots.wheel_leg_rescue import FirmwareRecovery
from robots.wheel_leg_stance import solve_stances
from tools.live_view import TrainingView


PITCHES = (-105.0, -85.0, -65.0, -45.0, -25.0, 25.0, 45.0, 65.0, 85.0, 105.0)


class RecoveryProbe(WheelLegLocomotionEnv):
    """Keep tipped starts intact while measuring a bounded recovery attempt."""

    def _termination(self, height, gravity, contact):
        super()._termination(height, gravity, contact)
        return torch.zeros_like(height, dtype=torch.bool), False

    def place_on_ground(self, angles):
        qpos = self.robot.get_qpos().clone()
        pitch = torch.deg2rad(angles) * 0.5
        qpos[:, self.base_quat] = torch.stack((pitch.cos(), torch.zeros_like(pitch),
                                               pitch.sin(), torch.zeros_like(pitch)), dim=1)
        self.robot.set_qpos(qpos, zero_velocity=True)
        qpos[:, self.pose_indices[0]] += 0.001 - self.robot.get_AABB()[:, 0, 2]
        self.robot.set_qpos(qpos, zero_velocity=True)
        if self.heading_hold_enabled:
            self.heading_target.copy_(self._yaw_from_quat(self.robot.get_quat()))


def load_policy(env, ppo, checkpoint):
    runner = OnPolicyRunner(env, runner_config(ppo), device=str(env.device))
    load_checkpoint(runner, checkpoint, load_optimizer=False)
    return runner.get_inference_policy(device=env.device)


def build_probe(checkpoint, num_envs, seconds=8.0):
    settings, run = load_snapshot(checkpoint.with_name("run_config.json"))
    task = settings["task"]
    locomotion = task["locomotion"]
    locomotion.update(terrain="plane", terrain_traversal=False, terrain_forward_limit=0.0)
    locomotion["low_recovery"]["flat_episode_fraction"] = 0.0
    model = TRAINING_DIR / run["model"]
    env = RecoveryProbe(model, num_envs, episode_seconds=seconds, task_config=task)
    env.begin_demo(env.height_low, None)
    return env, model, settings["ppo"]


@torch.inference_mode()
def probe(options):
    offsets = (-0.27, -0.24, -0.18, -0.11, -0.075, 0.0, 0.075, 0.11, 0.18, 0.24, 0.27)
    cases = list(product(PITCHES, offsets))
    env, model, ppo = build_probe(options.drive, len(cases))
    env.place_on_ground(env.commands.new_tensor([angle for angle, _ in cases]))
    drive_policy = load_policy(env, ppo, options.drive)
    recovery_policy = drive_policy
    if options.recovery is not None:
        recovery_settings, _ = load_snapshot(options.recovery.with_name("run_config.json"))
        recovery_policy = load_policy(env, recovery_settings["ppo"], options.recovery)

    stances = solve_stances(model, (env.height_low, env.height_high), env.wheel_radius,
                            wheel_offsets=[(offset, offset) for offset in offsets])
    leg_presets = env.commands.new_tensor(stances)[:, env.leg_columns]
    nominal = env._poses(env.commands[:, 2])[:, env.leg_columns]
    preset_destinations = leg_presets.repeat(len(PITCHES), 1) + env.leg_bias
    preset_targets = nominal + env.leg_bias
    preset_actions = torch.zeros_like(env.last_actions)

    applied_controls = env._apply_controls
    phase = torch.zeros(len(cases), dtype=torch.long, device=env.device)
    preset = phase == 0
    zero_wheel_torque = env.commands.new_zeros((len(cases), 2))
    assisted_speed = env.commands.new_zeros(len(cases))

    def apply_controls(actions):
        applied = applied_controls(actions)
        ids = env.all_envs[preset]
        if ids.numel():
            env.robot.control_dofs_position(preset_targets[ids], env.leg_dofs, envs_idx=ids)
            if options.wheel_assist:
                wheel_targets = assisted_speed[ids, None] * env.wheel_direction
                env.robot.control_dofs_velocity(wheel_targets, env.wheel_dofs, envs_idx=ids)
            else:
                env.robot.control_dofs_force(zero_wheel_torque[ids], env.wheel_dofs, envs_idx=ids)
        return torch.where(preset[:, None], preset_actions, applied)

    env._apply_controls = apply_controls
    start_recovery = torch.full_like(phase, -1)
    start_drive = torch.full_like(phase, -1)
    steady = torch.zeros_like(phase)
    survived = torch.ones_like(phase, dtype=torch.bool)
    first_body_contact = torch.zeros_like(survived)
    wheels_during_preset = torch.zeros_like(survived)
    peak_body_force = torch.zeros_like(env.commands[:, 0])
    best_preset_tilt = torch.full_like(peak_body_force, 180.0)
    initial_height = env.robot.get_pos()[:, 2].clone()
    max_preset_steps = round(1.5 / env.control_dt)
    max_recovery_steps = round(2.0 / env.control_dt)
    steady_steps = round(.24 / env.control_dt)
    drive_steps = round(.5 / env.control_dt)
    obs = env.get_observations()
    for tick in range(round(5.0 / env.control_dt)):
        preset = phase == 0
        recovering = phase == 1
        driving = phase == 2
        if tick >= 5:
            preset_targets += (preset_destinations - preset_targets).clamp(-.06, .06)
        preset_actions[:, :4] = (preset_targets - nominal - env.leg_bias) * env.leg_direction / env.leg_action_scale
        actions = recovery_policy(obs)
        if options.recovery is not None:
            actions = torch.where(driving[:, None], drive_policy(obs), actions)
        obs, _, _, _ = env.step(actions)
        vel, ang, gravity, _ = env._base_state()
        forces = env.robot.get_links_net_contact_force()
        loaded = (forces[:, env.wheel_links, 2] > 5.0).all(dim=1)
        if options.wheel_assist:
            limit = .65 / env.wheel_radius
            wheel_speed = env.scripted_wheel_speed(vel, ang, gravity, 0.0).clamp(-limit, limit)
            assisted_speed.copy_(torch.where(loaded, wheel_speed, 0.0))
            preset_actions[:, 4:] = assisted_speed[:, None] / 10.0
        body_force = forces[:, env.body_links].norm(dim=-1).amax(dim=1)
        body_contact = body_force > 10.0
        first_body_contact |= preset & body_contact
        wheels_during_preset |= preset & loaded
        torch.maximum(peak_body_force, torch.where(preset, body_force, 0.0), out=peak_body_force)
        tilt = -gravity[:, 2]
        angle = torch.rad2deg(tilt.clamp(-1.0, 1.0).acos())
        torch.minimum(best_preset_tilt, torch.where(preset, angle, 180.0), out=best_preset_tilt)
        can_recover = preset & (tick >= 5) & loaded & (tilt > cos(radians(35.0)))
        phase.masked_fill_(preset & ~can_recover & (tick + 1 >= max_preset_steps), 3)
        phase.masked_fill_(can_recover, 1)
        start_recovery.masked_fill_(can_recover, tick + 1)
        stable = (recovering & loaded & ~body_contact & (tilt > cos(radians(10.0)))
                  & (ang.norm(dim=1) < .8)
                  & ((env.body_clearance - env.height_low - env.body_height_offset).abs() < .04))
        steady = torch.where(stable, steady + 1, 0)
        can_drive = stable & (steady >= steady_steps)
        phase.masked_fill_(recovering & ~can_drive & (tick + 1 - start_recovery >= max_recovery_steps), 3)
        phase.masked_fill_(can_drive, 2)
        start_drive.masked_fill_(can_drive, tick + 1)
        survived &= ~(driving & (tick - start_drive < drive_steps)
                      & (body_contact | (tilt < cos(radians(25.0)))))

    completed = (start_drive >= 0) & (tick - start_drive >= drive_steps) & survived
    fields = ("initial_pitch_deg", "wheel_offset_m", "initial_height_m", "body_touched",
              "peak_preset_body_force_n", "both_wheels_loaded_during_preset", "best_preset_tilt_deg",
              "recovery_step", "drive_step", "handover_ok")
    with options.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(zip((case[0] for case in cases), (case[1] for case in cases),
                             initial_height.tolist(), first_body_contact.tolist(), peak_body_force.tolist(),
                             wheels_during_preset.tolist(), best_preset_tilt.tolist(),
                             start_recovery.tolist(), start_drive.tolist(), completed.tolist()))
    print(f"All starts ({len(cases)}): reach recovery policy {int((start_recovery >= 0).sum())}; "
          f"stand and hand back to drive {int(completed.sum())}", flush=True)
    print(f"Starts with body contact ({int(first_body_contact.sum())}): "
          f"reach recovery policy {int((first_body_contact & (start_recovery >= 0)).sum())}; "
          f"stand and hand back to drive {int((first_body_contact & completed).sum())}", flush=True)


@torch.inference_mode()
def firmware_probe(options, live_view=None):
    angles = (options.pitch,) if options.pitch is not None else PITCHES
    env, model, ppo = build_probe(options.drive, len(angles), options.seconds + 1.0)
    env.place_on_ground(env.commands.new_tensor(angles))
    if live_view is not None:
        env.on_step = lambda reset: live_view.capture(env, model, reset)
    policy = load_policy(env, ppo, options.drive)
    controller = FirmwareRecovery(env, model)
    springs = env._apply_springs
    valid_at_every_step = torch.ones_like(controller.enabled)
    peak_length = torch.zeros_like(controller.wait)
    firmware_active = True

    def apply_firmware():
        springs()
        if firmware_active:
            length, valid = controller.advance()
            valid_at_every_step.logical_and_(~controller.enabled | valid.all(dim=1))
            torch.maximum(peak_length, torch.where(controller.enabled, length.amax(dim=1), 0.0), out=peak_length)

    env._apply_springs = apply_firmware
    reached = torch.full((len(angles), 4), -1, dtype=torch.long, device=env.device)
    touched = torch.zeros_like(controller.enabled)
    supported = torch.zeros_like(controller.enabled)
    balanced_after_handover = torch.ones_like(controller.enabled)
    first_drop = torch.full_like(controller.mode, -1)
    readiness_streak = torch.zeros_like(controller.mode)
    handover_ticks = max(1, ceil(options.handover_hold / env.control_dt))
    evaluation_ticks = ceil(.5 / env.control_dt)
    ticks = round(options.seconds / env.control_dt)
    obs = env.get_observations()
    frames = []
    frame_stride = max(1, round(1.0 / (25 * env.control_dt)))
    if options.video is not None:
        frames.append((env.robot.get_qpos()[0].cpu().numpy().copy(), 0.0, 0.0, 0.0))
    for tick in range(1, ticks + 1):
        actions = policy(obs).masked_fill(controller.enabled[:, None], 0.0)
        obs, _, _, _ = env.step(actions)
        vel, ang, gravity, _ = env._base_state()
        forces = env.robot.get_links_net_contact_force()
        loaded = (forces[:, env.wheel_links, 2] > 5.0).all(dim=1)
        body_contact = (forces[:, env.body_links].norm(dim=-1) > 10.0).any(dim=1)
        touched |= controller.enabled & body_contact
        supported |= controller.enabled & loaded
        mode = controller.mode
        safe_to_handover = (controller.enabled & controller.ready & loaded & ~body_contact
                            & (-gravity[:, 2] > cos(radians(12))) & (ang.norm(dim=1) < .8))
        readiness_streak = torch.where(safe_to_handover, readiness_streak + 1, 0)
        readiness = readiness_streak >= handover_ticks
        milestones = torch.stack(((mode == controller.SWING) | (mode == controller.SINGLE),
                                  (mode == controller.RETRACT) | (mode == controller.BALANCE),
                                  mode == controller.BALANCE, readiness), dim=1)
        reached.masked_fill_(milestones & (reached < 0), tick)
        drive_age = tick - reached[:, 3]
        grace_period = ~controller.enabled & (drive_age <= evaluation_ticks)
        failed = body_contact | (-gravity[:, 2] < cos(radians(25)))
        first_drop = torch.where(grace_period & failed & (first_drop < 0), drive_age, first_drop)
        balanced_after_handover &= ~(grace_period & failed)
        controller.enabled &= ~readiness
        firmware_active = firmware_active and bool(controller.enabled.any())
        if options.video is not None and tick % frame_stride == 0:
            frames.append((env.robot.get_qpos()[0].cpu().numpy().copy(), tick * env.control_dt,
                           0.0, vel[0, 0].item()))
        if tick % round(1.0 / env.control_dt) == 0:
            print(f"t={tick * env.control_dt:.0f}s, swing={int((reached[:, 0] >= 0).sum())}, "
                  f"getup={int((reached[:, 1] >= 0).sum())}, drive={int((reached[:, 3] >= 0).sum())}", flush=True)
    passed = (reached[:, 3] >= 0) & (ticks - reached[:, 3] >= evaluation_ticks) & balanced_after_handover
    times = torch.where(reached >= 0, reached * env.control_dt, -1.0).tolist()
    drop_times = torch.where(first_drop >= 0, first_drop * env.control_dt, -1.0).tolist()
    with options.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("initial_pitch_deg", "body_touched", "both_wheels_loaded", "peak_leg_length_m",
                         "valid_vmc", "swing_s", "getup_s", "balance_s", "drive_s",
                         "dropped_after_drive_s", "handover_ok"))
        for row in zip(angles, touched.tolist(), supported.tolist(), peak_length.tolist(),
                       valid_at_every_step.tolist(), times, drop_times, passed.tolist(), strict=True):
            angle, body, wheels, length, valid, milestones, dropped, success = row
            writer.writerow((angle, body, wheels, length, valid, *milestones, dropped, success))
    print(f"All starts: {int(passed.sum())}/{len(angles)} handed over to drive. "
          f"Body-contact starts: {int((passed & touched).sum())}/{int(touched.sum())}. "
          f"CAD VMC remained finite in {int(valid_at_every_step.sum())}/{len(angles)}.", flush=True)
    if frames:
        from tools.render_wheelbipe import record

        record(model, env.robot, frames, options.video, round(1 / (env.control_dt * frame_stride)),
               env.label, "RESCUE / GETUP / PPO")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, help="Use the drive checkpoint as the initial comparison if omitted")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel-assist", action="store_true", help="Test scripted wheel balance during leg setup")
    parser.add_argument("--firmware", action="store_true", help="Run the torque-based firmware RESCUE and GETUP port")
    parser.add_argument("--seconds", type=float, default=8.0, help="Firmware probe duration")
    parser.add_argument("--pitch", type=float, help="Test a single pitch angle in firmware mode")
    parser.add_argument("--handover-hold", type=float, default=0.0, help="Require continuous readiness for this long")
    parser.add_argument("--live-port", type=int, help="Loopback monitor port in firmware mode")
    parser.add_argument("--video", type=Path, help="Save a replay of a single-pitch firmware run")
    options = parser.parse_args()
    if options.firmware and (options.recovery or options.wheel_assist):
        parser.error("--firmware uses the ported get-up controller and the drive checkpoint")
    if not options.firmware and (options.pitch is not None or options.live_port is not None or options.handover_hold):
        parser.error("--pitch, --live-port and --handover-hold require --firmware")
    if options.video is not None and options.pitch is None:
        parser.error("--video requires a single --pitch")
    if not isfinite(options.seconds) or options.seconds < 1.0:
        parser.error("--seconds must be finite and at least 1 s")
    if not isfinite(options.handover_hold) or not 0 <= options.handover_hold < options.seconds:
        parser.error("--handover-hold must be nonnegative and shorter than the probe")
    if options.pitch is not None and (not isfinite(options.pitch) or abs(options.pitch) > 180.0):
        parser.error("--pitch must be between -180 and 180 degrees")
    gs.init(backend=gs.cuda, seed=42, logging_level="warning")
    try:
        if not options.firmware:
            probe(options)
        elif options.live_port is None:
            firmware_probe(options)
        else:
            with TrainingView(options.live_port, iterations=None, robot="Wheel Leg V0.2",
                              num_envs=1 if options.pitch is not None else len(PITCHES), stream=True, fps=10) as live:
                print(f"Live monitor: http://127.0.0.1:{options.live_port}/", flush=True)
                firmware_probe(options, live)
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
