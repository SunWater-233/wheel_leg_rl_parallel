"""Load experiment settings and record each training run."""

from collections.abc import Sequence
from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from pathlib import Path
import tomllib


TRAINING_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("flat.toml")
SNAPSHOT_NAME = "run_config.json"
MAX_FORWARD_SPEED = 2.5
YAW_REFERENCE = 11.0
HEIGHT_LIMITS = (0.17, 0.33)
MAX_HEIGHT_RATE = 1.2
MAX_FORWARD_ACCELERATION = 4.0


def _apply_overrides(settings, overrides, section=""):
    if not isinstance(overrides, dict):
        raise ValueError(f"Expected a table for {section or 'the experiment configuration'}")
    for key, value in overrides.items():
        name = f"{section}.{key}" if section else key
        if key not in settings:
            raise ValueError(f"Unknown setting: {name}")
        default = settings[key]
        if isinstance(default, dict):
            _apply_overrides(default, value, name)
        elif type(value) is type(default) or (type(default) is float and type(value) is int):
            settings[key] = value
        else:
            raise ValueError(f"Invalid value for {name}: {value!r}")


def _validate_range(settings, name):
    limits = settings[name]
    if (len(limits) != 2 or any(type(v) not in (int, float) or not isfinite(v) for v in limits)
            or limits[0] > limits[1]):
        raise ValueError(f"{name} must contain two finite limits in ascending order")
    return limits


def normalize_range_spec(value, *, name="range", limit=None):
    """Normalize a [low, high] pair or list of [low, high] pairs into sorted, merged tuples."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a [low, high] pair or a list of [low, high] pairs")
    items = list(value)
    if not items:
        raise ValueError(f"{name} cannot be empty")
    if len(items) == 2 and all(isinstance(v, Real) and not isinstance(v, bool) for v in items):
        raw_ranges = [(float(items[0]), float(items[1]))]
    else:
        raw_ranges = []
        for entry in items:
            if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) != 2:
                raise ValueError(f"Invalid range entry in {name}: {entry!r}")
            low, high = entry
            if (not isinstance(low, Real) or isinstance(low, bool)
                    or not isinstance(high, Real) or isinstance(high, bool)):
                raise ValueError(f"Invalid range bounds in {name}: {entry!r}")
            raw_ranges.append((float(low), float(high)))

    normalized = []
    for low, high in raw_ranges:
        if not (isfinite(low) and isfinite(high)):
            raise ValueError(f"{name} bounds must be finite")
        if low > high:
            raise ValueError(f"{name} bounds must be in ascending order: [{low}, {high}]")
        if limit is not None and (low < -limit or high > limit):
            raise ValueError(f"{name} bounds [{low}, {high}] exceed ±{limit:g}")
        normalized.append((low, high))

    normalized.sort(key=lambda pair: pair[0])
    merged = []
    for low, high in normalized:
        if not merged or low > merged[-1][1] + 1e-6:
            merged.append([low, high])
        else:
            merged[-1][1] = max(merged[-1][1], high)
    return tuple((pair[0], pair[1]) for pair in merged)


def _validate_scut_reward(reward):
    divisors = {
        "tracking": {"forward_error_width", "yaw_error_width", "pitch_width", "roll_width",
                     "tilt_speed_width", "height_error_width", "ground_pitch_error_scale", "ground_pitch_rate_scale",
                     "flat_roll_error_scale", "flat_roll_rate_scale"},
        "braking": {"pitch_rate_scale", "recovery_pitch_scale", "recovery_speed_scale", "distance_scale"},
        "airborne": {"wheel_torque_scale", "wheel_speed_scale", "centering_width"},
    }
    for section, settings in reward.items():
        for name, value in settings.items():
            path = f"task.scut_rough_reward.{section}.{name}"
            if not isfinite(value) or value < 0:
                raise ValueError(f"{path} must be finite and nonnegative")
            if value == 0 and name in divisors.get(section, ()):
                raise ValueError(f"{path} must be positive")
    tracking = reward["tracking"]
    if tracking["vel_height_gate_zero_error"] < tracking["vel_height_gate_full_error"]:
        raise ValueError("task.scut_rough_reward.tracking.vel_height_gate_zero_error must be at least "
                         "vel_height_gate_full_error")
    airborne = reward["airborne"]
    if airborne["knee_margin_fraction"] > 0.5:
        raise ValueError("task.scut_rough_reward.airborne.knee_margin_fraction must be at most 0.5")
    if airborne["post_contact_end_seconds"] < airborne["wheel_contact_transition_seconds"]:
        raise ValueError("task.scut_rough_reward.airborne.post_contact_end_seconds must not precede "
                         "wheel_contact_transition_seconds")
    braking = reward["braking"]
    if braking["enabled"] and braking["start_speed"] <= braking["zero_speed"]:
        raise ValueError("Braking start_speed must exceed zero_speed")
    if braking["preserve_across_airborne"] and not braking["enabled"]:
        raise ValueError("Preserving braking state requires enabled braking cost")


def _validate_settings(settings):
    task = settings["task"]
    _validate_scut_reward(task["scut_rough_reward"])
    termination = task["termination"]
    instant_pitch = termination["instant_pitch_deg"]
    sustained_pitch = termination["sustained_pitch_deg"]
    sustained_seconds = termination["sustained_pitch_seconds"]
    if not isfinite(instant_pitch) or not 0.0 < instant_pitch <= 90.0:
        raise ValueError("task.termination.instant_pitch_deg must be in (0, 90]")
    if not isfinite(sustained_pitch) or not 0.0 < sustained_pitch <= instant_pitch:
        raise ValueError("task.termination.sustained_pitch_deg must be in (0, instant_pitch_deg]")
    if not isfinite(sustained_seconds) or sustained_seconds < 0.0:
        raise ValueError("task.termination.sustained_pitch_seconds must be finite and nonnegative")
    if not isfinite(task["episode_seconds"]) or task["episode_seconds"] <= 0:
        raise ValueError("task.episode_seconds must be finite and positive")
    if not isfinite(task["friction"]) or task["friction"] < 0:
        raise ValueError("task.friction must be finite and nonnegative")
    for name in ("forward_command", "yaw_command"):
        _validate_range(task, name)
    for name, std in task["observation_noise"].items():
        if not isfinite(std) or std < 0:
            raise ValueError(f"task.observation_noise.{name} must be finite and nonnegative")

    locomotion = task["locomotion"]
    if task["scut_rough_reward"]["tracking"]["use_projected_leg_height"]:
        if not locomotion["enabled"] or locomotion["terrain"] != "plane":
            raise ValueError("Projected leg height tracking requires plane locomotion")
    if locomotion["critic_terrain_observations"] and not locomotion["enabled"]:
        raise ValueError("Terrain observations for the critic require enabled locomotion")
    if locomotion["actor_true_forward_speed"] and not locomotion["enabled"]:
        raise ValueError("True forward speed for the actor requires enabled locomotion")
    history_steps = locomotion["actor_history_steps"]
    if type(history_steps) is not int or not 1 <= history_steps <= 20:
        raise ValueError("actor_history_steps must be an integer in [1, 20]")
    if history_steps > 1 and not locomotion["enabled"]:
        raise ValueError("Actor observation history requires enabled locomotion")
    heading_hold = locomotion["heading_hold"]
    if heading_hold["enabled"] and not locomotion["enabled"]:
        raise ValueError("Heading hold requires enabled locomotion")
    for name in ("gain", "max_yaw_rate"):
        if not isfinite(heading_hold[name]) or heading_hold[name] <= 0.0:
            raise ValueError(f"task.locomotion.heading_hold.{name} must be finite and positive")
    if not 0.0 <= heading_hold["deadband_deg"] < 180.0:
        raise ValueError("task.locomotion.heading_hold.deadband_deg must be in [0, 180)")
    recovery = locomotion["low_recovery"]
    if not 0.0 <= recovery["flat_episode_fraction"] <= 1.0:
        raise ValueError("low_recovery.flat_episode_fraction must be in [0, 1]")
    unpowered = recovery["unpowered_seconds"]
    deadline = recovery["deadline_seconds"]
    if recovery["flat_episode_fraction"]:
        if not locomotion["enabled"]:
            raise ValueError("Low recovery requires enabled locomotion")
        if not 0.0 < unpowered < deadline <= task["episode_seconds"]:
            raise ValueError("Low recovery needs 0 < unpowered_seconds < deadline_seconds <= episode_seconds")
    if task["scut_rough_reward"]["braking"]["enabled"] and not locomotion["enabled"]:
        raise ValueError("The braking reward requires enabled locomotion")
    tracking = task["scut_rough_reward"]["tracking"]
    if (tracking["flat_roll_error_weight"] or tracking["flat_roll_rate_weight"]) and not locomotion["enabled"]:
        raise ValueError("The flat-specific roll reward requires enabled locomotion")
    for name in ("leg_torque_limit", "wheel_torque_limit"):
        limit = locomotion[name]
        if not isfinite(limit) or limit <= 0:
            raise ValueError(f"task.locomotion.{name} must be finite and positive")
    wheel_delay = locomotion["wheel_target_delay_steps"]
    if type(wheel_delay) is not int or not 0 <= wheel_delay < 10:
        raise ValueError("wheel_target_delay_steps must be an integer from 0 to 9")
    for name in ("height_range", "command_seconds", "slope_degrees", "obstacle_height"):
        _validate_range(locomotion, name)
    low, high = locomotion["height_range"]
    if not HEIGHT_LIMITS[0] <= low <= locomotion["height_nominal"] <= high <= HEIGHT_LIMITS[1]:
        raise ValueError("height_range and height_nominal must fit the 0.17–0.33 m chassis range")
    for subrange_name in ("obstacle_height_range", "hard_obstacle_height_range"):
        subrange = locomotion[subrange_name]
        if subrange:
            sub_low, sub_high = _validate_range(locomotion, subrange_name)
            if not low <= sub_low <= sub_high <= high:
                raise ValueError(f"{subrange_name} must fit within height_range [{low:g}, {high:g}]")
    schedule = locomotion["height_lowering_schedule"]
    if schedule:
        if not locomotion["enabled"] or any(
            not isinstance(point, list) or len(point) != 2 or type(point[0]) is not int or point[0] < 0
            or type(point[1]) not in (int, float) or not low <= point[1] <= high for point in schedule
        ):
            raise ValueError("height_lowering_schedule requires locomotion and [iteration, height] pairs in height_range")
        if schedule[0][0] != 0 or any(
            a[0] >= b[0] or a[1] < b[1] for a, b in zip(schedule, schedule[1:])
        ):
            raise ValueError("height_lowering_schedule must start at iteration 0 and lower height over increasing iterations")
    if not 0 <= locomotion["height_max_probability"] < 0.40:
        raise ValueError("height_max_probability must be in [0, 0.40) to leave room for continuous sampling")
    if not 0 < locomotion["height_rate"] <= MAX_HEIGHT_RATE or locomotion["command_seconds"][0] <= 0:
        raise ValueError("Use a positive command period and a height_rate within 0–1.2 m/s")
    if not isfinite(locomotion["height_acceleration"]) or locomotion["height_acceleration"] < 0:
        raise ValueError("height_acceleration must be finite and nonnegative")
    if not 0 < locomotion["manual_forward_acceleration"] <= MAX_FORWARD_ACCELERATION:
        raise ValueError(f"manual_forward_acceleration must be in (0, {MAX_FORWARD_ACCELERATION:g}] m/s²")
    if locomotion["braking_maneuvers"] and not (locomotion["enabled"] and locomotion["speed_maneuvers"]):
        raise ValueError("braking_maneuvers requires enabled locomotion and speed_maneuvers")
    if locomotion["launch_maneuvers"] and not (locomotion["enabled"] and locomotion["speed_maneuvers"]):
        raise ValueError("launch_maneuvers requires enabled locomotion and speed_maneuvers")
    if locomotion["height_maneuvers"] and not (locomotion["enabled"] and locomotion["speed_maneuvers"]):
        raise ValueError("height_maneuvers requires enabled locomotion and speed_maneuvers")
    height_sequence = locomotion["height_maneuver_sequence"]
    if height_sequence and (
        not locomotion["height_maneuvers"] or len(height_sequence) < 2
        or any(type(height) not in (int, float) or not low <= height <= high for height in height_sequence)
        or any(a == b for a, b in zip(height_sequence, height_sequence[1:] + height_sequence[:1]))
    ):
        raise ValueError("height_maneuver_sequence requires height maneuvers and distinct consecutive heights in height_range")
    if not 0 <= locomotion["stationary_episode_fraction"] <= 1:
        raise ValueError("stationary_episode_fraction must be in [0, 1]")
    for name in ("launch_press_seconds", "launch_stop_seconds"):
        durations = locomotion[name]
        if durations and (not locomotion["launch_maneuvers"] or any(
            type(v) not in (int, float) or not 0 < v < task["episode_seconds"] for v in durations
        )):
            raise ValueError(f"{name} requires launch_maneuvers and positive durations shorter than an episode")
    for name, maximum in (("command_speed_levels", MAX_FORWARD_SPEED),
                          ("transition_speed_levels", MAX_FORWARD_SPEED),
                          ("scut_ground_bars_speed_levels", MAX_FORWARD_SPEED),
                          ("command_acceleration_levels", MAX_FORWARD_ACCELERATION)):
        if any(type(value) not in (int, float) or not 0 < value <= maximum for value in locomotion[name]):
            raise ValueError(f"{name} must contain positive levels up to {maximum:g}")
    maneuver_count = (
        4 + locomotion["braking_maneuvers"] + locomotion["launch_maneuvers"] + locomotion["height_maneuvers"]
        if locomotion["speed_maneuvers"] else 0
    )
    for name, length in (("speed_maneuver_weights", maneuver_count),
                         ("command_acceleration_weights", len(locomotion["command_acceleration_levels"])),
                         ("scut_ground_bars_variant_weights", 3),
                         ("scut_ground_rough_slope_variant_weights", 3)):
        weights = locomotion[name]
        if weights and (len(weights) != length or any(
            type(value) not in (int, float) or not isfinite(value) or value < 0 for value in weights
        ) or not any(weights)):
            raise ValueError(f"{name} requires {length} nonnegative weights with a positive sum")
    terrain = locomotion["terrain"]
    if locomotion["transition_stop_feedback"] and terrain != "transitions":
        raise ValueError("transition_stop_feedback requires transitions terrain")
    transition_speeds = locomotion["transition_speed_levels"]
    if transition_speeds and terrain != "transitions":
        raise ValueError("transition_speed_levels requires transitions terrain")
    if terrain not in ("plane", "mixed", "scut_v1", "scut_ground", "transitions"):
        raise ValueError("terrain must be plane, mixed, scut_v1, scut_ground or transitions")
    has_terrain = terrain != "plane"
    if locomotion["terrain_traversal"] and not (locomotion["enabled"] and has_terrain):
        raise ValueError("terrain_traversal requires enabled terrain locomotion")
    terrain_limit = locomotion["terrain_forward_limit"]
    if not 0 <= terrain_limit <= MAX_FORWARD_SPEED or (terrain_limit and not locomotion["terrain_traversal"]):
        raise ValueError("terrain_forward_limit requires terrain_traversal and a speed within the chassis range")
    if terrain in ("scut_v1", "scut_ground", "transitions") and not locomotion["terrain_traversal"]:
        raise ValueError(f"{terrain} requires terrain_traversal")
    if terrain == "transitions":
        speeds = locomotion["command_speed_levels"]
        low, high = task["forward_command"]
        if (not speeds or speeds != sorted(set(speeds)) or max(speeds + transition_speeds) > min(-low, high)
                or terrain_limit):
            raise ValueError("transitions requires increasing speed levels within both command directions and no terrain cap")
    scale = locomotion["scut_ground_scale"]
    if not 0.50 <= scale <= 1.0 or (scale != 1.0 and terrain != "scut_ground"):
        raise ValueError("scut_ground_scale must be in [0.50, 1.0] and is only adjustable on scut_ground")
    bars_fraction = locomotion["scut_ground_bars_fraction"]
    bars_reverse = locomotion["scut_ground_bars_reverse_probability"]
    compound_fraction = locomotion["scut_ground_compound_fraction"]
    single_side_fraction = locomotion["scut_ground_single_side_fraction"]
    bump_fraction = locomotion["scut_ground_speed_bump_fraction"]
    low_rough_slope = locomotion["scut_ground_rough_slope_low_height_probability"]
    obstacle_fractions = (bars_fraction, compound_fraction, single_side_fraction, bump_fraction)
    if (any(not 0.0 <= fraction <= 0.60 for fraction in obstacle_fractions)
            or sum(obstacle_fractions) > 0.60):
        raise ValueError("scut_ground obstacle sampling fractions must sum to at most 0.60")
    bump_low, bump_high = _validate_range(locomotion, "scut_ground_speed_bump_height")
    if not 0 < bump_low <= bump_high <= 0.15 or not 0.20 <= locomotion["scut_ground_speed_bump_length"] <= 1.0:
        raise ValueError("Speed bump height must be in (0, 0.15] m and length in [0.20, 1.0] m")
    if not (0.0 <= bars_reverse <= 1.0 and 0.0 <= low_rough_slope <= 1.0):
        raise ValueError("scut_ground reverse and low-height probabilities must be in [0, 1]")
    rough_slope_focus = low_rough_slope or locomotion["scut_ground_rough_slope_variant_weights"]
    if rough_slope_focus and not compound_fraction:
        raise ValueError("Rough-slope sampling requires a positive scut_ground_compound_fraction")
    has_scut_sampling = (bars_fraction != 0.10 or bars_reverse != 0.50 or compound_fraction or single_side_fraction
                         or bump_fraction
                         or rough_slope_focus or locomotion["scut_ground_bars_variant_weights"]
                         or locomotion["scut_ground_bars_speed_levels"])
    if has_scut_sampling and terrain != "scut_ground":
        raise ValueError("The scut_ground sampling settings require scut_ground terrain")
    drops = locomotion["cliff_drop_heights"]
    if drops and (terrain != "scut_v1" or len(drops) != 3
                  or any(type(drop) not in (int, float) or not 0.04 <= drop <= 0.40 for drop in drops)
                  or not drops[0] < drops[1] < drops[2]
                  or locomotion["height_range"][1] < 0.23):
        raise ValueError("cliff_drop_heights requires scut_v1, three increasing drops in [0.04, 0.40] m, and at least 0.23 m leg height")
    approach_speed = locomotion["cliff_approach_speed"]
    if not 0 <= approach_speed <= MAX_FORWARD_SPEED or (approach_speed and not drops):
        raise ValueError("cliff_approach_speed requires forward drops and a speed within the chassis range")
    if (not 0 <= locomotion["slope_degrees"][0] <= locomotion["slope_degrees"][1] <= 15
            or not 0 <= locomotion["roughness"] <= 0.03
            or not 0 < locomotion["obstacle_height"][0] <= locomotion["obstacle_height"][1] <= 0.04
            or locomotion["terrain_seed"] < 0):
        raise ValueError("Invalid conventional terrain dimensions or seed")
    if locomotion["enabled"]:
        for name, limit in (("forward_command", MAX_FORWARD_SPEED), ("yaw_command", YAW_REFERENCE)):
            low, high = task[name]
            if not -limit <= low <= 0 <= high <= limit:
                raise ValueError(f"{name} must include zero and fit ±{limit:g}")

    if locomotion["direct_velocity_commands"] and not locomotion["enabled"]:
        raise ValueError("direct_velocity_commands requires enabled locomotion")
    direct_prob = locomotion["direct_velocity_probability"]
    if not isfinite(direct_prob) or not 0.0 <= direct_prob <= 1.0:
        raise ValueError("direct_velocity_probability must be in [0, 1]")

    terrain_cmds = locomotion["terrain_commands"]
    if terrain_cmds["enabled"]:
        if not (locomotion["enabled"] and locomotion["terrain_traversal"]):
            raise ValueError("terrain_commands requires enabled terrain_traversal")
        for rname in ("fast_forward_range", "hard_forward_range"):
            rlow, rhigh = _validate_range(terrain_cmds, rname)
            if not 0.0 < rlow <= rhigh <= MAX_FORWARD_SPEED:
                raise ValueError(f"terrain_commands.{rname} must be in (0, {MAX_FORWARD_SPEED:g}]")
        for yname in ("fast_yaw_max", "hard_yaw_max"):
            ymax = terrain_cmds[yname]
            if not isfinite(ymax) or not 0.0 <= ymax <= YAW_REFERENCE:
                raise ValueError(f"terrain_commands.{yname} must be in [0, {YAW_REFERENCE:g}]")
        for pname in ("zero_yaw_probability", "stop_probability", "spin_probability"):
            pval = terrain_cmds[pname]
            if not isfinite(pval) or not 0.0 <= pval <= 1.0:
                raise ValueError(f"terrain_commands.{pname} must be in [0, 1]")
        if terrain_cmds["stop_probability"] + terrain_cmds["spin_probability"] > 0.90:
            raise ValueError("terrain_commands stop_probability + spin_probability must leave room for traversal")
        spin_low, spin_high = _validate_range(terrain_cmds, "spin_yaw_range")
        if not 0.0 <= spin_low <= spin_high <= YAW_REFERENCE:
            raise ValueError(f"terrain_commands.spin_yaw_range must fit [0, {YAW_REFERENCE:g}]")
        if _validate_range(terrain_cmds, "resampling_time_range")[0] <= 0.0:
            raise ValueError("terrain_commands.resampling_time_range must be positive")
        if not isfinite(terrain_cmds["engage_x"]) or terrain_cmds["engage_x"] <= 0.0:
            raise ValueError("terrain_commands.engage_x must be positive")

    curriculum = locomotion["terrain_curriculum"]
    if curriculum["enabled"]:
        if not (locomotion["enabled"] and locomotion["terrain_traversal"]):
            raise ValueError("terrain_curriculum requires enabled terrain_traversal")
        if (not isfinite(curriculum["promote_distance"]) or not isfinite(curriculum["demote_distance"])
                or not 0.0 < curriculum["demote_distance"] <= curriculum["promote_distance"]):
            raise ValueError("terrain_curriculum requires 0 < demote_distance <= promote_distance")
        mweights = curriculum["max_level_weights"]
        if len(mweights) != 3 or any(type(w) not in (int, float) or not isfinite(w) or w < 0 for w in mweights) or not any(mweights):
            raise ValueError("terrain_curriculum.max_level_weights requires 3 nonnegative weights with positive sum")

    events = locomotion["events"]
    if events["enabled"]:
        if not locomotion["enabled"]:
            raise ValueError("events requires enabled locomotion")
        if _validate_range(events, "push_interval_seconds")[0] <= 0.0:
            raise ValueError("events.push_interval_seconds must be positive")
        if _validate_range(events, "wrench_interval_seconds")[0] <= 0.0:
            raise ValueError("events.wrench_interval_seconds must be positive")
        for nonneg in ("push_vel_xy", "push_ang_vel_xy", "reset_roll_pitch_rad", "reset_yaw_rad",
                       "reset_joint_pos_rad", "reset_lin_vel", "reset_ang_vel", "air_drop_speed_max"):
            if not isfinite(events[nonneg]) or events[nonneg] < 0.0:
                raise ValueError(f"events.{nonneg} must be finite and nonnegative")
        for vec_name in ("external_force_xyz", "external_torque_xyz"):
            vec = events[vec_name]
            if len(vec) != 3 or any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in vec):
                raise ValueError(f"events.{vec_name} must contain 3 nonnegative finite values")
        for prob_name in ("wrench_zero_probability", "air_drop_flat_fraction", "air_drop_brake_probability"):
            if not isfinite(events[prob_name]) or not 0.0 <= events[prob_name] <= 1.0:
                raise ValueError(f"events.{prob_name} must be in [0, 1]")
        drop_low, drop_high = _validate_range(events, "air_drop_height_range")
        if drop_low < 0.0 or drop_high > 0.45:
            raise ValueError("events.air_drop_height_range must be within [0, 0.45] m")
    special = locomotion["special_mode_commands"]
    if special["enabled"]:
        if not locomotion["enabled"]:
            raise ValueError("special_mode_commands requires enabled locomotion")
        if locomotion["speed_maneuvers"]:
            raise ValueError("special_mode_commands cannot be combined with speed_maneuvers")
        for prob_name in ("rel_standing_envs", "rel_zero_yaw_envs"):
            prob = special[prob_name]
            if not isfinite(prob) or not 0.0 <= prob <= 1.0:
                raise ValueError(f"task.locomotion.special_mode_commands.{prob_name} must be in [0, 1]")
        resample_low, resample_high = _validate_range(special, "resampling_time_range")
        if resample_low <= 0.0:
            raise ValueError("task.locomotion.special_mode_commands.resampling_time_range must be positive")
        for nonneg_name in ("min_episode_time", "stable_gravity_xy_max", "stable_lin_vel_max", "stable_ang_vel_max"):
            val = special[nonneg_name]
            if not isfinite(val) or val < 0.0:
                raise ValueError(f"task.locomotion.special_mode_commands.{nonneg_name} must be finite and nonnegative")
        if special["normal_forward_range"]:
            normalize_range_spec(
                special["normal_forward_range"],
                name="task.locomotion.special_mode_commands.normal_forward_range",
                limit=MAX_FORWARD_SPEED,
            )
        if special["normal_yaw_range"]:
            normalize_range_spec(
                special["normal_yaw_range"],
                name="task.locomotion.special_mode_commands.normal_yaw_range",
                limit=YAW_REFERENCE,
            )
        allowed_mode_keys = {"name", "rel_envs", "iteration_start", "iteration_end", "lin_vel_x", "ang_vel_z"}
        for idx, mode in enumerate(special["modes"]):
            prefix = f"task.locomotion.special_mode_commands.modes[{idx}]"
            if not isinstance(mode, dict):
                raise ValueError(f"{prefix} must be a table")
            unknown = set(mode) - allowed_mode_keys
            if unknown:
                raise ValueError(f"Unknown keys in {prefix}: {sorted(unknown)}")
            for required in ("rel_envs", "lin_vel_x", "ang_vel_z"):
                if required not in mode:
                    raise ValueError(f"{prefix} requires {required}")
            rel = mode["rel_envs"]
            if not isinstance(rel, Real) or isinstance(rel, bool) or not isfinite(rel) or not 0.0 <= rel <= 1.0:
                raise ValueError(f"{prefix}.rel_envs must be in [0, 1]")
            start = mode.get("iteration_start", 0)
            end = mode.get("iteration_end", -1)
            if type(start) is not int or start < 0:
                raise ValueError(f"{prefix}.iteration_start must be a nonnegative integer")
            if type(end) is not int or (end >= 0 and end <= start):
                raise ValueError(f"{prefix}.iteration_end must be -1 or an integer greater than iteration_start")
            normalize_range_spec(mode["lin_vel_x"], name=f"{prefix}.lin_vel_x", limit=MAX_FORWARD_SPEED)
            normalize_range_spec(mode["ang_vel_z"], name=f"{prefix}.ang_vel_z", limit=YAW_REFERENCE)

    randomization = task["domain_randomization"]
    if randomization["enabled"]:
        if not locomotion["enabled"] or not 0.01 <= task["friction"] <= 5.0:
            raise ValueError("Domain randomization requires locomotion and friction in [0.01, 5.0]")
        if not 0 <= randomization["nominal_probability"] <= 1:
            raise ValueError("nominal_probability must be in [0, 1]")
        friction = randomization["contact_friction"]
        if not friction or any(type(mu) not in (int, float) or not 0.01 <= mu <= 5.0 for mu in friction):
            raise ValueError("contact_friction must contain values in [0.01, 5.0]")
        if _validate_range(randomization, "body_mass_scale")[0] <= 0:
            raise ValueError("body_mass_scale must be positive")
        for scale_name in ("spring_force_scale", "leg_kp_scale", "leg_kv_scale", "wheel_kv_scale"):
            if _validate_range(randomization, scale_name)[0] <= 0:
                raise ValueError(f"{scale_name} must be positive")
        offsets = randomization["load_offsets"]
        if not offsets or any(not isinstance(offset, list) or len(offset) != 3 or
                              any(type(v) not in (int, float) or not isfinite(v) for v in offset)
                              for offset in offsets):
            raise ValueError("load_offsets must contain finite XYZ positions")
        for group in ("hips", "passive", "springs", "wheels"):
            for key in ("frictionloss_delta", "damping_delta"):
                _validate_range(randomization[group], key)

    ppo = settings["ppo"]
    for name in ("num_steps_per_env", "save_interval"):
        if ppo[name] <= 0:
            raise ValueError(f"ppo.{name} must be positive")
    max_action_std = ppo["max_action_std"]
    if not isfinite(max_action_std) or (max_action_std != 0.0 and max_action_std < 1e-6):
        raise ValueError("ppo.max_action_std must be zero or a finite value of at least 1e-6")
    init_min_std = ppo["init_checkpoint_min_std"]
    if not isfinite(init_min_std) or init_min_std < 0.0:
        raise ValueError("ppo.init_checkpoint_min_std must be finite and nonnegative")
    for name in ("num_learning_epochs", "num_mini_batches"):
        if ppo["algorithm"][name] <= 0:
            raise ValueError(f"ppo.algorithm.{name} must be positive")
    for name in ("actor_hidden_dims", "critic_hidden_dims"):
        dims = ppo["policy"][name]
        if not dims or any(type(dim) is not int or dim <= 0 for dim in dims):
            raise ValueError(f"ppo.policy.{name} must contain positive integers")


def _merge_config(overrides):
    with DEFAULT_CONFIG.open("rb") as file:
        settings = tomllib.load(file)
    if overrides is not None:
        _apply_overrides(settings, overrides)
    _validate_settings(settings)
    return settings


def load_snapshot(path):
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    run = snapshot.pop("run", {})
    return _merge_config(snapshot), run


def load_config(path=None):
    if path is None:
        return _merge_config(None)
    if path.suffix.lower() == ".json":
        return load_snapshot(path)[0]
    with path.open("rb") as file:
        return _merge_config(tomllib.load(file))


def runner_config(ppo, *, actor_true_forward_speed=False, actor_history_steps=1):
    # RSL-RL removes class names and alters the configuration it receives.
    config = deepcopy(ppo)
    config.pop("max_action_std", None)
    config.pop("init_checkpoint_min_std", None)
    config["obs_groups"] = {"policy": ["policy"], "critic": ["critic"]}
    if actor_true_forward_speed:
        config["obs_groups"]["policy"].append("body_forward_speed")
    if actor_history_steps > 1:
        config["obs_groups"]["policy"].append("policy_history")
    config["policy"]["class_name"] = "ActorCritic"
    config["algorithm"]["class_name"] = "PPO"
    return config


def save_run_config(output, settings, args, robot_class, num_envs):
    model = args.model.resolve()
    snapshot = {
        **settings,
        "run": {
            "task": robot_class.task_name,
            "robot": args.robot,
            "robot_label": robot_class.label,
            "model": model.relative_to(TRAINING_DIR).as_posix() if model.is_relative_to(TRAINING_DIR) else str(model),
            "model_sha256": sha256(model.read_bytes()).hexdigest(),
            "seed": args.seed,
            "backend": args.backend,
            "num_envs": num_envs,
            "iterations": args.iterations,
            "physics_dt": robot_class.physics_dt,
            "decimation": robot_class.decimation,
            "control_dt": robot_class.physics_dt * robot_class.decimation,
            "init_at_random_ep_len": True,
        },
    }
    if args.resume is not None:
        snapshot["run"]["resume_checkpoint"] = str(args.resume.resolve())
    if args.init_checkpoint is not None:
        snapshot["run"]["init_checkpoint"] = str(args.init_checkpoint.resolve())
    contents = json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {output}. Choose a new --output path.")
    with (output / SNAPSHOT_NAME).open("x", encoding="utf-8") as file:
        file.write(contents)
