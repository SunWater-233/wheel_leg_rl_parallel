"""SCUT-style SpecialModeUniformVelocityCommand for Genesis wheel-leg locomotion."""

import torch

from configs.experiment import normalize_range_spec


def sample_range_spec(range_spec, count, device):
    """Uniformly sample `count` values across one or more disjoint (low, high) intervals."""
    if count <= 0:
        return torch.zeros(0, device=device)
    if len(range_spec) == 1:
        low, high = range_spec[0]
        if low == high:
            return torch.full((count,), low, dtype=torch.float32, device=device)
        return torch.empty(count, dtype=torch.float32, device=device).uniform_(low, high)

    lengths = torch.tensor([max(high - low, 0.0) for low, high in range_spec], dtype=torch.float32, device=device)
    total = float(lengths.sum().item())
    if total <= 0.0:
        probs = torch.full((len(range_spec),), 1.0 / len(range_spec), dtype=torch.float32, device=device)
    else:
        probs = lengths / total

    segments = torch.multinomial(probs, count, replacement=True)
    samples = torch.empty(count, dtype=torch.float32, device=device)
    for idx, (low, high) in enumerate(range_spec):
        mask = segments == idx
        if not mask.any():
            continue
        n = int(mask.sum().item())
        if low == high:
            samples[mask] = low
        else:
            samples[mask] = torch.empty(n, dtype=torch.float32, device=device).uniform_(low, high)
    return samples


class SpecialModeUniformVelocityCommand:
    """SCUT-style uniform velocity command sampler with iteration-gated special modes.

    Priority chain at each resample:
    1. Standing envs (`rel_standing_envs`): forward = 0, yaw = 0.
    2. Active special modes (within non-standing envs that pass readiness gates):
       disjoint probability buckets proportional to each mode's `rel_envs`, with
       shuffled bucket order on every resample. Special-mode envs bypass heading hold.
    3. Normal envs: uniform sampling within `normal_forward_range` and `normal_yaw_range`,
       with `rel_zero_yaw_envs` straight-line envs (yaw = 0) so heading hold still trains.
    """

    def __init__(self, settings, *, num_envs, device, default_forward_range, default_yaw_range):
        self.num_envs = num_envs
        self.device = device
        self.rel_standing_envs = float(settings["rel_standing_envs"])
        self.rel_zero_yaw_envs = float(settings["rel_zero_yaw_envs"])
        self.resampling_time_range = tuple(float(v) for v in settings["resampling_time_range"])
        self.min_episode_time = float(settings["min_episode_time"])
        self.require_stable = bool(settings["require_stable"])
        self.stable_gravity_xy_max = float(settings["stable_gravity_xy_max"])
        self.stable_lin_vel_max = float(settings["stable_lin_vel_max"])
        self.stable_ang_vel_max = float(settings["stable_ang_vel_max"])

        normal_vx = settings["normal_forward_range"] or default_forward_range
        normal_wz = settings["normal_yaw_range"] or default_yaw_range
        self.normal_forward_range = normalize_range_spec(normal_vx, name="normal_forward_range")
        self.normal_yaw_range = normalize_range_spec(normal_wz, name="normal_yaw_range")

        self.modes = []
        for idx, entry in enumerate(settings["modes"]):
            self.modes.append({
                "name": str(entry.get("name", f"mode_{idx}")),
                "rel_envs": float(entry["rel_envs"]),
                "iteration_start": int(entry.get("iteration_start", 0)),
                "iteration_end": int(entry.get("iteration_end", -1)),
                "lin_vel_x": normalize_range_spec(entry["lin_vel_x"], name=f"modes[{idx}].lin_vel_x"),
                "ang_vel_z": normalize_range_spec(entry["ang_vel_z"], name=f"modes[{idx}].ang_vel_z"),
            })

        self.is_standing_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.special_mode_id = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.training_iteration = 0
        self._mode_enabled = [False] * len(self.modes)
        self.set_training_iteration(0, log=False)

    def reset(self, ids):
        if ids.numel() == 0:
            return
        self.is_standing_env[ids] = False
        self.special_mode_id[ids] = -1

    def set_training_iteration(self, iteration, *, log=True):
        self.training_iteration = int(iteration)
        for idx, mode in enumerate(self.modes):
            start = mode["iteration_start"]
            end = mode["iteration_end"]
            enabled = self.training_iteration >= start and (end < 0 or self.training_iteration < end)
            if log and enabled and not self._mode_enabled[idx] and mode["rel_envs"] > 0.0:
                print(
                    f"[SpecialMode] mode={mode['name']} enabled at iteration={self.training_iteration}",
                    flush=True,
                )
            self._mode_enabled[idx] = enabled

    def is_special_active(self):
        return self.special_mode_id >= 0

    def _active_modes(self):
        return [
            (idx, mode["rel_envs"])
            for idx, mode in enumerate(self.modes)
            if mode["rel_envs"] > 0.0 and self._mode_enabled[idx]
        ]

    def _ready_mask(self, ids, *, episode_time, gravity, lin_vel, ang_vel):
        ready = torch.ones(ids.numel(), dtype=torch.bool, device=self.device)
        if self.min_episode_time > 0.0 and episode_time is not None:
            ready &= episode_time[ids] >= self.min_episode_time
        if self.require_stable:
            if self.stable_gravity_xy_max > 0.0 and gravity is not None:
                ready &= torch.linalg.norm(gravity[ids, :2], dim=-1) <= self.stable_gravity_xy_max
            if self.stable_lin_vel_max > 0.0 and lin_vel is not None:
                ready &= lin_vel[ids].abs().amax(dim=-1) <= self.stable_lin_vel_max
            if self.stable_ang_vel_max > 0.0 and ang_vel is not None:
                ready &= ang_vel[ids].abs().amax(dim=-1) <= self.stable_ang_vel_max
        return ready

    def sample(self, ids, *, episode_time=None, gravity=None, lin_vel=None, ang_vel=None):
        """Sample (motion[N, 2], duration[N]) for the given environment indices."""
        count = int(ids.numel())
        if count == 0:
            return (
                torch.zeros((0, 2), dtype=torch.float32, device=self.device),
                torch.zeros((0,), dtype=torch.float32, device=self.device),
            )

        low_t, high_t = self.resampling_time_range
        duration = low_t + torch.rand(count, device=self.device) * (high_t - low_t)
        motion = torch.zeros((count, 2), dtype=torch.float32, device=self.device)

        # 1. Standing envs (highest priority)
        standing = torch.rand(count, device=self.device) < self.rel_standing_envs
        self.is_standing_env[ids] = standing
        self.special_mode_id[ids] = -1

        non_standing_pos = (~standing).nonzero(as_tuple=True)[0]
        if non_standing_pos.numel() == 0:
            return motion, duration

        # 2. Special modes within ready non-standing envs
        active_modes = self._active_modes()
        normal_pos = non_standing_pos
        if active_modes:
            non_standing_ids = ids[non_standing_pos]
            ready = self._ready_mask(
                non_standing_ids,
                episode_time=episode_time,
                gravity=gravity,
                lin_vel=lin_vel,
                ang_vel=ang_vel,
            )
            candidate_pos = non_standing_pos[ready]
            delayed_pos = non_standing_pos[~ready]

            assigned_mask = torch.zeros(candidate_pos.numel(), dtype=torch.bool, device=self.device)
            if candidate_pos.numel() > 0:
                order = torch.randperm(len(active_modes), device=self.device).tolist()
                draws = torch.rand(candidate_pos.numel(), device=self.device)
                cumulative = 0.0
                for order_idx in order:
                    mode_idx, rel = active_modes[order_idx]
                    slot = (draws >= cumulative) & (draws < cumulative + rel)
                    cumulative += rel
                    if not slot.any():
                        continue
                    assigned_mask |= slot
                    chosen_pos = candidate_pos[slot]
                    chosen_ids = ids[chosen_pos]
                    self.special_mode_id[chosen_ids] = mode_idx
                    mode = self.modes[mode_idx]
                    n_chosen = int(chosen_pos.numel())
                    motion[chosen_pos, 0] = sample_range_spec(mode["lin_vel_x"], n_chosen, self.device)
                    motion[chosen_pos, 1] = sample_range_spec(mode["ang_vel_z"], n_chosen, self.device)

            normal_pos = torch.cat((delayed_pos, candidate_pos[~assigned_mask]))

        # 3. Normal uniform sampling
        if normal_pos.numel() > 0:
            n_normal = int(normal_pos.numel())
            motion[normal_pos, 0] = sample_range_spec(self.normal_forward_range, n_normal, self.device)
            yaw_samples = sample_range_spec(self.normal_yaw_range, n_normal, self.device)
            if self.rel_zero_yaw_envs > 0.0:
                zero_yaw = torch.rand(n_normal, device=self.device) < self.rel_zero_yaw_envs
                yaw_samples = yaw_samples.masked_fill(zero_yaw, 0.0)
            motion[normal_pos, 1] = yaw_samples

        return motion, duration
