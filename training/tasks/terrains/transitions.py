"""Seeded sequences of distinct surfaces, with explicit traversable joins and drops."""

import numpy as np

from tasks.terrains.lanes import ScutV1Lanes


class TransitionLanes(ScutV1Lanes):
    """Seeded sequences of distinct surfaces, with explicit traversable joins and drops."""

    surfaces = ("flat", "transitions")
    surface_weights = (0.35, 0.65)
    variants = 8
    variant_weights = (1 / variants,) * variants
    spacing = 0.05
    lane_width = 4.0
    lateral_limit = 1.8
    half_length = 12.0
    travel_limit = 11.6
    approach_x = 4.2
    seams = (-2.8, 0.0, 2.8)
    families = ("flat", "slope", "rough", "stairs", "rough_slope")

    def __init__(self, settings):
        rng = np.random.default_rng(settings["terrain_seed"])
        cells = round(self.half_length / self.spacing)
        x = np.arange(-cells, cells + 1) * self.spacing
        width = round(self.lane_width / self.spacing)
        self.centers = (0.5 + np.arange(2 * self.variants)) * self.lane_width
        self.field = np.zeros((x.size, 2 * self.variants * width + 1), dtype=np.int16)
        self.spawn_x = np.zeros_like(self.centers)
        self.spawn_x[self.variants:] = self.approach_x
        self.spawn_z = np.zeros_like(self.centers)
        self.reverse_spawn_z = np.zeros_like(self.centers)
        self.forward_only = np.zeros_like(self.centers, dtype=bool)
        self.drop_decks = []
        self.segment_families = []
        self.seam_steps = []
        for variant in range(self.variants):
            families = tuple(rng.choice(self.families, 4, replace=False))
            joins = rng.choice((0.0, 0.0, 0.025, 0.045, 0.065, 0.085), len(self.seams))
            joins *= rng.choice((-1, 1), len(self.seams))
            # A minority of lanes reproduce the larger downward cross-lane step seen in free drive.
            if variant % 3 == 2:
                joins[rng.integers(len(joins))] = -rng.uniform(0.12, 0.18)
                self.forward_only[self.variants + variant] = True
            self.segment_families.append(families)
            self.seam_steps.append(tuple(joins))
            breaks = (-4.0, *self.seams, 4.0)
            skew = rng.uniform(-0.22, 0.22) * (np.arange(width) * self.spacing - self.lane_width / 2)
            local_x = x[:, None] - skew
            baseline = 0.0
            profiles = []
            for i, (kind, start, end) in enumerate(zip(families, breaks, breaks[1:])):
                if i:
                    baseline = profiles[-1][1] + joins[i - 1]
                length = end - start
                grade = rng.uniform(0.07, 0.21) * rng.choice((-1, 1)) if kind in ("slope", "rough_slope") else 0.0
                rise = rng.uniform(0.010, 0.025) * rng.choice((-1, 1)) if kind == "stairs" else 0.0
                steps = 7
                dx = (local_x - start).clip(0, length)
                risers = np.floor(dx * (steps + 1) / length + 1e-8).clip(0, steps)
                tile = baseline + grade * dx + rise * risers
                if kind in ("rough", "rough_slope"):
                    block = 4
                    noise = rng.uniform(-1, 1, (x.size // block + 1, width // block + 1))
                    noise = np.repeat(np.repeat(noise, block, axis=0), block, axis=1)[:x.size, :width]
                    taper = np.minimum(dx, length - dx).clip(0, 0.22) / 0.22
                    tile += rng.uniform(0.006, 0.020) * noise * taper
                profiles.append((tile, baseline + grade * length + rise * steps))
            grid = profiles[0][0]
            for seam, step, (tile, _) in zip(self.seams, joins, profiles[1:]):
                # Continuous joins blend the adjacent grades; nonzero steps retain a sharp lip.
                blend = np.clip((local_x - seam + 0.18) / 0.36, 0, 1) if not step else (local_x >= seam).astype(float)
                grid = grid * (1 - blend) + tile * blend
            section = slice((self.variants + variant) * width, (self.variants + variant + 1) * width)
            quantized = np.rint(grid / self.vertical_scale).astype(np.int16)
            self.field[:, section] = quantized
            approach = round(self.approach_x / self.spacing)
            self.spawn_z[self.variants + variant] = quantized[cells - approach, width // 2] * self.vertical_scale
            self.reverse_spawn_z[self.variants + variant] = quantized[cells + approach, width // 2] * self.vertical_scale
        self.field[:, -1] = self.field[:, -2]
