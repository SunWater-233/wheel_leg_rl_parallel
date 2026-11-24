"""Deterministic height-field lanes with clear spawn pads and bounded episodes."""

import genesis as gs
import numpy as np


SURFACES = ("flat", "slope", "rough", "bars", "stairs")
SURFACE_WEIGHTS = (0.40, 0.25, 0.20, 0.075, 0.075)
SCUT_V1_SURFACES = SURFACES + ("inv_slope", "inv_stairs_low", "inv_stairs_high",
                               "stairs_slow", "stairs_fast", "cliff")
SCUT_GROUND_SURFACES = SURFACES + ("inv_slope", "inv_stairs_low", "inv_stairs_high")
SCUT_GROUND_COMPOUND_SURFACES = ("rough_slope", "staggered_bars")
SCUT_GROUND_SINGLE_SIDE_SURFACES = ("single_side_bars_left", "single_side_bars_right")
SCUT_GROUND_SPEED_BUMP = "speed_bump"
SURFACE_STYLES = {
    "flat": ("平地", (116, 141, 160)),
    "slope": ("坡道", (112, 157, 123)),
    "rough": ("粗糙路", (161, 143, 113)),
    "bars": ("横条障碍", (182, 151, 91)),
    "stairs": ("细台阶", (145, 130, 174)),
    "inv_slope": ("凹坡", (95, 155, 158)),
    "inv_stairs_low": ("低沉台阶", (121, 139, 182)),
    "inv_stairs_high": ("高沉台阶", (145, 123, 169)),
    "stairs_slow": ("低速高台阶", (177, 132, 110)),
    "stairs_fast": ("高速高台阶", (169, 115, 117)),
    "cliff": ("断崖", (122, 144, 153)),
    "rough_slope": ("粗糙坡道", (119, 149, 107)),
    "staggered_bars": ("错位凸条", (185, 130, 90)),
    "single_side_bars_left": ("单侧凸条（左轮先过）", (174, 116, 86)),
    "single_side_bars_right": ("单侧凸条（右轮先过）", (177, 100, 83)),
    SCUT_GROUND_SPEED_BUMP: ("曲面减速带", (191, 144, 79)),
    "transitions": ("随机地形过渡", (132, 145, 121)),
}


def raised_cosine_profile(x, length):
    """Unit-height bump with zero height and slope at both ends."""
    return 0.5 * (1 + np.cos(np.pi * np.clip(2 * x / length, -1, 1)))


class TerrainLanes:
    surfaces = SURFACES
    surface_weights = SURFACE_WEIGHTS
    spacing = 0.05
    vertical_scale = 0.001
    half_length = 5.0
    lane_width = 2.2
    variants = 2
    travel_limit = 4.6
    lateral_limit = 0.85

    @classmethod
    def surface_names(cls, settings):
        return cls.surfaces

    def __init__(self, settings):
        rng = np.random.default_rng(settings["terrain_seed"])
        x = np.linspace(-self.half_length, self.half_length, round(2 * self.half_length / self.spacing) + 1)
        width = round(self.lane_width / self.spacing)
        self.centers = (0.5 + np.arange(len(SURFACES) * self.variants)) * self.lane_width
        self.field = np.zeros((len(x), len(self.centers) * width + 1), dtype=np.int16)
        distance = np.maximum(np.abs(x) - 0.65, 0.0)
        for index in range(len(self.centers)):
            surface = SURFACES[index // self.variants]
            direction = -1 if index % self.variants else 1
            slope = np.tan(np.deg2rad(rng.uniform(*settings["slope_degrees"])))
            obstacle = rng.uniform(*settings["obstacle_height"])
            profile = x * 0.0
            if surface == "slope":
                profile = direction * np.sign(x) * distance.clip(max=2.5) * slope
            elif surface == "bars":
                for edge in (0.85, 1.55, 2.25, 2.95):
                    profile[(np.abs(x) >= edge) & (np.abs(x) < edge + 0.12)] = obstacle
            elif surface == "stairs":
                steps = np.floor(distance / 0.45).clip(max=4)
                profile = direction * np.sign(x) * steps * obstacle
            section = slice(index * width, (index + 1) * width)
            self.field[:, section] = np.rint(profile[:, None] / self.vertical_scale).astype(np.int16)
            if surface == "rough":
                # Nearby cells share height samples. Keep initial wheel contacts on a level pad.
                coarse = rng.uniform(-settings["roughness"], settings["roughness"], (len(x) // 5 + 1, width // 4 + 1))
                rough = np.repeat(np.repeat(coarse, 5, axis=0), 4, axis=1)[:len(x), :width]
                rough *= (distance > 0)[:, None]
                self.field[:, section] = np.rint(rough / self.vertical_scale).astype(np.int16)
        self.field[:, -1] = self.field[:, -2]

    def lane_columns(self, lane):
        if lane is None:
            return slice(0, self.field.shape[1])
        width = round(self.lane_width / self.spacing)
        # Include both wheel footprints and terrain scan points just outside the lane.
        padding = round(1.0 / self.spacing)
        return slice(max(0, lane * width - padding),
                     min(self.field.shape[1], (lane + 1) * width + padding + 1))

    @property
    def width(self):
        return (self.field.shape[1] - 1) * self.spacing

    def morph(self, viewer, lane=None):
        section = self.lane_columns(lane)
        return gs.morphs.Terrain(pos=(-self.half_length, section.start * self.spacing, 0.0),
                                 height_field=self.field[:, section],
                                 horizontal_scale=self.spacing, vertical_scale=self.vertical_scale,
                                 visualization=viewer)

    def describe(self, surface_index, variant):
        surface = self.surfaces[surface_index]
        name = SURFACE_STYLES[surface][0]
        return name if surface == "flat" else f"{name} · 档位 {variant + 1}/{self.variants}"

    def visual_surface(self, lane=None):
        """Paint each lane and darken terrain edges using the same height field as collision."""
        section = self.lane_columns(lane)
        heights = self.field[:, section].astype(np.float32) * self.vertical_scale
        across, along = np.gradient(heights)
        edge = np.clip(np.maximum(np.abs(across), np.abs(along)) / 0.035, 0.0, 1.0)
        bands = np.floor(np.abs(heights) / 0.10) % 2
        shade = 1.0 + 0.28 * np.clip(heights / 0.20, -1.0, 1.0) - 0.36 * edge - 0.08 * bands
        lane_width = round(self.lane_width / self.spacing)
        columns = np.arange(section.start, section.stop)
        lane = (columns // lane_width).clip(max=len(self.centers) - 1)
        colors = np.array([SURFACE_STYLES[surface][1] for surface in self.surfaces])
        rgb = colors[lane // self.variants] * shade[..., None]
        # UV.x follows the terrain's x axis, and UV.y starts at the bottom of a texture.
        image = np.ascontiguousarray(rgb.transpose(1, 0, 2)[::-1].clip(0, 255), dtype=np.uint8)
        return gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_array=image), roughness=0.95)


class ScutV1Lanes(TerrainLanes):
    """SCUT Rough-v1 dimensions and terrain families adapted to Genesis height fields."""

    surfaces = SCUT_V1_SURFACES
    # Keep 40% plane. Non-flat weights follow SCUT's relative proportions.
    surface_weights = (0.40, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05)
    difficulties = (0.0, 0.5, 1.0)
    variant_weights = (0.55, 0.30, 0.15)
    variants = len(difficulties)
    spacing = 0.10
    vertical_scale = 0.005
    lane_width = 9.0
    travel_limit = 4.8
    lateral_limit = 4.20
    approach_x = 4.2
    deck_half_length = 2.5
    deck_half_width = 3.5
    deck_spawn_x = 1.4
    stairs = {
        # Height range, tread depth, central platform width, flat border, ascent direction
        "stairs": ((0.015, 0.032), 0.10, 1.0, 1.0, 1.0),
        "inv_stairs_low": ((0.005, 0.015), 0.10, 2.0, 1.0, -1.0),
        "inv_stairs_high": ((0.015, 0.032), 0.10, 2.0, 1.0, -1.0),
        "stairs_slow": ((0.15, 0.35), 1.50, 3.0, 0.0, 1.0),
        "stairs_fast": ((0.15, 0.40), 1.50, 3.0, 0.0, 1.0),
    }

    def __init__(self, settings):
        rng = np.random.default_rng(settings["terrain_seed"])
        half_cells = round(self.half_length / self.spacing)
        x = np.arange(-half_cells, half_cells + 1) * self.spacing
        half_lane_cells = round(self.lane_width / (2 * self.spacing))
        y = np.arange(-half_lane_cells, half_lane_cells) * self.spacing
        width = len(y)
        radius = np.maximum(np.abs(x)[:, None], np.abs(y)[None, :])
        self.centers = (0.5 + np.arange(len(self.surfaces) * self.variants)) * self.lane_width
        self.field = np.zeros((len(x), len(self.centers) * width + 1), dtype=np.int16)
        self.spawn_x = np.full(len(self.centers), self.approach_x)
        self.spawn_z = np.zeros(len(self.centers))
        drops = settings["cliff_drop_heights"]
        self.drop_decks = []
        if drops:
            # Keep flat rehearsals and give the new forward drop more samples.
            self.surface_weights = (0.40, *(0.45 * weight / 0.55 for weight in self.surface_weights[1:-1]), 0.15)
        for surface_index, surface in enumerate(self.surfaces):
            for variant, difficulty in enumerate(self.difficulties):
                lane = surface_index * self.variants + variant
                if surface == "cliff" and drops:
                    height = drops[variant]
                    self.spawn_x[lane] = self.deck_spawn_x
                    self.spawn_z[lane] = height
                    self.drop_decks.append((self.centers[lane], height))
                    continue
                profile = self._tile(surface, difficulty, x, y, radius, rng)
                section = slice(lane * width, (lane + 1) * width)
                self.field[:, section] = np.rint(profile / self.vertical_scale).astype(np.int16)
                if surface in ("flat", "cliff"):
                    self.spawn_x[lane] = 0.0
                    self.spawn_z[lane] = self.field[len(x) // 2, lane * width + width // 2] * self.vertical_scale
        self.field[:, -1] = self.field[:, -2]

    def deck_boxes(self):
        """Use boxes for vertical faces; height fields interpolate between grid samples."""
        for y, height in self.drop_decks:
            yield (0.0, y, height / 2), (self.deck_half_length, self.deck_half_width, height / 2)

    def describe(self, surface_index, variant):
        if self.surfaces[surface_index] == "cliff" and self.drop_decks:
            return f"断崖 · 落差 {100 * self.drop_decks[variant][1]:g} cm"
        return super().describe(surface_index, variant)

    def _tile(self, surface, difficulty, x, y, radius, rng):
        if surface in self.stairs:
            heights, tread, platform, border, sign = self.stairs[surface]
            riser = self.vertical_scale * np.rint(np.interp(difficulty, (0.0, 1.0), heights) / self.vertical_scale)
            outer = ScutV1Lanes.lane_width / 2 - border
            depth = np.clip(outer - radius, 0.0, outer - platform / 2)
            return sign * riser * np.floor(depth / tread + 1e-8)
        if surface == "slope":
            return np.interp(difficulty, (0.0, 1.0), (0.15, 0.35)) * np.clip(2.5 - radius, 0.0, 1.5)
        if surface == "inv_slope":
            return -np.interp(difficulty, (0.0, 1.0), (0.05, 0.15)) * np.maximum(3.5 - radius, 0.0)
        if surface == "bars":
            count = 4 if difficulty >= 0.5 else 2
            width = np.interp(difficulty, (0.0, 1.0), (0.20, 0.05))
            height = np.interp(difficulty, (0.0, 1.0), (0.02, 0.06))
            x_bars = np.linspace(-2.5, 2.5, count)
            y_bars = np.linspace(-2.5, 2.5, count) + rng.uniform(-0.2, 0.2)
            across = np.min(np.abs(x[:, None] - x_bars), axis=1)[:, None]
            along = np.min(np.abs(y[:, None] - y_bars), axis=1)[None, :]
            half_cell = max(width, self.spacing) / 2 + 1e-8
            return height * ((across <= half_cell) | (along <= half_cell)) * (radius <= 3.5)
        if surface == "rough":
            cells = round(0.20 / self.spacing)
            coarse = rng.integers(0, 7, (len(x) // cells + 1, len(y) // cells + 1)) * self.vertical_scale
            rough = np.repeat(np.repeat(coarse, cells, axis=0), cells, axis=1)[:len(x), :len(y)]
            return rough * (radius <= 3.5)
        if surface == "cliff":
            offset = np.interp(difficulty, (0.0, 1.0), (0.30, 0.40))
            riser = self.vertical_scale * np.rint(np.interp(difficulty, (0.0, 1.0), (0.025, 0.032)) / self.vertical_scale)
            depth = np.clip(2.5 - radius, 0.0, 1.0)
            return (offset - riser * np.floor(depth / 0.10 + 1e-8)) * (radius <= 2.5)
        return np.zeros_like(radius)


class ScutGroundLanes(ScutV1Lanes):
    """Grounded SCUT terrain with an extended straight for flat starts and stops."""

    surfaces = SCUT_GROUND_SURFACES
    surface_weights = (0.40, 0.10, 0.12, 0.10, 0.08, 0.08, 0.06, 0.06)
    half_length = 12.0
    flat_travel_limit = 11.8

    @classmethod
    def surface_names(cls, settings):
        surfaces = cls.surfaces
        if settings["scut_ground_compound_fraction"]:
            surfaces += SCUT_GROUND_COMPOUND_SURFACES
        if settings["scut_ground_single_side_fraction"]:
            surfaces += SCUT_GROUND_SINGLE_SIDE_SURFACES
        if settings["scut_ground_speed_bump_fraction"]:
            surfaces += (SCUT_GROUND_SPEED_BUMP,)
        return surfaces

    def __init__(self, settings):
        self.scale = settings["scut_ground_scale"]
        self.surfaces = self.surface_names(settings)
        bars = self.surfaces.index("bars")
        fraction = settings["scut_ground_bars_fraction"]
        compound_fraction = settings["scut_ground_compound_fraction"]
        single_side_fraction = settings["scut_ground_single_side_fraction"]
        bump_fraction = settings["scut_ground_speed_bump_fraction"]
        if bump_fraction:
            self.spacing = 0.05
            self.lane_width = 4.0
            self.lateral_limit = 1.8
            self.bump_heights = settings["scut_ground_speed_bump_height"]
            self.bump_length = settings["scut_ground_speed_bump_length"]
        self.reverse_probabilities = tuple(
            settings["scut_ground_bars_reverse_probability"] if index == bars else 0.5
            for index in range(len(self.surfaces))
        )
        variant_overrides = {
            "bars": settings["scut_ground_bars_variant_weights"],
            "rough_slope": settings["scut_ground_rough_slope_variant_weights"],
        }
        self.surface_variant_weights = None
        if any(variant_overrides.values()):
            self.surface_variant_weights = tuple(
                variant_overrides.get(surface) or self.variant_weights for surface in self.surfaces
            )
        if compound_fraction or single_side_fraction or bump_fraction or fraction != self.surface_weights[bars]:
            obstacle_fraction = 1.0 - self.surface_weights[0]
            other_scale = ((obstacle_fraction - fraction - compound_fraction - single_side_fraction - bump_fraction)
                           / (obstacle_fraction - self.surface_weights[bars]))
            self.surface_weights = (self.surface_weights[0], *(
                fraction if index == bars else weight * other_scale
                for index, weight in enumerate(self.surface_weights[1:], start=1)
            ))
            if compound_fraction:
                count = len(SCUT_GROUND_COMPOUND_SURFACES)
                self.surface_weights += (compound_fraction / count,) * count
            if single_side_fraction:
                count = len(SCUT_GROUND_SINGLE_SIDE_SURFACES)
                self.surface_weights += (single_side_fraction / count,) * count
            if bump_fraction:
                self.surface_weights += (bump_fraction,)
        super().__init__(settings)

    def _tile(self, surface, difficulty, x, y, radius, rng):
        if surface == SCUT_GROUND_SPEED_BUMP:
            height = np.interp(difficulty, (0.0, 1.0), self.bump_heights)
            across = np.clip((0.80 - np.abs(y)) / 0.05, 0.0, 1.0)
            return height * raised_cosine_profile(x, self.bump_length)[:, None] * across
        if surface == "rough_slope":
            slope = super()._tile("slope", difficulty, x, y, radius, rng)
            rough = super()._tile("rough", difficulty, x, y, radius, rng)
            taper = np.clip((2.8 - radius) / 0.45, 0.0, 1.0)
            return self.scale * (slope + (0.30 + 0.30 * difficulty) * rough * taper)
        if surface == "staggered_bars":
            centers = np.array((-2.25, -0.75, 0.75, 2.25)) + rng.uniform(-0.10, 0.10, 4)
            offset = (0.12 + 0.13 * difficulty) * np.clip(y / 0.50, -1.0, 1.0)
            # Alternate which wheel reaches each ridge first, with a continuous join across the lane.
            centers = centers + offset[:, None] * np.array((1.0, -1.0, 1.0, -1.0))
            distance = np.min(np.abs(x[:, None, None] - centers), axis=2)
            ridge = np.clip((0.15 - distance) / 0.075, 0.0, 1.0)
            taper = np.clip((3.5 - radius) / 0.25, 0.0, 1.0)
            return self.scale * (0.020 + 0.020 * difficulty) * ridge * taper
        if surface in SCUT_GROUND_SINGLE_SIDE_SURFACES:
            first_side = 1.0 if surface == SCUT_GROUND_SINGLE_SIDE_SURFACES[0] else -1.0
            centers = np.array((-2.35, -1.15, 0.0, 1.15, 2.35))
            sides = first_side * np.array((1.0, -1.0, 1.0, -1.0, 1.0))
            distance = np.abs(x[:, None] - centers)
            ridges = np.clip((0.18 - distance) / 0.08, 0.0, 1.0)
            lateral = np.clip((y[:, None] * sides - 0.02) / 0.09, 0.0, 1.0)
            lateral *= np.clip((0.90 - np.abs(y[:, None])) / 0.18, 0.0, 1.0)
            return self.scale * (0.01875 + 0.01875 * difficulty) * (ridges @ lateral.T)
        profile = super()._tile(surface, difficulty, x, y, radius, rng)
        if surface == "rough":
            # The reference rough tile has one amplitude; expose three levels for validation.
            profile *= 0.60 + 0.40 * difficulty
        return profile * self.scale


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


TERRAIN_LANES = {"mixed": TerrainLanes, "scut_v1": ScutV1Lanes, "scut_ground": ScutGroundLanes,
                 "transitions": TransitionLanes}
