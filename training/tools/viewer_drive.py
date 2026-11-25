"""Keyboard driving and camera following in the native Genesis viewer."""

import numpy as np

from genesis.vis.keybindings import Key
from genesis.vis.viewer_plugins import EVENT_HANDLED, ViewerPlugin

from tools.command_slope import FIRMWARE_LEG_LENGTH_SLOPE_SPEED, KeyboardCommandSlope


class ViewerFollow(ViewerPlugin):
    def __init__(self):
        super().__init__()
        self.robot_xy = (0.0, 0.0)
        self.followed_xy = (0.0, 0.0)
        self.trackball = None
        self.terrain = None
        self.shown_terrain = None
        self.window_title = None

    def follow_position(self, x, y):
        self.robot_xy = (x, y)

    def show_terrain(self, description):
        self.terrain = description

    def on_draw(self):
        terrain = self.terrain
        flags = self.viewer.viewer_flags
        if terrain != self.shown_terrain:
            if self.window_title is None:
                self.window_title = flags["window_title"]
            caption = f"{self.window_title} | {terrain}"
            flags["window_title"] = caption
            self.viewer.set_caption(f"{caption} (RECORDING)" if flags["record"] else caption)
            self.shown_terrain = terrain
        # Mouse gestures also run on the viewer thread. Shift both the current
        # view and the mouse-down reference frame so an active drag stays relative.
        trackball = self.viewer._trackball
        if trackball is not self.trackball:
            self.trackball = trackball
            self.followed_xy = (0.0, 0.0)

        x, y = self.robot_xy
        dx, dy = x - self.followed_xy[0], y - self.followed_xy[1]
        if dx == 0.0 and dy == 0.0:
            return
        offset = np.array((dx, dy, 0.0))
        drag_pose = trackball._pose.copy()
        current_pose = trackball.pose.copy()
        drag_pose[:3, 3] += offset
        current_pose[:3, 3] += offset
        trackball._pose = drag_pose
        trackball.set_camera_pose(current_pose)
        trackball._target = trackball._target + offset
        trackball._n_target = trackball._n_target + offset
        self.followed_xy = (x, y)


class ViewerDrive(ViewerFollow):
    motion_keys = frozenset((Key.W, Key.S, Key.A, Key.D))

    def __init__(
        self,
        *,
        enable_height=False,
        enable_reset=False,
        vx_accel=1.8,
        vx_decel=3.0,
        wz_accel=8.0,
        wz_decel=None,
        height_slope_speed=FIRMWARE_LEG_LENGTH_SLOPE_SPEED,
        initial_height=0.20,
    ):
        super().__init__()
        if enable_height:
            self.motion_keys = self.motion_keys | {Key.Z, Key.X}
        self.pressed = set()
        self.enable_reset = enable_reset
        self.reset_requested = False
        self.command_slope = KeyboardCommandSlope(
            vx_accel=vx_accel,
            vx_decel=vx_decel,
            wz_accel=wz_accel,
            wz_decel=wz_decel,
            height_slope_speed=height_slope_speed,
            initial_height=initial_height,
        )

    def reset_slopes(self, *, vx=0.0, wz=0.0, height=None):
        self.command_slope.reset(vx=vx, wz=wz, height=height)

    def ramped_commands(
        self,
        vx_target,
        wz_target,
        dt,
        *,
        vx_real=0.0,
        wz_real=0.0,
        height_target=None,
        height_real=0.0,
    ):
        return self.command_slope.calculate(
            vx_target,
            wz_target,
            dt,
            vx_real=vx_real,
            wz_real=wz_real,
            height_target=height_target,
            height_real=height_real,
        )

    def on_key_press(self, symbol, modifiers):
        if self.enable_reset and symbol == Key.R:
            self.pressed.clear()
            self.reset_requested = True
            return EVENT_HANDLED
        if symbol == Key.SPACE:
            self.pressed.clear()
            self.pressed.add(Key.SPACE)
            return EVENT_HANDLED
        if symbol in self.motion_keys:
            self.pressed.add(symbol)
            return EVENT_HANDLED

    def on_key_release(self, symbol, modifiers):
        if symbol in self.motion_keys or symbol == Key.SPACE:
            self.pressed.discard(symbol)
            return EVENT_HANDLED

    def on_deactivate(self):
        self.pressed.clear()

    def motion_axes(self):
        keys = self.pressed.copy()
        if Key.SPACE in keys:
            return 0, 0
        return (Key.W in keys) - (Key.S in keys), (Key.A in keys) - (Key.D in keys)

    def height_axis(self):
        keys = self.pressed.copy()
        return (Key.Z in keys) - (Key.X in keys)

    def take_reset_request(self):
        if not self.reset_requested:
            return False
        self.reset_requested = False
        return True
