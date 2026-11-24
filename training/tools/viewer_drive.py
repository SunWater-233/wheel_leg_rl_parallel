"""Keyboard driving and camera following in the native Genesis viewer."""

import numpy as np

from genesis.vis.keybindings import Key
from genesis.vis.viewer_plugins import EVENT_HANDLED, ViewerPlugin


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

    def __init__(self, *, enable_height=False, enable_reset=False):
        super().__init__()
        if enable_height:
            self.motion_keys = self.motion_keys | {Key.Z, Key.X}
        self.pressed = set()
        self.enable_reset = enable_reset
        self.reset_requested = False

    def on_key_press(self, symbol, modifiers):
        if self.enable_reset and symbol == Key.R:
            self.pressed.clear()
            self.reset_requested = True
            return EVENT_HANDLED
        if symbol == Key.SPACE:
            self.pressed.clear()
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
        return (Key.W in keys) - (Key.S in keys), (Key.A in keys) - (Key.D in keys)

    def height_axis(self):
        keys = self.pressed.copy()
        return (Key.Z in keys) - (Key.X in keys)

    def take_reset_request(self):
        if not self.reset_requested:
            return False
        self.reset_requested = False
        return True
