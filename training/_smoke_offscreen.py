"""Throwaway probe: does Genesis offscreen camera rendering work here?"""
import sys
from pathlib import Path

import genesis as gs
import numpy as np


def main():
    gs.init(backend=gs.cuda, logging_level="warning")
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.01),
        renderer=gs.renderers.Rasterizer(),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0], background_color=(0.3, 0.4, 0.5)),
    )
    scene.add_entity(gs.morphs.Plane(visualization=True))
    box = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.3, 0.1), pos=(0.0, 0.0, 0.2), visualization=True),
        surface=gs.surfaces.Default(color=(0.9, 0.5, 0.2)),
    )
    cam = scene.add_camera(res=(320, 240), pos=(1.0, -1.0, 0.8), lookat=(0.0, 0.0, 0.2), fov=45)
    scene.build(n_envs=4, env_spacing=(1.5, 1.5))
    for _ in range(3):
        scene.step()
    rgb, depth, seg, normal = cam.render(rgb=True, depth=True)
    print("rgb:", None if rgb is None else (rgb.shape, rgb.dtype, int(rgb.min()), int(rgb.max())))
    print("depth:", None if depth is None else (depth.shape, depth.dtype, float(depth.min()), float(depth.max())))
    print("seg:", None if seg is None else (seg.shape, seg.dtype))
    print("normal:", None if normal is None else (normal.shape, normal.dtype))
    out = Path(__file__).with_name("_smoke_offscreen.png")
    if rgb is not None:
        from PIL import Image
        arr = rgb if rgb.dtype == np.uint8 else (rgb * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(out)
        print("saved", out, arr.shape)
    print("platform env:", __import__("os").environ.get("PYOPENGL_PLATFORM"))


if __name__ == "__main__":
    main()
