"""Draw Genesis states with MuJoCo's off-screen renderer."""

from contextlib import closing
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def _visual_model(model_path, terrain=None):
    xml = ET.parse(model_path).getroot()
    compiler = xml.find("compiler")
    if compiler is not None and "meshdir" in compiler.attrib:
        compiler.set("meshdir", (model_path.resolve().parent / compiler.attrib["meshdir"]).as_posix())
    assets = xml.find("asset")
    if assets is None:
        assets = ET.SubElement(xml, "asset")
    ET.SubElement(assets, "texture", name="replay_tiles", type="2d", builtin="checker", width="256", height="256",
                  rgb1="0.19 0.22 0.25", rgb2="0.23 0.27 0.30")
    ET.SubElement(assets, "material", name="replay_ground", texture="replay_tiles", texrepeat="3 3", texuniform="true")
    world = xml.find("worldbody")
    for geom in world.findall("geom[@type='plane']"):
        world.remove(geom)
    if terrain is None:
        ET.SubElement(world, "geom", type="plane", size="0 0 0.15", material="replay_ground")
    else:
        field = terrain.field.T * terrain.vertical_scale
        rows, columns = field.shape
        half_x = (columns - 1) * terrain.spacing * 0.5
        half_y = (rows - 1) * terrain.spacing * 0.5
        low, high = float(field.min()), float(field.max())
        height = max(high - low, terrain.vertical_scale)
        ET.SubElement(assets, "hfield", name="replay_terrain", nrow=str(rows), ncol=str(columns),
                      size=f"{half_x} {half_y} {height} 0.1")
        ET.SubElement(world, "geom", type="hfield", hfield="replay_terrain", pos=f"0 {half_y} {low}",
                      material="replay_ground")
        if hasattr(terrain, "deck_boxes"):
            for center, half_size in terrain.deck_boxes():
                ET.SubElement(world, "geom", type="box", pos=" ".join(map(str, center)),
                              size=" ".join(map(str, half_size)), material="replay_ground")
    ET.SubElement(world, "light", pos="1 -1 3", dir="-0.3 0.3 -1", diffuse="0.85 0.85 0.85")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    if terrain is not None:
        start = model.hfield("replay_terrain").adr[0]
        model.hfield_data[start:start + field.size] = ((field - low) / height).ravel()
        # MuJoCo scales clip distances by the whole course's extent, including distant lanes.
        model.vis.map.znear = min(model.vis.map.znear, 0.05 / model.stat.extent)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    return model


class RobotRenderer:
    def __init__(self, model_path, robot, *, measure_closure=True, terrain=None):
        self.model = model = _visual_model(model_path, terrain)
        self.data = mujoco.MjData(model)
        self.measure_closure = measure_closure
        self.end_pose = np.empty(model.nq)
        self.pose_delta = np.empty(model.nv)
        self.target_qpos = []
        self.source_qpos = []
        for index in range(model.njnt):
            name = model.joint(index).name or model.body(model.jnt_bodyid[index]).name
            joint = robot.get_joint(name)
            start = model.jnt_qposadr[index]
            self.target_qpos.extend(range(start, start + joint.n_qs))
            self.source_qpos.extend(joint.qs_idx_local)
        if sorted(self.target_qpos) != list(range(model.nq)):
            raise ValueError("Genesis and rendering model have different joint coordinates")
        self.base = model.body("base_link").id
        self.camera = mujoco.MjvCamera()
        self.home_distance = 1.5 if model.nmesh else 1.2
        self.camera.distance = self.home_distance
        self.camera.azimuth = 111
        self.camera.elevation = -15
        self.max_error = 0.0
        self.visual_options = mujoco.MjvOption()
        self.visual_options.sitegroup[:] = 0
        if not measure_closure:
            model.vis.quality.offsamples = 0
        self.renderer = mujoco.Renderer(model, width=640, height=384, max_geom=model.ngeom + 100)
        if not measure_closure:
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False

    def set_orbit(self, azimuth, elevation, zoom):
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation
        self.camera.distance = self.home_distance * zoom

    def render(self, qpos, *, previous=None, fraction=1.0):
        model, data = self.model, self.data
        if previous is None or fraction >= 1.0:
            data.qpos[self.target_qpos] = qpos[self.source_qpos]
        else:
            data.qpos[self.target_qpos] = previous[self.source_qpos]
            self.end_pose[self.target_qpos] = qpos[self.source_qpos]
            mujoco.mj_differentiatePos(model, self.pose_delta, 1.0, data.qpos, self.end_pose)
            mujoco.mj_integratePos(model, data.qpos, self.pose_delta, fraction)
        if self.measure_closure:
            mujoco.mj_forward(model, data)
            delta = data.site_xpos[model.eq_obj1id] - data.site_xpos[model.eq_obj2id]
            self.max_error = max(self.max_error, np.linalg.norm(delta, axis=-1).max())
        else:
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_camlight(model, data)
        self.camera.lookat[:] = data.xpos[self.base]
        if model.nmesh:
            self.camera.lookat[2] += 0.10
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.visual_options)
        return Image.fromarray(self.renderer.render())

    def close(self):
        self.renderer.close()


def record(model_path, robot, frames, output, fps, label, controller, *, terrain=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    with (closing(RobotRenderer(model_path, robot, terrain=terrain)) as renderer,
          imageio.get_writer(output, fps=fps, codec="libx264", pixelformat="yuv420p",
                             ffmpeg_params=["-movflags", "+faststart"]) as writer):
        for qpos, seconds, target, velocity in frames:
            frame = renderer.render(qpos)
            draw = ImageDraw.Draw(frame)
            draw.rounded_rectangle((14, 14, 256, 72), radius=6, fill=(19, 27, 34))
            draw.text((25, 24), f"{label} | {controller}", fill=(228, 244, 250))
            draw.text((25, 46), f"{seconds:4.2f} s  vx {velocity:+.2f} / {target:+.2f} m/s", fill=(161, 208, 227))
            writer.append_data(np.asarray(frame))
    print(f"Replay saved to {output.resolve()}. Peak closure gap in frames: {renderer.max_error * 1000:.2f} mm")
