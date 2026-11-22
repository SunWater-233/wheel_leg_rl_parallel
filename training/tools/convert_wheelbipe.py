"""Convert the SCUTRobotLab Wheelbipe V14.2 physics asset to MJCF."""

import argparse
from collections import defaultdict
from math import isfinite, radians
from pathlib import Path
import xml.etree.ElementTree as ET

from pxr import Gf, Usd, UsdGeom, UsdPhysics


TRAINING_DIR = Path(__file__).resolve().parents[1]
SOURCE_USD = (
    TRAINING_DIR.parents[1] / "Ref_Git/wheeled-legged_RL/source/agent_world/agent_world"
    / "assets/usd_files/wheelbipeV14_2_1/wheelbipeV14_2.usd"
)
MODEL_PATH = TRAINING_DIR / "assets/wheelbipe_v14_2.xml"
AXES = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}


def numbers(values):
    return " ".join(f"{float(value):.10g}" for value in values)


def quaternion(quat):
    return numbers((quat.GetReal(), *quat.GetImaginary()))


def add_collider(body, prim, transform):
    # USD's row-vector transforms can carry box dimensions as a nonuniform scale.
    rows = [transform.GetRow3(i) for i in range(3)]
    scales = [row.GetLength() for row in rows]
    rotation = Gf.Matrix3d(*(value for row, scale in zip(rows, scales) for value in row / scale))
    quat = rotation.ExtractRotation().GetQuat()
    if prim.IsA(UsdGeom.Cube):
        one_half = 0.5 * UsdGeom.Cube(prim).GetSizeAttr().Get()
        size = [one_half * scale for scale in scales]
        kind = "box"
    elif prim.IsA(UsdGeom.Cylinder):
        cylinder = UsdGeom.Cylinder(prim)
        axis = cylinder.GetAxisAttr().Get()
        axis_idx = "XYZ".index(axis)
        radial_scales = [scale for i, scale in enumerate(scales) if i != axis_idx]
        if abs(radial_scales[0] - radial_scales[1]) > 1e-6:
            raise ValueError(f"Noncircular cylinder {prim.GetPath()}")
        size = [cylinder.GetRadiusAttr().Get() * radial_scales[0],
                0.5 * cylinder.GetHeightAttr().Get() * scales[axis_idx]]
        quat *= Gf.Rotation(AXES["Z"], AXES[axis]).GetQuat()
        kind = "cylinder"
    else:
        raise ValueError(f"Unsupported collider {prim.GetPath()}")

    name = body.attrib["name"]
    color = "0.27 0.39 0.48 1"
    if "wheel" in name or "guide" in name:
        color = "0.16 0.19 0.22 1"
    elif "front" in name or "rear" in name:
        color = "0.23 0.59 0.71 1"
    ET.SubElement(body, "geom", type=kind, size=numbers(size), quat=quaternion(quat),
                  pos=numbers(transform.ExtractTranslation()), rgba=color)


def convert(source: Path, output: Path):
    stage = Usd.Stage.Open(str(source.resolve(strict=True)))
    if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("Expected the V14.2 Z-up USD in meters")
    prims = list(stage.Traverse(Usd.TraverseInstanceProxies()))
    transforms = UsdGeom.XformCache()
    links = {prim.GetPath(): prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    children = defaultdict(list)
    incoming = set()
    closing_joints = []
    colliders = defaultdict(list)
    for prim in prims:
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            if not joint.GetJointEnabledAttr().Get():
                continue
            (parent,), (child,) = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if parent not in links or child not in links:
                raise ValueError(f"Missing rigid body for {prim.GetPath()}")
            if joint.GetExcludeFromArticulationAttr().Get():
                closing_joints.append((joint, parent, child))
            else:
                if child in incoming:
                    raise ValueError(f"Multiple tree parents for {child}")
                incoming.add(child)
                children[parent].append((child, joint))
        elif prim.IsA(UsdGeom.Gprim) and prim.HasAPI(UsdPhysics.CollisionAPI):
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                ancestor = prim.GetParent()
                while ancestor.IsValid() and ancestor.GetPath() not in links:
                    ancestor = ancestor.GetParent()
                if ancestor.IsValid():
                    colliders[ancestor.GetPath()].append(prim)

    roots = set(links) - incoming
    if len(roots) != 1 or len(closing_joints) != 6:
        raise ValueError(f"Expected one root and six closure joints, found {roots}, {len(closing_joints)}")
    model = ET.Element("mujoco", model="wheelbipe_v14_2")
    model.append(ET.Comment(" Physics geometry derived from SCUTRobotLab/wheeled-legged_RL (MIT). "))
    ET.SubElement(model, "compiler", angle="radian", inertiafromgeom="false", fusestatic="false")
    ET.SubElement(model, "option", timestep="0.005")
    world = ET.SubElement(model, "worldbody")
    bodies = {}

    def append_body(path, parent_node, world_to_parent, joint=None):
        prim = links[path]
        to_world = transforms.GetLocalToWorldTransform(prim)
        relative = to_world * world_to_parent
        body = ET.SubElement(parent_node, "body", name=prim.GetName(),
                             pos=numbers(relative.ExtractTranslation()),
                             quat=quaternion(relative.ExtractRotationQuat()))
        bodies[path] = body
        mass = UsdPhysics.MassAPI(prim)
        ET.SubElement(body, "inertial", pos=numbers(mass.GetCenterOfMassAttr().Get()),
                      quat=quaternion(mass.GetPrincipalAxesAttr().Get()),
                      mass=numbers([mass.GetMassAttr().Get()]),
                      diaginertia=numbers(mass.GetDiagonalInertiaAttr().Get()))
        if joint is None:
            ET.SubElement(body, "freejoint", name="base_free")
        else:
            prim_joint = joint.GetPrim()
            if prim_joint.IsA(UsdPhysics.RevoluteJoint):
                kind, scale = "hinge", radians(1)
            elif prim_joint.IsA(UsdPhysics.PrismaticJoint):
                kind, scale = "slide", 1.0
            else:
                raise ValueError(f"Unsupported tree joint {prim_joint.GetPath()}")
            axis = AXES[prim_joint.GetAttribute("physics:axis").Get()]
            local_quat = joint.GetLocalRot1Attr().Get()
            axis = Gf.Rotation(Gf.Quatd(local_quat)).TransformDir(axis)
            limits = [prim_joint.GetAttribute(f"physics:{side}Limit").Get() for side in ("lower", "upper")]
            attrs = {}
            if all(isfinite(limit) for limit in limits):
                attrs = {"limited": "true", "range": numbers(limit * scale for limit in limits)}
            ET.SubElement(body, "joint", name=prim_joint.GetName(), type=kind,
                          pos=numbers(joint.GetLocalPos1Attr().Get()), axis=numbers(axis), **attrs)

        world_to_link = to_world.GetInverse()
        for collider in colliders[path]:
            add_collider(body, collider, transforms.GetLocalToWorldTransform(collider) * world_to_link)
        for child_path, child_joint in children[path]:
            append_body(child_path, body, world_to_link, child_joint)

    append_body(roots.pop(), world, Gf.Matrix4d(1))
    if len(bodies) != len(links):
        raise ValueError("The USD has links outside the articulation tree")

    equality = ET.SubElement(model, "equality")
    for joint, parent, child in closing_joints:
        if not joint.GetPrim().IsA(UsdPhysics.SphericalJoint):
            raise ValueError(f"Expected a spherical closure joint at {joint.GetPath()}")
        name = joint.GetPrim().GetName()
        # Site pairs retain both authored anchors, including small assembly offsets.
        site1, site2 = f"{name}_a", f"{name}_b"
        ET.SubElement(bodies[parent], "site", name=site1, pos=numbers(joint.GetLocalPos0Attr().Get()))
        ET.SubElement(bodies[child], "site", name=site2, pos=numbers(joint.GetLocalPos1Attr().Get()))
        # The preloaded sliders deflect the default soft equality notably even at rest.
        ET.SubElement(equality, "connect", name=name, site1=site1, site2=site2,
                      solref="0.01 1", solimp="0.999 0.999 0.001 0.5 2")

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(model, space="  ")
    ET.ElementTree(model).write(output, encoding="utf-8", xml_declaration=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=SOURCE_USD)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    print(convert(args.usd, args.output))


if __name__ == "__main__":
    main()
