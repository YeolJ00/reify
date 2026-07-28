"""Render one PROBE initial-frame per object of the authored scene.

For each target object we lift it above the table and leave the rest of the scene exactly
where it is. That single frame is what the video model is conditioned on, so the probe
video shows this object falling in its own scene — which is what makes the recovered
value belong to that object rather than to a stand-in.

A drop is used because it is the probe our matrix measured as the one that exposes
restitution (19.0 px of image effect, vs 0.1 px in a push).

Run: CUDA_VISIBLE_DEVICES=<g> <blender> --background --python scripts/blender_object_probes.py
Writes outputs/scene/probes/I0_<object>.png + probes.json
"""
import json
import math
import os
from pathlib import Path

import bpy
import mathutils

REPO = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
SCENE = REPO / "outputs" / "scene"
OUT = SCENE / "probes"
CITY = REPO / "assets" / "scenes" / "city"

# tighter camera than the hero shot: the object must be big in frame to be trackable
CAM = {"eye": [0.72, -0.80, 1.12], "target": [-0.02, 0.02, 0.80], "fov_deg": 46,
       "width": 544, "height": 448}
import os as _os
# ESCALATION (the planner's ladder): the first pass dropped only 16 cm and the generated
# clips barely rebounded, so restitution was weakly observable. Drop from twice as high;
# the camera above was re-framed to keep the lifted objects in shot.
DROP_H = float(_os.environ.get("DROP_H", "0.16"))
TARGETS = ["baseball", "apple", "brass_pot", "ceramic_vase", "rubber_duck"]


def clear():
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
    for c in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for b in list(c):
            c.remove(b)


def world_hdri(path):
    w = bpy.data.worlds.new("w"); bpy.context.scene.world = w; w.use_nodes = True
    nt = w.node_tree; nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(path)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def import_gltf(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new:
        return None
    for o in bpy.data.objects:
        o.select_set(False)
    for o in new:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new[0]
    if len(new) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SCENE / "scene.json").read_text())
    GZ = cfg["ground_z"]

    clear()
    world_hdri(str(CITY / "pretville_street_2k.hdr"))
    bpy.ops.mesh.primitive_plane_add(size=40)
    fl = bpy.context.active_object
    m = bpy.data.materials.new("floor"); m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (.05, .05, .06, 1)
    fl.data.materials.append(m)

    table = import_gltf(REPO / "assets/scenes/wooden_table_02/wooden_table_02_2k.gltf")
    bpy.context.view_layer.update()
    ztop = max((table.matrix_world @ v.co).z for v in table.data.vertices)
    table.location.z += GZ - ztop

    objs = {}
    for name, o in cfg["objects"].items():
        gl = list((REPO / "assets" / o["asset"]).glob("*.gltf"))
        if not gl:
            continue
        b = import_gltf(gl[0])
        if b is None:
            continue
        b.name = name
        b.scale = (o["scale"],) * 3
        b.rotation_euler = (b.rotation_euler[0], b.rotation_euler[1], o["rot_z"])
        b.location = tuple(o["pos"])
        objs[name] = b

    cam_d = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("c", cam_d)
    bpy.context.collection.objects.link(cam)
    cam.location = mathutils.Vector(CAM["eye"])
    look = mathutils.Vector(CAM["target"]) - cam.location
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    cam_d.sensor_fit = "HORIZONTAL"; cam_d.angle = math.radians(CAM["fov_deg"])
    bpy.context.scene.camera = cam

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.cycles.samples = 72
    sc.render.resolution_x = CAM["width"]; sc.render.resolution_y = CAM["height"]
    sc.render.image_settings.file_format = "PNG"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"; prefs.get_devices()
        for d in prefs.devices:
            d.use = (d.type == "CUDA")
        sc.cycles.device = "GPU"
    except Exception as e:
        print("GPU off:", e)

    probes = {}
    for name in TARGETS:
        if name not in objs:
            print("skip", name); continue
        o = objs[name]
        rest = tuple(cfg["objects"][name]["pos"])
        o.location = (rest[0], rest[1], rest[2] + DROP_H)      # lift it
        sc.render.filepath = str(OUT / f"I0_{name}.png")
        bpy.ops.render.render(write_still=True)
        probes[name] = {"rest_pos": list(rest), "drop_pos": list(o.location),
                        "drop_h": DROP_H, "asset": cfg["objects"][name]["asset"],
                        "scale": cfg["objects"][name]["scale"],
                        "size_cm": cfg["objects"][name]["size_cm"]}
        o.location = rest                                       # put it back
        print(f"  probe {name:13s} lifted to z={rest[2]+DROP_H:.3f}")

    (OUT / "probes.json").write_text(json.dumps(
        {"camera": CAM, "ground_z": GZ, "drop_h": DROP_H, "probes": probes}, indent=2))
    print("wrote", OUT / "probes.json")


main()
