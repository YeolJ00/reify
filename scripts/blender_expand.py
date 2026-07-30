"""Stage the expanded lab: every object, drops at three heights, plus a slide.

Two changes from the previous lab, both driven by measurement rather than taste:

  * COLLIDE IS DROPPED. Of the 50 takes that moved but yielded no observable, 42 were
    collisions (20 "mover was not approaching", 17 "target never moved"). Collide yields
    ~24% against ~80% for drop and slide, so the same GPU hours buy three times the data
    on the probes that work.
  * DROPS AT THREE HEIGHTS. Restitution is not one number: e generally falls as impact
    speed rises. Three heights turn the best-yielding probe into a measurement of e(v),
    which is a new parameter rather than a repeat of an old one, and simultaneously
    triples the samples behind each object's e.

All seven scene objects are included; the wooden bowl and the book have never been probed.

Run: <blender> --background --python scripts/blender_expand.py
"""
import json
import math
from pathlib import Path

import bpy
import mathutils

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
SCENE = REPO / "outputs" / "scene"
OUT = SCENE / "expand"
CITY = REPO / "assets" / "scenes" / "city"

CAM = {"eye": [0.72, -0.80, 1.12], "target": [-0.02, 0.02, 0.80], "fov_deg": 46,
       "width": 544, "height": 448}
GROUND_Z = 0.706
LANE_Y = -0.02
START_X = -0.30
PARK_Y = 0.46
# impact speed goes as sqrt(2gh), so these span roughly 1.25 m/s to 2.4 m/s -- a factor
# of ~2 in speed, which is what makes a slope in e(v) resolvable at all
HEIGHTS = {"low": 0.08, "mid": 0.18, "high": 0.30}


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
    table.location.z += GROUND_Z - ztop

    objs, home = {}, {}
    for name, o in cfg["objects"].items():
        gl = list((REPO / "assets" / o["asset"]).glob("*.gltf"))
        if not gl:
            print("no gltf for", name); continue
        b = import_gltf(gl[0])
        if b is None:
            continue
        b.name = name; b.scale = (o["scale"],) * 3
        b.rotation_euler = (b.rotation_euler[0], b.rotation_euler[1], o["rot_z"])
        b.location = tuple(o["pos"]); objs[name] = b; home[name] = tuple(o["pos"])

    cam_d = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("c", cam_d)
    bpy.context.collection.objects.link(cam)
    cam.location = mathutils.Vector(CAM["eye"])
    look = mathutils.Vector(CAM["target"]) - cam.location
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    cam_d.sensor_fit = "HORIZONTAL"; cam_d.angle = math.radians(CAM["fov_deg"])
    bpy.context.scene.camera = cam

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.cycles.samples = 64
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

    exps = {}
    for subj in sorted(objs):
        for n, o in objs.items():
            if n != subj:
                o.location = (home[n][0], home[n][1] + PARK_Y, home[n][2])
        s = objs[subj]
        for hname, h in HEIGHTS.items():
            s.location = (START_X, LANE_Y, home[subj][2] + h)
            bpy.context.view_layer.update()
            key = f"{subj}_drop_{hname}"
            sc.render.filepath = str(OUT / f"I0_{key}.png")
            bpy.ops.render.render(write_still=True)
            exps[key] = {"subject": subj, "kind": "drop", "partner": None,
                         "subject_pos": list(s.location), "partner_pos": None,
                         "lift": h, "height_name": hname, "height_m": h}
            print(f"  staged {key}")
        s.location = (START_X, LANE_Y, home[subj][2])
        bpy.context.view_layer.update()
        key = f"{subj}_slide"
        sc.render.filepath = str(OUT / f"I0_{key}.png")
        bpy.ops.render.render(write_still=True)
        exps[key] = {"subject": subj, "kind": "slide", "partner": None,
                     "subject_pos": list(s.location), "partner_pos": None,
                     "lift": 0.0, "height_name": None, "height_m": 0.0}
        print(f"  staged {key}")
        for n in objs:
            objs[n].location = home[n]

    (OUT / "lab.json").write_text(json.dumps(
        {"camera": CAM, "ground_z": GROUND_Z, "reference": None,
         "heights": HEIGHTS, "experiments": exps, "assets": cfg["objects"]}, indent=2))
    print(f"wrote {OUT}/lab.json  ({len(exps)} experiments)")


main()
