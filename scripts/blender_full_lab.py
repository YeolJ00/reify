"""Stage the full lab: every experiment needed to determine every object's physics.

Per object we need three different excitations, because no single one reveals everything:
    <obj>_drop      it falls              -> restitution
    <obj>_slide     it travels            -> friction
    <obj>_collide   it strikes the REFERENCE object -> mass ratio

Every collision uses the same reference object (the baseball), so all the recovered mass
ratios land on one common scale and the scene's objects become comparable to each other —
which a set of unrelated pairwise collisions would not give.

The reference itself is measured against a second object so it is not the odd one out.

Run: CUDA_VISIBLE_DEVICES=<g> <blender> --background --python scripts/blender_full_lab.py
"""
import json
import math
from pathlib import Path

import bpy
import mathutils

REPO = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
SCENE = REPO / "outputs" / "scene"
OUT = SCENE / "fulllab"
CITY = REPO / "assets" / "scenes" / "city"

CAM = {"eye": [0.72, -0.80, 1.12], "target": [-0.02, 0.02, 0.80], "fov_deg": 46,
       "width": 544, "height": 448}
GROUND_Z = 0.706
REFERENCE = "baseball"                      # the common partner for every collision
SUBJECTS = ["apple", "rubber_duck", "brass_pot", "ceramic_vase"]
DROP_H = 0.18
LANE_Y = -0.02                              # the clear runway everything is staged on
START_X, TARGET_X = -0.34, -0.12            # ~11 cm surface gap: they actually meet
PARK_Y = 0.46                               # everything else pushed back out of the lane


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
            continue
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

    def park_all_except(keep):
        for n, o in objs.items():
            if n in keep:
                continue
            o.location = (home[n][0], home[n][1] + PARK_Y, home[n][2])

    exps = {}
    subjects = SUBJECTS + [REFERENCE]
    for subj in subjects:
        if subj not in objs:
            print("missing", subj); continue
        partner = REFERENCE if subj != REFERENCE else "apple"
        for kind in ("drop", "slide", "collide"):
            keep = {subj} | ({partner} if kind == "collide" else set())
            park_all_except(keep)
            s = objs[subj]
            if kind == "drop":
                s.location = (START_X + 0.06, LANE_Y, home[subj][2] + DROP_H)
                tgt_pos = None
            else:
                s.location = (START_X, LANE_Y, home[subj][2])
                if kind == "collide":
                    p = objs[partner]
                    p.location = (TARGET_X, LANE_Y, home[partner][2])
                    tgt_pos = list(p.location)
                else:
                    tgt_pos = None
            bpy.context.view_layer.update()
            key = f"{subj}_{kind}"
            sc.render.filepath = str(OUT / f"I0_{key}.png")
            bpy.ops.render.render(write_still=True)
            exps[key] = {"subject": subj, "kind": kind,
                         "partner": partner if kind == "collide" else None,
                         "subject_pos": list(s.location), "partner_pos": tgt_pos,
                         "lift": DROP_H if kind == "drop" else 0.0}
            print(f"  staged {key}")
        for n in objs:
            objs[n].location = home[n]

    (OUT / "lab.json").write_text(json.dumps(
        {"camera": CAM, "ground_z": GROUND_Z, "reference": REFERENCE,
         "experiments": exps, "assets": cfg["objects"]}, indent=2))
    print(f"wrote {OUT}/lab.json  ({len(exps)} experiments)")


main()
