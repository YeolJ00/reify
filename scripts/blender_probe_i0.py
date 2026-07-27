"""Render the initial frame (I0) of each PROBE for the Cosmos-generated matrix:

    drop     — ball A held above the table, ball B parked aside
    push     — both balls resting, A at the left with a clear path
    collide  — both resting, A at the left aimed straight at B

Two distinctly coloured balls so the tracker can tell them apart. Same real scene
(wooden table + city HDRI) and the same camera as the rest of the pipeline, so the
recovered sim can be projected back onto the generated video.

Run: CUDA_VISIBLE_DEVICES=<g> <blender> --background --python scripts/blender_probe_i0.py
Writes outputs/probes_i2v/I0_<probe>.png + scene.json
"""
import json
import math
from pathlib import Path

import bpy
import mathutils

REPO = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
OUT = REPO / "outputs" / "probes_i2v"
CITY = REPO / "assets" / "scenes" / "city"

CAM = {"eye": [0.72, -0.86, 1.03], "target": [0.02, 0.06, 0.78], "fov_deg": 52,
       "width": 544, "height": 448}
GROUND_Z = 0.706
R_BALL = 0.055
REST = GROUND_Z + R_BALL

# (posA, posB) per probe, in world coords
LAYOUTS = {
    "drop":    ([0.02, 0.02, 0.90], [0.30, 0.14, REST]),
    "push":    ([-0.19, 0.00, REST], [0.32, 0.20, REST]),
    "collide": ([-0.19, 0.00, REST], [0.07, 0.00, REST]),
}
COL_A = (0.85, 0.10, 0.08)      # red   — the driven ball
COL_B = (0.10, 0.22, 0.80)      # blue  — the target ball


def clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
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


def matte(obj, color, rough=0.85):
    m = bpy.data.materials.new("m"); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    obj.data.materials.clear(); obj.data.materials.append(m)


def add_ball(loc, color, name):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=R_BALL, location=loc, segments=48, ring_count=32)
    o = bpy.context.active_object; o.name = name
    bpy.ops.object.shade_smooth()
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = 0.32
    o.data.materials.clear(); o.data.materials.append(m)
    return o


def build_static():
    clear()
    world_hdri(str(CITY / "pretville_street_2k.hdr"))
    bpy.ops.mesh.primitive_plane_add(size=30); matte(bpy.context.active_object, (0.05, 0.05, 0.06), 0.95)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(REPO / "assets/scenes/wooden_table_02/wooden_table_02_2k.gltf"))
    tbl = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    for o in bpy.data.objects:
        o.select_set(False)
    for o in tbl:
        o.select_set(True)
    bpy.context.view_layer.objects.active = tbl[0]
    if len(tbl) > 1:
        bpy.ops.object.join()
    table = bpy.context.view_layer.objects.active
    bpy.context.view_layer.update()
    ztop = max((table.matrix_world @ v.co).z for v in table.data.vertices)
    table.location.z += GROUND_Z - ztop

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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build_static()
    for probe, (pa, pb) in LAYOUTS.items():
        for nm in ("ballA", "ballB"):
            if nm in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)
        add_ball(pa, COL_A, "ballA")
        add_ball(pb, COL_B, "ballB")
        bpy.context.scene.render.filepath = str(OUT / f"I0_{probe}.png")
        bpy.ops.render.render(write_still=True)
        print("rendered", probe)
    (OUT / "scene.json").write_text(json.dumps(
        {"camera": CAM, "ground_z": GROUND_Z, "ball_radius": R_BALL, "layouts": LAYOUTS,
         "color_A": COL_A, "color_B": COL_B}, indent=2))
    print("wrote", OUT / "scene.json")


main()
