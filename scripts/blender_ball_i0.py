"""Render a single initial frame (I0) for the Wan i2v test: a bright bouncy ball
poised above the real wooden table in the city street scene. Same HDRI, table,
props and camera as blender_drop.py, so the recovered sim can reuse the camera.

Run: CUDA_VISIBLE_DEVICES=<g> <blender> --background --python scripts/blender_ball_i0.py
Writes outputs/ball/I0.png + ball.json (camera, ground_z, ball radius/center).
"""
import json
import math
import os
from pathlib import Path

import bpy
import mathutils

REPO = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
OUT = REPO / "outputs" / "ball"
CITY = REPO / "assets" / "scenes" / "city"

CAM = {"eye": [0.9, -0.9, 1.12], "target": [0.13, 0.0, 0.95], "fov_deg": 46, "width": 680, "height": 560}
GROUND_Z = 0.706
import os as _os
BALL_C = [float(x) for x in _os.environ.get("BALL_C", "0.13,0.0,0.98").split(",")]
BALL_R = float(_os.environ.get("BALL_R", "0.058"))


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


def import_gltf(folder, name):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(CITY / folder / f"{name}_2k.gltf"))
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


def place(obj, loc, rot_z=0.0):
    if obj is None:
        return
    obj.rotation_euler = (0, 0, rot_z)
    bpy.context.view_layer.update()
    zmin = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)
    obj.location = (loc[0], loc[1], loc[2] - zmin)


def matte(obj, color, rough=0.85):
    m = bpy.data.materials.new("m"); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    obj.data.materials.clear(); obj.data.materials.append(m)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clear()
    world_hdri(str(CITY / "pretville_street_2k.hdr"))

    bpy.ops.mesh.primitive_plane_add(size=30); floor = bpy.context.active_object
    matte(floor, (0.05, 0.05, 0.06), rough=0.95)
    tv_before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(REPO / "assets/scenes/wooden_table_02/wooden_table_02_2k.gltf"))
    tbl = [o for o in bpy.data.objects if o not in tv_before and o.type == "MESH"]
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

    place(import_gltf("street_lamp_01", "street_lamp_01"), (0.9, 0.7, 0.0), rot_z=0.5)
    place(import_gltf("fire_hydrant", "fire_hydrant"), (-0.7, 0.5, 0.0), rot_z=1.0)

    # the bouncy ball — glossy, saturated red-orange so it reads clearly and tracks by color
    bpy.ops.mesh.primitive_uv_sphere_add(radius=BALL_R, location=BALL_C, segments=48, ring_count=32)
    ball = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    m = bpy.data.materials.new("ball"); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.86, 0.18, 0.12, 1)
    b.inputs["Roughness"].default_value = 0.35
    try:
        b.inputs["Specular IOR Level"].default_value = 0.6
    except KeyError:
        pass
    ball.data.materials.clear(); ball.data.materials.append(m)

    cam_d = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("c", cam_d)
    bpy.context.collection.objects.link(cam)
    cam.location = mathutils.Vector(CAM["eye"])
    look = mathutils.Vector(CAM["target"]) - cam.location
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    cam_d.sensor_fit = "HORIZONTAL"
    cam_d.angle = math.radians(CAM["fov_deg"])
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

    sc.render.filepath = str(OUT / "I0.png")
    bpy.ops.render.render(write_still=True)
    (OUT / "ball.json").write_text(json.dumps(
        {"camera": CAM, "ground_z": GROUND_Z, "ball_center": BALL_C, "ball_radius": BALL_R}, indent=2))
    print("rendered", OUT / "I0.png")


main()
