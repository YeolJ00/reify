"""Render the drop scene photorealistically in Blender: the real scanned object
falling onto the real wooden table, city HDRI lighting + a prop, driven frame by
frame from the sim poses. Camera matches src/render/camera.py exactly.

Run: CUDA_VISIBLE_DEVICES=<g> <blender> --background --python scripts/blender_drop.py
"""
import json
import math
from pathlib import Path

import bpy
import mathutils
import numpy as np

REPO = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
import os
DROP = REPO / "outputs" / os.environ.get("SCENE", "drop")
CITY = REPO / "assets" / "scenes" / "city"


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


def matte(obj, color, rough=0.85, metal=0.0):
    m = bpy.data.materials.new("m"); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    obj.data.materials.clear(); obj.data.materials.append(m)


def main():
    scene_cfg = json.loads((DROP / "scene.json").read_text())
    cam_cfg = scene_cfg["camera"]
    ground_z = scene_cfg["ground_z"]
    poses = np.load(DROP / "poses.npz")
    pos, quat = poses["pos"], poses["quat"]   # quat is (x,y,z,w)

    clear()
    world_hdri(str(CITY / "pretville_street_2k.hdr"))

    # floor + real wooden table (top aligned to ground_z)
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
    table.location.z += ground_z - ztop      # tabletop -> ground_z

    place(import_gltf("street_lamp_01", "street_lamp_01"), (0.9, 0.7, 0.0), rot_z=0.5)
    place(import_gltf("fire_hydrant", "fire_hydrant"), (-0.7, 0.5, 0.0), rot_z=1.0)

    # the dropping object — textured (high-contrast noise) so LK has features to track
    bpy.ops.wm.obj_import(filepath=str(DROP / "object.obj"), up_axis="Z", forward_axis="Y")
    obj = bpy.context.selected_objects[0]
    obj.rotation_mode = "QUATERNION"
    mt = bpy.data.materials.new("obj"); mt.use_nodes = True
    nt = mt.node_tree; bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexNoise"); tex.inputs["Scale"].default_value = 22.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.55, 0.20, 0.12, 1)
    ramp.color_ramp.elements[1].color = (0.90, 0.82, 0.30, 1)
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    obj.data.materials.clear(); obj.data.materials.append(mt)
    for p in obj.data.polygons:
        p.use_smooth = True

    # camera matching src/render/camera.py (look-at, horizontal FOV = fov_deg)
    cam_d = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("c", cam_d)
    bpy.context.collection.objects.link(cam)
    cam.location = mathutils.Vector(cam_cfg["eye"])
    look = mathutils.Vector(cam_cfg["target"]) - cam.location
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    cam_d.sensor_fit = "HORIZONTAL"
    cam_d.angle = math.radians(cam_cfg["fov_deg"])
    bpy.context.scene.camera = cam

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.cycles.samples = 48
    sc.render.resolution_x = cam_cfg["width"]; sc.render.resolution_y = cam_cfg["height"]
    sc.render.image_settings.file_format = "PNG"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"; prefs.get_devices()
        for d in prefs.devices:
            d.use = (d.type == "CUDA")
        sc.cycles.device = "GPU"
    except Exception as e:
        print("GPU off:", e)

    fdir = DROP / "frames"; fdir.mkdir(exist_ok=True)
    for f in range(len(pos)):
        obj.location = mathutils.Vector(pos[f].tolist())
        q = quat[f]  # (x,y,z,w) -> blender (w,x,y,z)
        obj.rotation_quaternion = mathutils.Quaternion((float(q[3]), float(q[0]), float(q[1]), float(q[2])))
        sc.render.filepath = str(fdir / f"f{f:03d}.png")
        bpy.ops.render.render(write_still=True)
    print(f"rendered {len(pos)} frames -> {fdir}")


main()
