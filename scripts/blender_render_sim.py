"""Render the simulated rollout as actual frames, in the scene the clips were staged in.

This replaces the sprite compositing entirely. Same scene, same HDRI, same camera, same
Cycles settings as the initial frames -- so the simulated video differs from the generated
one only in what the object does, which is the whole point of putting them side by side.

Reads outputs/scene/expand/sim_poses.json (written by scripts/export_sim_poses.py) and
writes sim_<key>/f####.png.

Run: <blender> --background --python scripts/blender_render_sim.py
"""
import json
import math
from pathlib import Path

import bpy
import mathutils

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
SCENE = REPO / "outputs" / "scene"
import os
LAB = Path(os.environ.get("LAB") or (SCENE / "expand"))
CITY = REPO / "assets" / "scenes" / "city"
CAM = {"eye": [0.72, -0.80, 1.12], "target": [-0.02, 0.02, 0.80], "fov_deg": 46,
       "width": 544, "height": 448}
GROUND_Z = 0.706
PARK_Y = 0.46


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
    poses = json.loads((LAB / "sim_poses.json").read_text())
    cfg = json.loads((LAB / "lab.json").read_text())
    scene_cfg = json.loads((SCENE / "scene.json").read_text())
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
    for name, o in scene_cfg["objects"].items():
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
    # Quality is overridable so the CEM loop can render drafts. Defaults are unchanged,
    # so every existing caller renders exactly as before. Whether a draft is good enough
    # for the judge is a measured question, not an assumed one -- see j3_optimize.py.
    sc.render.engine = os.environ.get("ENGINE", "CYCLES")
    if sc.render.engine == "CYCLES":
        sc.cycles.samples = int(os.environ.get("SAMPLES", 48))
        if os.environ.get("NO_DENOISE"):
            sc.cycles.use_denoising = False
    scale = float(os.environ.get("RES_SCALE", 1.0))
    sc.render.resolution_x = int(CAM["width"] * scale)
    sc.render.resolution_y = int(CAM["height"] * scale)
    sc.render.image_settings.file_format = "PNG"
    if sc.render.engine == "CYCLES":
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "CUDA"; prefs.get_devices()
            for d in prefs.devices:
                d.use = (d.type == "CUDA")
            sc.cycles.device = "GPU"
        except Exception as e:
            print("GPU off:", e)

    for key, info in poses.items():
        subj = info["subject"]
        if subj not in objs:
            print(f"  {key}: no object"); continue
        # park everything else exactly as the staging did
        for n, o in objs.items():
            o.location = ((home[n][0], home[n][1] + PARK_Y, home[n][2])
                          if n != subj else home[n])
        s = objs[subj]
        s.rotation_mode = "QUATERNION"
        outdir = LAB / f"sim_{key}"
        outdir.mkdir(exist_ok=True)
        for t, p in enumerate(info["poses"]):
            M = mathutils.Matrix([[p["mat"][r][c] for c in range(3)] for r in range(3)])
            s.rotation_quaternion = M.to_quaternion()
            s.location = mathutils.Vector(p["loc"])
            bpy.context.view_layer.update()
            sc.render.filepath = str(outdir / f"f{t:04d}.png")
            bpy.ops.render.render(write_still=True)
        s.rotation_mode = "XYZ"
        s.rotation_euler = (s.rotation_euler[0], s.rotation_euler[1],
                            scene_cfg["objects"][subj]["rot_z"])
        print(f"  rendered {key}: {len(info['poses'])} frames -> {outdir.name}")


main()
