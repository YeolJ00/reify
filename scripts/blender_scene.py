"""Build the authored scene: a finished-looking tabletop that is physically dead.

Six props that read as ordinary scene dressing but must move completely differently —
that is the whole argument for recovering per-object values instead of guessing one
default. The brass pot and the ceramic vase are the pointed pair: same silhouette
family, one is dense metal and one is thin ceramic.

Renders the hero image and writes scene.json (camera, table height, and each object's
placement + mesh path) so the simulator and the USD writer can both consume it.

Run: CUDA_VISIBLE_DEVICES=<g> <blender> --background --python scripts/blender_scene.py
"""
import json
import math
from pathlib import Path

import bpy
import mathutils

REPO = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
OUT = REPO / "outputs" / "scene"
CITY = REPO / "assets" / "scenes" / "city"

CAM = {"eye": [0.92, -1.02, 1.22], "target": [-0.02, 0.02, 0.80], "fov_deg": 44,
       "width": 960, "height": 640}
GROUND_Z = 0.706

# name -> (asset folder, (x, y) on the table, z-rotation, scale)
PROPS = {
    "rubber_duck":  ("soft/rubber_duck_toy", (-0.30, 0.07), 0.9, 0.62),
    "brass_pot":    ("rigid/brass_pot_01", (0.28, 0.13), 0.0, 0.62),
    "ceramic_vase": ("rigid/ceramic_vase_01", (0.06, 0.20), 0.4, 0.62),
    "wooden_bowl":  ("rigid/wooden_bowl_01", (-0.04, -0.09), 0.0, 0.62),
    "baseball":     ("rigid/baseball_01", (0.26, -0.14), 0.0, 1.0),
    "apple":        ("rigid/food_apple_01", (-0.24, -0.18), 0.0, 1.0),
    "book":         ("rigid/book_encyclopedia_set_01", (-0.02, 0.06), 1.35, 0.42),
}


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
    bpy.context.view_layer.update()
    xs = [(table.matrix_world @ v.co).x for v in table.data.vertices]
    ys = [(table.matrix_world @ v.co).y for v in table.data.vertices]
    print(f"table top at z={GROUND_Z}, extent x [{min(xs):.2f},{max(xs):.2f}] y [{min(ys):.2f},{max(ys):.2f}]")

    placed = {}
    for name, (folder, (x, y), rz, sc) in PROPS.items():
        gl = list((REPO / "assets" / folder).glob("*.gltf"))
        if not gl:
            print("MISSING", folder); continue
        o = import_gltf(gl[0])
        if o is None:
            print("IMPORT FAIL", folder); continue
        o.name = name
        o.scale = (sc, sc, sc)
        o.rotation_euler = (o.rotation_euler[0], o.rotation_euler[1], rz)
        bpy.context.view_layer.update()
        zmin = min((o.matrix_world @ v.co).z for v in o.data.vertices)
        o.location = (x, y, GROUND_Z - zmin)
        bpy.context.view_layer.update()
        vs = [o.matrix_world @ v.co for v in o.data.vertices]
        lo = [min(v[i] for v in vs) for i in range(3)]
        hi = [max(v[i] for v in vs) for i in range(3)]
        placed[name] = {"asset": folder, "pos": [x, y, o.location.z], "rot_z": rz, "scale": sc,
                        "bbox_min": lo, "bbox_max": hi,
                        "size_cm": [(hi[i] - lo[i]) * 100 for i in range(3)]}
        print(f"  placed {name:13s} size {placed[name]['size_cm'][0]:5.1f} x "
              f"{placed[name]['size_cm'][1]:5.1f} x {placed[name]['size_cm'][2]:5.1f} cm")

    cam_d = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("c", cam_d)
    bpy.context.collection.objects.link(cam)
    cam.location = mathutils.Vector(CAM["eye"])
    look = mathutils.Vector(CAM["target"]) - cam.location
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    cam_d.sensor_fit = "HORIZONTAL"; cam_d.angle = math.radians(CAM["fov_deg"])
    bpy.context.scene.camera = cam

    sc_ = bpy.context.scene
    sc_.render.engine = "CYCLES"; sc_.cycles.samples = 96
    sc_.render.resolution_x = CAM["width"]; sc_.render.resolution_y = CAM["height"]
    sc_.render.image_settings.file_format = "PNG"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"; prefs.get_devices()
        for d in prefs.devices:
            d.use = (d.type == "CUDA")
        sc_.cycles.device = "GPU"
    except Exception as e:
        print("GPU off:", e)

    import os
    if not os.environ.get("SKIP_RENDER"):
        sc_.render.filepath = str(OUT / "hero.png")
        bpy.ops.render.render(write_still=True)
    # geometry export — the physics writer layers UsdPhysics attributes onto this stage
    try:
        bpy.ops.wm.usd_export(filepath=str(OUT / "scene_geom.usdc"), export_materials=True,
                              export_textures=False)
        print("exported", OUT / "scene_geom.usdc")
    except Exception as e:
        print("USD export failed:", e)
    (OUT / "scene.json").write_text(json.dumps(
        {"camera": CAM, "ground_z": GROUND_Z, "objects": placed}, indent=2))
    print("wrote", OUT / "hero.png", "and scene.json")


main()
